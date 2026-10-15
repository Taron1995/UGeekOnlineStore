"""Per-file processing: decode (sampled) -> motion gate -> detect -> track -> zones/lines/machine -> events."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from .storage import write_shard
from .tracker import Tracker
from .zones import Line, MachineState, Motion, Zone

DEFAULTS = dict(fps=3.0, conf=0.35, min_hits=2, iou_thr=0.2, max_lost_s=2.0, gate_motion=0.0005, keepalive_s=3.0,
                clips=True, max_clips_per_file=10, clip_pad_s=5.0, active_hours=None, hw_decode=False)


def fmt_ts(ts: float, f: str = "%Y%m%d_%H%M%S") -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime(f)


def active_intervals(start_ts: float, duration: float, hours):
    """File-relative [a, b) second ranges that fall inside the daily hour ranges, e.g. [[8, 20]] or [[20, 6]]."""
    if not hours:
        return [(0.0, duration)]
    ranges = []
    for h1, h2 in hours:
        ranges += [(h1, 24), (0, h2)] if h2 <= h1 else [(h1, h2)]
    end_ts, out = start_ts + duration, []
    for day in range(int(start_ts // 86400), int(end_ts // 86400) + 1):
        for h1, h2 in ranges:
            a, b = max(start_ts, day * 86400 + h1 * 3600), min(end_ts, day * 86400 + h2 * 3600)
            if b > a:
                out.append((a - start_ts, b - start_ts))
    merged = []
    for a, b in sorted(out):  # e.g. 22-24 and 00-06 touch at midnight: one pass, no tracker reset
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def open_capture(path: str, hw: bool = False):
    if hw:  # D3D11VA/VAAPI if the OpenCV build has it; silently falls back to software
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(path)


def save_clip(path: str, t_rel: float, out_path: Path, pad: float, hw: bool = False):
    cap = open_capture(path, hw)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 120 else 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_rel - pad) * 1000)
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(int(2 * pad * fps)):
        ok, f = cap.read()
        if not ok:
            break
        out.write(f)
    out.release()
    cap.release()


def process_video(src: dict, cam: dict, opts: dict, detect, out_dir: Path | None = None, limit_s: float | None = None):
    """src: {camera, path, rel, start_ts, duration}. detect(frame) -> [(x1,y1,x2,y2,score)].
    Returns (events, minutes, meta). events: (ts, type, subject, value, clip, thumb)."""
    t0 = time.time()
    cap = open_capture(src["path"], opts["hw_decode"])
    if not cap.isOpened():
        raise OSError(f"cannot open {src['path']}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if 1 <= src_fps <= 120 else 25.0
    step = max(1, round(src_fps / opts["fps"]))
    dt = step / src_fps
    zones = [Zone(z["name"], z["polygon"], z.get("kind", "workstation")) for z in cam.get("zones") or []]
    mcfg = cam.get("machine")
    roi = Zone("machine", mcfg["polygon"]) if mcfg else None
    events, minutes, n_clips = [], {}, 0
    st = dict(frames=0, det_frames=0, processed_s=0.0)

    for a, b in active_intervals(src["start_ts"], src["duration"], opts["active_hours"]):
        if limit_s is not None and st["processed_s"] >= limit_s:
            break
        if a > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, a * 1000.0)
        fi = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        lines = [Line(l["name"], l["p1"], l["p2"]) for l in cam.get("lines") or []]  # fresh per pass: track ids restart
        motion = Motion()
        tracker = Tracker(opts["iou_thr"], max(2, int(opts["max_lost_s"] * opts["fps"])), opts["min_hits"])
        mstate = MachineState(**{k: mcfg[k] for k in ("motion_thr", "window_s", "on_frac", "off_frac") if k in mcfg}) \
            if roi else None
        seen, in_zone, last_det, t_rel = set(), {}, -1e9, a
        warm_until = a + (mstate.window_s if mstate else 0)  # machine state is not trusted until its window has filled
        while limit_s is None or st["processed_s"] < limit_s:
            if not cap.grab():
                break
            fi += 1
            if (fi - 1) % step:
                continue
            t_rel = (fi - 1) / src_fps
            if t_rel >= b:
                break
            ok, frame = cap.retrieve()
            if not ok:
                continue
            h, w = frame.shape[:2]
            abs_t = src["start_ts"] + t_rel
            fm = motion.step(frame)
            detected = 0
            if fm > opts["gate_motion"] or tracker.tracks or t_rel - last_det >= opts["keepalive_s"]:
                tracks = tracker.update(detect(frame))
                last_det, detected = t_rel, 1
            else:
                tracks = []  # quiet scene and nobody tracked: skip the expensive detector
            for tr in tracks:
                if tr.id not in seen:
                    seen.add(tr.id)
                    events.append((abs_t, "person_appeared", f"t{tr.id}", "", "", ""))
                for z in zones:
                    key, inside = (tr.id, z.name), z.contains(*tr.pt, w, h)
                    if inside and key not in in_zone:
                        in_zone[key] = t_rel
                        events.append((abs_t, "zone_enter", f"t{tr.id}", z.name, "", ""))
                    elif not inside and key in in_zone:
                        events.append((abs_t, "zone_exit", f"t{tr.id}", f"{z.name} dwell={t_rel - in_zone.pop(key):.0f}s", "", ""))
                for ln in lines:
                    d = ln.update(tr.id, tr.pt, w, h)
                    if d:
                        events.append((abs_t, "line_cross", f"t{tr.id}", f"{ln.name}:{d}", "", ""))
            alive = {t.id for t in tracker.tracks}
            for key in [k for k in in_zone if k[0] not in alive]:
                events.append((abs_t, "zone_exit", f"t{key[0]}", f"{key[1]} dwell={t_rel - in_zone.pop(key):.0f}s", "", ""))

            state = None
            if mstate:
                ratio = motion.roi_ratio(roi, [t.box for t in tracks], (h, w))
                state, changed = mstate.update(ratio, t_rel)
                if t_rel < warm_until:
                    state, changed = None, False
                if changed:
                    clip = thumb = ""
                    if opts["clips"] and out_dir is not None and n_clips < opts["max_clips_per_file"]:
                        n_clips += 1
                        base = f"clips/{src['camera']}/{fmt_ts(abs_t)}_{state}"
                        (out_dir / base).parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(out_dir / (base + ".jpg")), cv2.resize(frame, (480, max(1, int(h * 480 / w)))))
                        save_clip(src["path"], t_rel, out_dir / (base + ".mp4"), opts["clip_pad_s"], opts["hw_decode"])
                        clip, thumb = base + ".mp4", base + ".jpg"
                    events.append((abs_t, "machine_state", "machine", state, clip, thumb))
            row = minutes.setdefault(int(abs_t // 60), [0.0, 0.0, 0, 0.0, 0.0, 0])
            row[0] += dt
            row[1] += dt * len(tracks)
            row[2] = max(row[2], len(tracks))
            if state:
                row[3 if state == "work" else 4] += dt
            row[5] += detected
            st["frames"] += 1
            st["det_frames"] += detected
            st["processed_s"] += dt
        for key in list(in_zone):
            events.append((src["start_ts"] + t_rel, "zone_exit", f"t{key[0]}", f"{key[1]} dwell={t_rel - in_zone.pop(key):.0f}s", "", ""))
    cap.release()
    meta = dict(camera=src["camera"], source=src["rel"], start_ts=src["start_ts"], video_s=src["duration"],
                processed_s=round(st["processed_s"], 1), frames=st["frames"], det_frames=st["det_frames"],
                elapsed_s=round(time.time() - t0, 2))
    return events, minutes, meta


_DET = None


def init_worker(model: str, conf: float):
    global _DET
    from .detector import PersonDetector
    _DET = PersonDetector(model, conf=conf, threads=1)


def run_task(task: dict) -> dict:
    """Pool entry point. Never raises: a corrupt file must not stop a multi-day run."""
    try:
        events, minutes, meta = process_video(task["src"], task["cam"], task["opts"], _DET.detect, Path(task["out"]))
        write_shard(Path(task["shard"]), events, minutes, meta)
        return dict(ok=True, **meta)
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, source=task["src"]["rel"], error=repr(e))
