"""Per-file processing: decode (sampled) -> motion gate -> detect -> track -> analyzers -> events and per-minute tables."""
from __future__ import annotations

import time
from pathlib import Path

import cv2

from .analyzers import Ctx, build_analyzers
from .config import DEFAULTS, normalize_camera
from .storage import write_shard
from .tracker import Tracker
from .video import active_intervals, fmt_ts, open_capture, save_clip  # noqa: F401  (re-exported)
from .zones import Motion, Zone


def process_video(src: dict, cam: dict, opts: dict, detect, out_dir: Path | None = None, limit_s: float | None = None,
                  pose=None, workpiece_detect=None):
    """src: {camera, path, rel, start_ts, duration}. detect(frame) -> [(x1,y1,x2,y2,score)] persons.
    Returns (events, tables, meta); events: (ts, type, subject, value, clip, thumb)."""
    t0 = time.time()
    cam = normalize_camera(cam)
    cap = open_capture(src["path"], opts["hw_decode"])
    if not cap.isOpened():
        raise OSError(f"cannot open {src['path']}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if 1 <= src_fps <= 120 else 25.0
    fps_proc = float(cam.get("fps") or opts["fps"])  # a camera can ask for more frames (e.g. cycle counting)
    step = max(1, round(src_fps / fps_proc))
    dt = step / src_fps
    zones = [Zone(z["name"], z["polygon"], z.get("kind", "workstation"), z.get("hold_s", 5.0)) for z in cam.get("zones") or []]
    ctx = Ctx(src, cam, opts, out_dir)
    st = dict(frames=0, det_frames=0, processed_s=0.0)

    intervals = active_intervals(src["start_ts"], src["duration"], opts["active_hours"])
    seg = src.get("seg")  # (start_s, end_s): one chunk of a long recording, processed by its own worker
    if seg:
        intervals = [(max(a, seg[0]), min(b, seg[1])) for a, b in intervals if min(b, seg[1]) > max(a, seg[0])]
    for a, b in intervals:
        if limit_s is not None and st["processed_s"] >= limit_s:
            break
        if a > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, a * 1000.0)
        fi = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        tracker = Tracker(opts["iou_thr"], max(2, int(opts["max_lost_s"] * fps_proc)), opts["min_hits"])
        motion = Motion()
        analyzers = build_analyzers(cam, zones, pose, workpiece_detect, fps_proc)  # fresh per pass: ids and windows restart
        ctx.t_start, ctx.motion, ctx.occupied, ctx.zone_counts = a, motion, {}, {}
        seen, last_det, t_rel = set(), -1e9, a
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
            ctx.frame, ctx.t, ctx.abs_t, ctx.dt = frame, t_rel, src["start_ts"] + t_rel, dt
            ctx.h, ctx.w = frame.shape[:2]
            fm = motion.step(frame)
            detected = 0
            if fm > opts["gate_motion"] or tracker.tracks or t_rel - last_det >= opts["keepalive_s"]:
                ctx.tracks = tracker.update(detect(frame))
                last_det, detected = t_rel, 1
            else:
                ctx.tracks = []  # quiet scene and nobody tracked: skip the expensive detector
            ctx.alive = {t.id for t in tracker.tracks}
            for tr in ctx.tracks:
                if tr.id not in seen:
                    seen.add(tr.id)
                    ctx.emit("person_appeared", f"t{tr.id}")
            for az in analyzers:
                az.update(ctx)
            row = ctx.minutes.setdefault(ctx.minute(), [0.0, 0.0, 0, 0])
            row[0] += dt
            row[1] += dt * len(ctx.tracks)
            row[2] = max(row[2], len(ctx.tracks))
            row[3] += detected
            fr = ctx.fine.setdefault(ctx.fine_bin(), [0.0, 0.0, 0])
            fr[0] += dt
            fr[1] += dt * len(ctx.tracks)
            fr[2] = max(fr[2], len(ctx.tracks))
            st["frames"] += 1
            st["det_frames"] += detected
            st["processed_s"] += dt
        ctx.t, ctx.abs_t = t_rel, src["start_ts"] + t_rel
        for az in analyzers:
            az.finish(ctx)
    cap.release()
    meta = dict(camera=src["camera"], source=src["rel"] + (f"#{int(seg[0])}-{int(seg[1])}" if seg else ""), start_ts=src["start_ts"], video_s=src["duration"],
                processed_s=round(st["processed_s"], 1), frames=st["frames"], det_frames=st["det_frames"],
                elapsed_s=round(time.time() - t0, 2))
    tables = dict(minutes=ctx.minutes, zone_minutes=ctx.zone_minutes, machine_minutes=ctx.machine_minutes,
                  fine=ctx.fine, zone_fine=ctx.zone_fine, machine_fine=ctx.machine_fine)
    return ctx.events, tables, meta


_DET = None
_CACHE: dict = {}


def init_worker(model: str, conf: float):
    global _DET
    from .detector import PersonDetector
    _DET = PersonDetector(model, conf=conf, threads=1)


def _extras(cam: dict):
    """Heavy optional models are created once per worker process and only for cameras that ask for them."""
    pose = wp = None
    ina, w = cam.get("inactivity"), cam.get("workpieces")
    if ina is not None and ina.get("method", "motion") == "pose":
        from .inactivity import PoseBackend
        pose = _CACHE.setdefault(("pose", ina.get("pose_model")), PoseBackend(ina.get("pose_model")))
    if w is not None:
        from .workpieces import RFDETRDetector
        key = ("rfdetr", w.get("weights"), w.get("model", "nano"), tuple(w.get("classes") or ()), w.get("threshold", 0.5))
        if key not in _CACHE:
            _CACHE[key] = RFDETRDetector(w.get("weights"), w.get("model", "nano"), w.get("classes"), w.get("threshold", 0.5),
                                         w.get("device"), bool(w.get("trust_checkpoint", False)))
        wp = _CACHE[key].detect
    return pose, wp


def run_task(task: dict) -> dict:
    """Pool entry point. Never raises: a corrupt file must not stop a multi-day run."""
    try:
        cam = normalize_camera(task["cam"])
        pose, wp = _extras(cam)
        events, tables, meta = process_video(task["src"], cam, task["opts"], _DET.detect, Path(task["out"]),
                                             pose=pose, workpiece_detect=wp)
        write_shard(Path(task["shard"]), events, tables, {**meta, "config_hash": task.get("config_hash", "")})
        return dict(ok=True, **meta)
    except Exception as e:  # noqa: BLE001
        return dict(ok=False, source=task["src"]["rel"], error=repr(e))
