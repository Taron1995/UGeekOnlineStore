"""What the program sees in your recordings, as one text: the pre-flight check (`doctor`) and the file written next to the results
(`diagnostics.txt`). It holds relative file names, numbers and versions - no pictures - and is meant to be sent along when something looks
wrong, so that the problem can be found without access to your computer."""
from __future__ import annotations

import os
import platform
import re
import sqlite3
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .sources import VIDEO_EXT
from .video import active_intervals, fmt_ts, open_capture

RECORDER_EXT = {".dav", ".h264", ".264", ".h265", ".265", ".asf", ".bin", ".rec"}
SAME_SCENE, DIFFERENT_SCENE = 0.85, 0.30  # correlation of 64x36 grey thumbnails: above = looks like the same view, below = hardly related
SLOW_DECODE_FPS = 15.0


def _dur(seconds: float) -> str:
    s = float(seconds)
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h"


def fourcc_text(v: float) -> str:
    n = int(v)
    chars = [chr((n >> (8 * i)) & 0xFF) for i in range(4)]
    return "".join(chars) if all(32 <= ord(c) < 127 for c in chars) else (f"0x{n:08x}" if n else "?")


# ---- one recording ---------------------------------------------------------------------------------------------------
def _picture(frame) -> dict:
    """brightness 0-255, and the share of clearly coloured pixels in percent (an infrared / night picture has almost none)."""
    hsv = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    return dict(brightness=float(hsv[..., 2].mean()), coloured=100.0 * float(((hsv[..., 1] > 50) & (hsv[..., 2] > 50)).mean()))


def _thumb(frame) -> np.ndarray:
    g = cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g -= g.mean()
    n = float(np.linalg.norm(g))
    return g / n if n > 1e-6 else g


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of two thumbnails: about 1 for the same view, near 0 for unrelated pictures."""
    return float((a * b).sum())


def inspect_file(path: str, frames: int = 20) -> dict:
    """What OpenCV (with its bundled FFmpeg) makes of one recording: format, size, how fast it decodes and seeks, how the picture looks."""
    info: dict = dict(path=path, opened=False)
    cap = open_capture(path)
    try:
        if not cap.isOpened():
            info["error"] = "cannot be opened"
            return info
        info.update(opened=True, backend=cap.getBackendName() if hasattr(cap, "getBackendName") else "?",
                    codec=fourcc_text(cap.get(cv2.CAP_PROP_FOURCC)), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps=float(cap.get(cv2.CAP_PROP_FPS)),
                    frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        first, n, t0 = None, 0, time.perf_counter()
        for _ in range(frames):
            ok, f = cap.read()
            if not ok:
                break
            first = f if first is None else first
            n += 1
        dt = time.perf_counter() - t0
        info.update(decoded=n, decode_fps=n / dt if n and dt > 0 else 0.0)
        if first is None:
            info["error"] = "opens, but no frame can be decoded (unsupported codec or a damaged file)"
            return info
        shown = first
        if info["frame_count"] > 60:  # the scene learning and the chunks seek into the file
            t0 = time.perf_counter()
            cap.set(cv2.CAP_PROP_POS_FRAMES, info["frame_count"] // 2)
            ok, mid = cap.read()
            info.update(seek_s=time.perf_counter() - t0, seek_ok=bool(ok))
            shown = mid if ok else first
        info.update(_picture(shown), thumb=_thumb(shown))
        return info
    finally:
        cap.release()


def pick_samples(srcs: list[dict], per_camera: int = 3, max_cameras: int = 12) -> list[dict]:
    """First / middle / last recording of every camera (the middle one only, beyond `max_cameras` cameras)."""
    by: dict = defaultdict(list)
    for s in sorted(srcs, key=lambda s: s["start_ts"]):
        by[s["camera"]].append(s)
    out = []
    for i, (cam, files) in enumerate(sorted(by.items())):
        picks = [files[len(files) // 2]] if i >= max_cameras or per_camera == 1 else [files[0], files[len(files) // 2], files[-1]]
        out += [s for k, s in enumerate(picks) if s not in picks[:k]]
    return out


def inspect_samples(srcs: list[dict], per_camera: int = 3) -> list[dict]:
    return [{**inspect_file(s["path"]), "camera": s["camera"], "rel": s["rel"], "duration": s["duration"]} for s in pick_samples(srcs, per_camera)]


# ---- the computer ------------------------------------------------------------------------------------------------------
def _commit() -> str:
    try:
        flags = 0x08000000 if os.name == "nt" else 0  # no console window flashing up from a window-less program
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5, creationflags=flags)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def _video_backends() -> str:
    lines = cv2.getBuildInformation().splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("Video I/O"))
    except StopIteration:
        return "unknown"
    found = []
    for ln in lines[start + 1:]:
        if not ln.strip():
            break
        key, _, val = ln.strip().partition(":")
        if val.strip().startswith("YES"):
            found.append(f"{key} {val.strip()}")
    return ", ".join(found) or "none"


def environment_lines() -> list[str]:
    try:
        import onnxruntime as ort
        ort_v = ort.__version__
    except Exception:  # noqa: BLE001
        ort_v = "missing"
    try:
        from importlib.metadata import version
        ver = version("videoanalytics")
    except Exception:  # noqa: BLE001
        ver = "?"
    commit = _commit()
    return [f"videoanalytics {ver}" + (f" ({commit})" if commit else ""),
            f"Python {platform.python_version()} on {platform.platform()}, {os.cpu_count()} CPU core(s)",
            f"numpy {np.__version__}, OpenCV {cv2.__version__}, onnxruntime {ort_v}",
            f"video backends of OpenCV: {_video_backends()}"]


# ---- the recordings ----------------------------------------------------------------------------------------------------
def other_files(paths) -> Counter:
    """File types in the given folders that are not videos (a recorder format the program does not know shows up here)."""
    seen: Counter = Counter()
    for raw in paths:
        p = Path(str(raw).strip().strip('"')).expanduser()
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() not in VIDEO_EXT:
                    seen[f.suffix.lower() or "(no extension)"] += 1
    return seen


def _coverage(files: list[dict]) -> tuple[int, float, float, int]:
    """(gaps over 5 s, total gap seconds, largest gap seconds, overlaps over 5 s) between consecutive recordings of one camera."""
    gaps, total, largest, overlaps = 0, 0.0, 0.0, 0
    for a, b in zip(files, files[1:]):
        g = b["start_ts"] - (a["start_ts"] + a["duration"])
        if g > 5:
            gaps, total, largest = gaps + 1, total + g, max(largest, g)
        elif g < -5:
            overlaps += 1
    return gaps, total, largest, overlaps


def camera_lines(srcs: list[dict], hours=None, limit: int = 12) -> list[str]:
    by: dict = defaultdict(list)
    for s in sorted(srcs, key=lambda s: s["start_ts"]):
        by[s["camera"]].append(s)
    lines = []
    for cam, files in sorted(by.items())[:limit]:
        video = sum(b - a for s in files for a, b in active_intervals(s["start_ts"], s["duration"], hours))
        durs = sorted(s["duration"] for s in files)
        gaps, gap_total, gap_max, overlaps = _coverage(files)
        extra = f"; {gaps} gap(s) over 5 s (together {_dur(gap_total)}, largest {_dur(gap_max)})" if gaps else "; no gaps"
        extra += f"; {overlaps} overlap(s)" if overlaps else ""
        lines.append(f"  {cam}: {len(files)} file(s), {_dur(video)} of video, {fmt_ts(files[0]['start_ts'], '%Y-%m-%d %H:%M')} -> "
                     f"{fmt_ts(files[-1]['start_ts'] + files[-1]['duration'], '%Y-%m-%d %H:%M')}; "
                     + (f"each {_dur(durs[0])}" if durs[0] == durs[-1] or _dur(durs[0]) == _dur(durs[-1])
                        else f"{_dur(durs[0])} - {_dur(durs[-1])} each (median {_dur(durs[len(durs) // 2])})") + extra)
    if len(by) > limit:
        lines.append(f"  ... and {len(by) - limit} more camera(s)")
    return lines


def sample_line(s: dict) -> str:
    if not s.get("opened") or s.get("error"):
        return f"  {s['camera']:12} {s['rel']}: {s.get('error', 'cannot be opened')}"
    seek = "" if "seek_s" not in s else (f", seek {s['seek_s']:.1f} s" if s["seek_ok"] else ", SEEK FAILS")
    return (f"  {s['camera']:12} {s['rel']}: {s['codec']} {s['width']}x{s['height']} at {s['fps']:.1f} fps, {_dur(s['duration'])}, "
            f"decodes {s['decode_fps']:.0f} fps{seek}, brightness {s['brightness']:.0f}/255, coloured pixels {s['coloured']:.1f}%")


def issue_hints(issues: dict, files_seen: Counter | None = None) -> list[str]:
    """Plain-language explanations for the problems found while reading the recordings."""
    notes = []
    bad = issues.get("unreadable", [])
    if bad:
        exts = Counter(Path(r).suffix.lower() for r in bad)
        note = (f"{len(bad)} file(s) cannot be opened or decoded and are skipped, e.g. {bad[:3]}.")
        if any(e in RECORDER_EXT for e in exts):
            note += (" Recorder formats such as .dav / .h264 usually have to be converted first: ffmpeg -i input.dav -c copy output.mp4"
                     " (or export MP4 from the recorder's own software).")
        if os.name == "nt" and any(not r.isascii() for r in bad):
            note += " The names contain non-ASCII letters: copy the videos to a folder with an ASCII-only path, e.g. C:\\videos."
        notes.append(note + " Corrupt or still-being-written files fail the same way.")
    if issues.get("no_timestamp"):
        notes.append(f"{len(issues['no_timestamp'])} file(s) have no date/time in their name: the file modification time is used, which for "
                     f"downloaded files is when you saved them, not when they were recorded. Tell the program the real start with "
                     f"--start \"2026-10-01 08:00\" (files without a time then follow each other), or use --time-regex for an unusual name pattern.")
    if issues.get("manual_time"):
        notes.append(f"{len(issues['manual_time'])} file(s) have no date/time in their name: the start you gave (--start) is used and these files "
                     f"are taken to follow each other in name order.")
    if issues.get("unmatched"):
        notes.append(f"{len(issues['unmatched'])} file(s) match no camera of the setup, e.g. {issues['unmatched'][:3]}.")
    if files_seen:
        top = ", ".join(f"{e} ({n})" for e, n in sorted(files_seen.items(), key=lambda kv: (-kv[1], kv[0]))[:5])
        unknown = sorted(e for e in files_seen if e in RECORDER_EXT)
        if unknown:
            notes.append(f"Other files in the folder: {top}. Recorder formats ({', '.join(unknown)}) are not video files the program can read directly.")
    return notes


def name_hints(srcs: list[dict], samples: list[dict], issues: dict | None = None) -> list[str]:
    notes = []
    cams = sorted({s["camera"] for s in srcs})
    no_time = bool((issues or {}).get("no_timestamp") or (issues or {}).get("manual_time"))  # names without a time are the unreliable ones
    if no_time and len({re.sub(r"\d+", "#", c) for c in cams}) < len(cams):  # cam1, cam2, ...: the same name with another number
        notes.append(f"The camera names differ only by a number ({', '.join(cams[:4])}{', ...' if len(cams) > 4 else ''}). That is right if these are "
                     f"different cameras. If they are parts of ONE camera, run with --one-camera.")
    by_cam: dict = defaultdict(list)
    for s in samples:
        if s.get("thumb") is not None:
            by_cam[s["camera"]].append(s)
    reps = {cam: v[len(v) // 2] for cam, v in by_cam.items()}
    names = sorted(reps)
    group = {n: n for n in names}  # cameras that show the same view, joined into groups

    def find(n):
        while group[n] != n:
            n = group[n]
        return n

    link_sim: dict = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sim = similarity(reps[a]["thumb"], reps[b]["thumb"])
            if sim >= SAME_SCENE:
                group[find(a)] = find(b)
                link_sim[(a, b)] = sim
    members: dict = defaultdict(list)
    for n in names:
        members[find(n)].append(n)
    for g in members.values():
        if len(g) > 1:
            low = min(sim for (a, b), sim in link_sim.items() if a in g and b in g)
            notes.append(f"Cameras {', '.join(g)} show almost the same picture (similarity {low:.2f} or more). If they are parts of one camera, "
                         f"run with --one-camera; if they really are different cameras with the same view, ignore this.")
    for cam, v in sorted(by_cam.items()):
        if len(v) >= 2:
            sim = similarity(v[0]["thumb"], v[-1]["thumb"])
            if sim < DIFFERENT_SCENE:
                notes.append(f"The first and the last file of camera {cam} show very different pictures (similarity {sim:.2f}): day and night can "
                             f"do that, but if several cameras are mixed under this name, split them into folders or use --config.")
    return notes


def sample_hints(samples: list[dict]) -> list[str]:
    notes = []
    for s in samples:
        if not s.get("opened") or s.get("error"):
            continue
        where = f"{s['camera']} ({s['rel']})"
        if s["frame_count"] <= 0:
            raw = Path(s["rel"]).suffix.lower() in (".h264", ".264", ".h265", ".265")  # a bare stream has no timing at all
            notes.append(f"{where}: the file does not say how long it is; it is read from its start to its end. "
                         + ("A bare stream also has no frame rate (25 is assumed): re-pack it with the real one, e.g. ffmpeg -r 10 -i in.h264 -c copy out.mp4"
                            if raw else "Re-packing it gives it a length: ffmpeg -i in -c copy out.mp4"))
        if not 1 <= s["fps"] <= 120:
            notes.append(f"{where}: frame rate {s['fps']:.1f} is not usable, 25 fps is assumed.")
        if s.get("seek_s") is not None and not s["seek_ok"]:
            notes.append(f"{where}: jumping into the middle of the file fails, so scene learning and long-file chunks cannot work on it. Re-pack it with ffmpeg.")
        if s["decode_fps"] and s["decode_fps"] < SLOW_DECODE_FPS:
            notes.append(f"{where}: decoding runs at only {s['decode_fps']:.0f} frames per second ({s['codec']}, {s['width']}x{s['height']}); "
                         f"expect a long run. --hw-decode may help; so may --hours.")
        if s["coloured"] < 0.2:
            notes.append(f"{where}: the picture has almost no colour (night / infrared mode?). People are still found, signal-lamp colours cannot be read.")
        if s["width"] < 480:
            notes.append(f"{where}: small picture ({s['width']}x{s['height']}); people far from the camera may be missed.")
    return list(dict.fromkeys(notes))


def notes(srcs: list[dict], issues: dict, samples: list[dict], paths=()) -> list[str]:
    """Everything worth a look, in plain language (the console prints these before the long part; the text lists them at its end)."""
    return issue_hints(issues, other_files(paths) if paths else None) + name_hints(srcs, samples, issues) + sample_hints(samples)


# ---- the whole text -----------------------------------------------------------------------------------------------------
def doctor_text(srcs: list[dict], issues: dict, *, paths=(), samples: list[dict] | None = None, hours=None, cost: float | None = None,
                workers: int | None = None) -> str:
    """The text of the pre-flight check. `samples` = inspect_samples() (made here when not given); `cost` = CPU-seconds per video-second from a
    benchmark (None: not measured)."""
    samples = inspect_samples(srcs) if samples is None else samples
    total_video = sum(b - a for s in srcs for a, b in active_intervals(s["start_ts"], s["duration"], hours))
    size = sum(os.path.getsize(s["path"]) for s in srcs if os.path.exists(s["path"]))
    exts = Counter(Path(s["rel"]).suffix.lower() for s in srcs)
    longest = max((len(s["path"]) for s in srcs), default=0)
    out = [f"VIDEO ANALYTICS - DIAGNOSTICS  ({datetime.now():%Y-%m-%d %H:%M:%S})",
           "Send this text if something looks wrong. It holds relative file names, numbers and versions, no pictures.", "",
           "== Computer", *environment_lines(), "",
           "== Recordings",
           f"{len(srcs)} readable video file(s) ({', '.join(f'{e} {n}' for e, n in exts.most_common())}), {size / 1e9:.1f} GB, {_dur(total_video)} of video"
           + (f"; {len(issues.get('unreadable', []))} unreadable" if issues.get("unreadable") else ""),
           f"longest path {longest} characters" + (" (Windows stops at 260)" if longest > 240 else ""),
           "Cameras, as the file names say:", *camera_lines(srcs, hours)]
    shown = [s["rel"] for s in sorted(srcs, key=lambda s: s["rel"])]
    out += ["Examples of the paths (relative):", *[f"  {r}" for r in dict.fromkeys(shown[:2] + shown[len(shown) // 2:len(shown) // 2 + 1] + shown[-1:])]]
    odd = set(issues.get("no_timestamp", [])) | set(issues.get("manual_time", []))
    out += [f"Time of the recordings: from the file name for {sum(1 for r in shown if r not in odd)} file(s)"
            + (f", from the file modification time for {len(issues['no_timestamp'])}" if issues.get("no_timestamp") else "")
            + (f", from --start for {len(issues['manual_time'])}" if issues.get("manual_time") else ""), "",
            "== Samples (first / middle / last recording of each camera)", *[sample_line(s) for s in samples], ""]
    if cost is not None:
        w = workers or os.cpu_count() or 1
        out += ["== Speed on this computer",
                f"{cost:.2f} CPU-seconds per second of video (decoder + person detector on every sampled frame, one thread)",
                f"Upper-bound estimate for {_dur(total_video)} of video on {w} worker(s): about {_dur(total_video * cost / w)} "
                "(the motion gate usually makes it faster)", ""]
    out += ["== What to look at", *([f"- {n}" for n in notes(srcs, issues, samples, paths)] or ["- Nothing unusual found."])]
    return "\n".join(out) + "\n"


def result_text(out_dir: Path) -> str:
    """After the analysis: what the learned setup contains and what came out (counts only)."""
    lines = ["", "== Result"]
    used = out_dir / "cameras.auto.yaml"
    if used.exists():  # what the analysis really used: learned, drawn by hand, or both
        try:
            import yaml
            for cam, c in sorted((yaml.safe_load(used.read_text(encoding="utf-8")) or {}).get("cameras", {}).items()):
                zones, machines = (c or {}).get("zones") or [], (c or {}).get("machines") or []
                lines.append(f"  {cam}: setup used: {sum(1 for z in zones if z.get('kind', 'workstation') == 'workstation')} station(s), "
                             f"{sum(1 for z in zones if z.get('kind', 'workstation') != 'workstation')} other zone(s), "
                             f"{sum(1 for m in machines if m.get('source', 'motion') == 'motion')} moving machine zone(s), "
                             f"{sum(1 for m in machines if m.get('source') == 'lamp')} signal lamp(s), {len((c or {}).get('areas') or [])} tracking area(s)")
        except Exception:  # noqa: BLE001 - a hand-edited file may be broken; the analysis already refused it if so
            lines.append("  setup used: (cameras.auto.yaml could not be read)")
    setup = out_dir / "setup" / "setup.json"
    if setup.exists():
        import json
        for cam, info in sorted(json.loads(setup.read_text(encoding="utf-8")).items()):
            lines.append(f"  {cam}: learned {len(info.get('spots', []))} station(s), {len(info.get('motion', []))} moving machine zone(s), "
                         f"{len(info.get('lamps', []))} signal lamp(s) from {info.get('sampled_s', 0)} s of footage "
                         f"({info.get('person_tracks', 0)} person track(s) seen)")
    db = out_dir / "analytics.db"
    if db.exists():
        con = sqlite3.connect(db)
        try:
            for cam, hrs, avg, peak in con.execute("SELECT camera, SUM(covered_s) / 3600.0, SUM(people_s) / NULLIF(SUM(covered_s), 0), MAX(people_max) "
                                                   "FROM minutes GROUP BY camera ORDER BY camera"):
                lines.append(f"  {cam}: analysed {_dur(hrs * 3600)}, {avg or 0:.2f} people in view on average, peak {peak}")
            done, video, cpu = con.execute("SELECT COUNT(*), COALESCE(SUM(processed_s), 0), COALESCE(SUM(elapsed_s), 0) FROM files").fetchone()
            if video > 0:
                lines.append(f"  speed of this run: {cpu / video:.2f} CPU-seconds per second of video ({done} job(s), {_dur(video)} of video in {_dur(cpu)} of processing)")
            events = con.execute("SELECT type, COUNT(*) FROM events GROUP BY type ORDER BY 2 DESC").fetchall()
            lines.append("  events: " + (", ".join(f"{t} {n}" for t, n in events) or "none"))
            for cam, mach, work, idle in con.execute("SELECT camera, machine, SUM(work_s), SUM(idle_s) FROM machine_minutes GROUP BY camera, machine"):
                lines.append(f"  machine {cam}/{mach}: working {100 * work / max(work + idle, 1):.0f}% of {_dur(work + idle)}")
            for cam, zone, occ, cov in con.execute("SELECT camera, zone, SUM(occupied_s), SUM(covered_s) FROM zone_minutes GROUP BY camera, zone"):
                lines.append(f"  station {cam}/{zone}: occupied {100 * occ / max(cov, 1):.0f}% of {_dur(cov)}")
        finally:
            con.close()
    errors = out_dir / "errors.log"
    if errors.exists() and errors.stat().st_size:
        lines += ["  errors:", *[f"    {ln}" for ln in errors.read_text(encoding="utf-8", errors="replace").splitlines()[:10]]]
    return "\n".join(lines) + "\n"

