"""Find recordings, map them to cameras and recover absolute start times (works for multi-day NVR exports)."""
from __future__ import annotations

import calendar
import os
import re
from datetime import datetime
from pathlib import Path

import cv2

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".ts", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"}
# 20261001_080000 | 2026-10-01_08-00-00 | 2026-10-01/08.00.00 (path separators become '_')
TS_RE = re.compile(r"(?<!\d)(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[ T_\-.]*(\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})(?!\d)")
DATE_RE = re.compile(r"(?<!\d)(20\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])(?!\d)")
TIME_RE = re.compile(r"(?<!\d)([01]\d|2[0-3])[-_.:]?([0-5]\d)[-_.:]?([0-5]\d)(?!\d)")


def naive_epoch(dt: datetime) -> float:
    """Wall-clock time of the recording as seconds (timezone/DST ignored on purpose)."""
    return float(calendar.timegm(dt.timetuple()))


def parse_start(rel: str, custom: str | None = None):
    path = rel.replace("\\", "/")
    flat = path.replace("/", "_")
    for m in (re.compile(custom) if custom else TS_RE).finditer(flat):
        try:
            return naive_epoch(datetime(*map(int, m.groups()[:6])))
        except ValueError:
            continue
    if custom:
        return None
    # layout 2026-10-01/cam1/08-00-00.mp4: date in a folder, time in the file name
    folders, _, name = path.rpartition("/")
    d, t = DATE_RE.search(folders), TIME_RE.search(name.rsplit(".", 1)[0])
    if d and t:
        return naive_epoch(datetime(*map(int, d.groups()), *map(int, t.groups())))
    return None


def camera_for(rel: str, cameras: dict):
    best = None
    for name, cfg in cameras.items():
        pat = (cfg or {}).get("match") or rf"(?<![A-Za-z0-9]){re.escape(name)}(?![0-9])"
        if re.search(pat, rel, re.I) and (best is None or len(name) > len(best)):
            best = name
    return best


def probe(path: str):
    """(fps, duration_s), or None if the file cannot be opened/decoded (corrupt, partial, unsupported codec)."""
    cap = cv2.VideoCapture(path)
    ok = cap.isOpened() and cap.grab()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if not ok:
        return None
    fps = fps if 1 <= fps <= 120 else 25.0
    return fps, (frames / fps if frames and frames > 0 else 3600.0)


def discover(root: str, cameras: dict, time_regex: str | None = None):
    """Returns (sources sorted by time, issues dict). Each source is a plain dict (picklable)."""
    root_p = Path(root)
    srcs, unmatched, no_ts, unreadable = [], [], [], []
    for p in sorted(root_p.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXT or not p.is_file():
            continue
        rel = str(p.relative_to(root_p))
        cam = camera_for(rel, cameras)
        if cam is None:
            unmatched.append(rel)
            continue
        info = probe(str(p))
        if info is None:
            unreadable.append(rel)
            continue
        fps, dur = info
        start = parse_start(rel, time_regex)
        if start is None:  # last resort: assume mtime is the end of the recording
            start = naive_epoch(datetime.fromtimestamp(os.path.getmtime(p))) - dur
            no_ts.append(rel)
        srcs.append(dict(camera=cam, path=str(p), rel=rel, start_ts=start, duration=dur, fps=fps))
    srcs.sort(key=lambda s: (s["start_ts"], s["camera"]))
    return srcs, dict(unmatched=unmatched, no_timestamp=no_ts, unreadable=unreadable)
