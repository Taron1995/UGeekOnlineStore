"""Small video helpers shared by the pipeline and the analyzers."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import cv2


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


def _ascii_path(path: str) -> str:
    """Windows: some OpenCV builds cannot open a name with non-ASCII letters (Armenian, Russian, ...); the short 8.3 form of the same
    path is plain ASCII. Anywhere else, or if there is no short form, the path is returned unchanged."""
    if os.name != "nt" or path.isascii():
        return path
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        n = ctypes.windll.kernel32.GetShortPathNameW(os.path.abspath(path), buf, 1024)
        return buf.value if 0 < n < 1024 else path
    except Exception:  # noqa: BLE001
        return path


def open_capture(path: str, hw: bool = False):
    if hw:  # D3D11VA/VAAPI if the OpenCV build has it; silently falls back to software
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        if cap.isOpened():
            return cap
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():  # only now (and only on Windows, for non-ASCII names) is a second attempt made
        alt = _ascii_path(path)
        if alt != path:
            cap = cv2.VideoCapture(alt)
    return cap


def save_clip(path: str, t_rel: float, out_path: Path, pad: float, hw: bool = False, max_w: int = 960):
    """The clip around an event. Pictures wider than max_w are scaled down: at 1080p / 4K a clip of the original size is tens of MB and takes
    seconds to encode, and 39 jobs of 10 clips each fill a disk."""
    cap = open_capture(path, hw)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 120 else 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (w, h) if w <= max_w else (max_w // 2 * 2, int(h * max_w / w) // 2 * 2)  # even numbers: some encoders refuse odd sides
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_rel - pad) * 1000)
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(int(2 * pad * fps)):
        ok, f = cap.read()
        if not ok:
            break
        out.write(f if (w, h) == size else cv2.resize(f, size, interpolation=cv2.INTER_AREA))
    out.release()
    cap.release()
