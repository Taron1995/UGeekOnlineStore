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
DATEISH_RE = re.compile(r"^\d{4}[-_.]?\d{2}[-_.]?\d{2}$")
_BARE = re.compile(r"(?<!\d)([01]\d|2[0-3])[-_.:]?([0-5]\d)[-_.:]?([0-5]\d)(?!\d)")  # a bare time such as 08-00-00


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


RESULT_MARKS = ("analytics.db", "cameras.auto.yaml")  # a folder holding one of these is a results folder of an earlier run


def list_videos(root: Path, only=None, exclude=()):
    """Video files below root (recursive), optionally only the given files. The evidence clips inside a results folder (this
    run's `exclude`, or one left by an earlier run) are never inputs."""
    keep = {Path(o).resolve() for o in only} if only else None
    skip = [Path(e).resolve() for e in exclude]
    results: dict[Path, bool] = {}

    def is_results(d: Path) -> bool:
        if d not in results:
            results[d] = any((d / m).exists() for m in RESULT_MARKS)
        return results[d]

    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXT or not p.is_file():
            continue
        r = p.resolve()
        inside = [root.joinpath(*p.relative_to(root).parts[:k]) for k in range(1, len(p.relative_to(root).parts))]  # folders between root and the file
        if (keep is None or r in keep) and not any(e == r or e in r.parents for e in skip) and not any(is_results(d) for d in inside):
            out.append(p)
    return out


def _source(p: Path, rel: str, cam: str, time_regex, issues: dict):
    info = probe(str(p))
    if info is None:
        issues["unreadable"].append(rel)
        return None
    fps, dur = info
    start = parse_start(rel, time_regex)
    if start is None:  # last resort: assume mtime is the end of the recording
        start = naive_epoch(datetime.fromtimestamp(os.path.getmtime(p))) - dur
        issues["no_timestamp"].append(rel)
    return dict(camera=cam, path=str(p), rel=rel, start_ts=start, duration=dur, fps=fps)


def discover(root: str, cameras: dict, time_regex: str | None = None, only=None, exclude=()):
    """Returns (sources sorted by time, issues dict). Each source is a plain dict (picklable)."""
    root_p = Path(root)
    srcs, issues = [], dict(unmatched=[], no_timestamp=[], unreadable=[])
    for p in list_videos(root_p, only, exclude):
        rel = str(p.relative_to(root_p))
        cam = camera_for(rel, cameras)
        if cam is None:
            issues["unmatched"].append(rel)
            continue
        s = _source(p, rel, cam, time_regex, issues)
        if s:
            srcs.append(s)
    srcs.sort(key=lambda s: (s["start_ts"], s["camera"]))
    return srcs, issues


# ---- cameras from file names (no config) ------------------------------------------------------------------------------
def _clean(s: str) -> str:
    return re.sub(r"[\W_]+", "_", s, flags=re.UNICODE).strip("_")  # separators (also '_') collapse into one underscore


def auto_camera_name(rel: str):
    """cam1_20261001_080000.mp4 -> cam1;  2026-10-01/cam2/08-00-00.mp4 -> cam2;  None when the path says nothing."""
    p = Path(rel.replace("\\", "/"))
    had_ts = bool(TS_RE.search(p.stem))
    stem = _BARE.sub(" ", DATE_RE.sub(" ", TS_RE.sub(" ", p.stem)))
    name = _clean(stem)
    if had_ts:
        name = re.sub(r"_\d{3,4}$", "", name)  # chunk counters: cam1_20261001_080000_001.mp4
    if name and not name.isdigit():
        return name
    for parent in reversed(p.parent.parts):
        n = _clean(parent)
        if n and not DATEISH_RE.match(parent) and not n.isdigit():
            return n
    return None



def discover_auto(root: str, only=None, time_regex: str | None = None, exclude=()):
    """Cameras are taken from the file names. Returns (sources, issues, cameras) where cameras maps each name to {'match': regex}
    that selects exactly its files again (stored in cameras.auto.yaml so an edited setup keeps working)."""
    root_p = Path(root)
    srcs, issues = [], dict(unmatched=[], no_timestamp=[], unreadable=[])
    fallback = _clean(root_p.name) or "camera"
    named, nameless = {}, []
    for p in list_videos(root_p, only, exclude):
        rel = str(p.relative_to(root_p))
        n = auto_camera_name(rel)
        (nameless if n is None else named.setdefault(n, [])).append((p, rel))
    groups = dict(named)
    if nameless:
        groups.setdefault(fallback, []).extend(nameless)
    for cam, files in groups.items():
        for p, rel in files:
            s = _source(p, rel, cam, time_regex, issues)
            if s:
                srcs.append(s)
    srcs.sort(key=lambda s: (s["start_ts"], s["camera"]))
    others = [re.escape(n) for n in named]
    cameras = {}
    for cam in groups:
        if cam in named:
            cameras[cam] = {"match": rf"(?<![A-Za-z0-9]){re.escape(cam)}(?![A-Za-z0-9])"}
        else:  # the files that carried no name: everything the named cameras do not claim
            excl = rf"^(?!.*(?<![A-Za-z0-9])(?:{'|'.join(others)})(?![A-Za-z0-9])).*" if others else ".*"
            cameras[cam] = {"match": excl}
    return srcs, issues, cameras
