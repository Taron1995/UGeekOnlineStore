"""The setup of a results folder: cameras.auto.yaml (what the analysis uses - stations, machines and tracking areas per camera), the picture
of every camera that shows it (setup/<camera>.jpg) and what the scene learning found (setup/setup.json).
Written by the scene learning and by the zone editor, read by analyze."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from .config import check_config
from .tools import snapshot

SETUP_FILE = "cameras.auto.yaml"
HEADER = """# Your setup: stations (where people work), machines (a processing section where people work, a moving part, or a signal lamp) and tracking areas, per camera.
# It was learned from the footage and / or drawn by hand (python -m videoanalytics zones ...). It is a starting point, not the truth:
# open the pictures in the 'setup' folder, correct what is wrong (by hand here, or in the zone editor), save, and run the same command
# again - this file is used as it is, and only the cameras whose setup changed are computed again.
# Coordinates are 0..1 (x left -> right, y top -> bottom). Add a work schedule to get absences, e.g.:
#   schedule: {work: [["08:00", "12:00"], ["13:00", "17:00"]], breaks: [["10:00", "10:15"]], days: [1, 2, 3, 4, 5]}
# Tracking areas (follow people only inside / never inside), per camera:
#   areas: [{name: floor, mode: watch, polygon: [[0.1, 0.2], [0.9, 0.2], [0.9, 0.95], [0.1, 0.95]]}]
"""


class _FlowLists(yaml.SafeDumper):
    """Polygons and points on one line ([[0.52, 0.25], ...]) so the file can be edited by hand."""


def _list(dumper, data):
    flat = all(isinstance(x, (int, float)) for x in data) or all(isinstance(x, list) and all(isinstance(y, (int, float)) for y in x) for x in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=bool(data) and flat)


_FlowLists.add_representer(list, _list)


def setup_file(out_dir) -> Path:
    return Path(out_dir) / SETUP_FILE


def write_setup(path, cfg_all: dict) -> Path:
    """Checks the whole config, then writes it (raises ConfigError listing every problem; nothing is written then). A file that was there is kept
    as <name>.bak: learning again or saving from the zone editor must not cost work that was drawn by hand."""
    check_config(cfg_all)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + ".bak"))
        print(f"(the previous setup is kept as {path.name}.bak)", flush=True)
    path.write_text(HEADER + yaml.dump(cfg_all, Dumper=_FlowLists, sort_keys=False, allow_unicode=True, width=150), encoding="utf-8")
    return path


def read_info(out_dir) -> dict:
    """What the scene learning found, per camera ({} when nothing was learned)."""
    path = Path(out_dir) / "setup" / "setup.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def write_info(out_dir, info: dict) -> None:
    path = Path(out_dir) / "setup" / "setup.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")


def camera_picture(out_dir, cam: str, video: str, at_s: float, cam_cfg: dict) -> str | None:
    """setup/<camera>.jpg: a frame with the camera's setup drawn on it. Returns the path relative to the results folder, or None."""
    rel = f"setup/{cam}.jpg"
    try:
        snapshot(video, Path(out_dir) / rel, cam_cfg, at_s)
    except OSError:
        return None
    return rel
