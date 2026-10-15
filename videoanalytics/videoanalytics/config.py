"""Defaults, legacy-key normalisation and strict validation of cameras.yaml (a typo must not silently cost a 6-day run)."""
from __future__ import annotations

import difflib

from .schedule import parse_hm

DEFAULTS = dict(fps=3.0, conf=0.35, min_hits=2, iou_thr=0.2, max_lost_s=2.0, gate_motion=0.0005, keepalive_s=3.0,
                clips=True, max_clips_per_file=10, clip_pad_s=5.0, active_hours=None, hw_decode=False)

ROOT = {"options", "schedule", "cameras"}
CAMERA = {"match", "fps", "zones", "lines", "machine", "machines", "gathering", "inactivity", "workpieces", "schedule"}
ZONE = {"name", "polygon", "kind", "hold_s", "schedule"}
LINE = {"name", "p1", "p2", "margin"}
MACHINE = {"name", "source", "roi", "polygon", "zone", "motion_thr", "window_s", "on_frac", "off_frac", "sat_min",
           "val_min", "min_frac", "work_colors", "hues", "cycle"}
CYCLE = {"window_s", "on_frac", "off_frac", "min_work_s", "min_idle_s"}
GATHERING = {"min_people", "dist_k", "min_duration_s", "gap_s"}
INACTIVITY = {"method", "zones", "min_duration_s", "gap_s", "sample_period_s", "motion_thr", "require_seated", "pose_model"}
WORKPIECES = {"weights", "model", "classes", "threshold", "line", "count", "min_hits", "trust_checkpoint", "device"}
SCHEDULE = {"work", "breaks", "days"}
SOURCES = {"motion", "lamp"}


class ConfigError(ValueError):
    pass


def normalize_camera(cam: dict | None) -> dict:
    """Accepts the legacy single `machine:` block (with `polygon`) and returns a `machines` list using `roi`."""
    cam = dict(cam or {})
    machines = []
    for i, m in enumerate(([cam["machine"]] if cam.get("machine") else []) + list(cam.get("machines") or [])):
        m = dict(m)
        m.setdefault("name", "machine" if i == 0 and cam.get("machine") else f"machine{i + 1}")
        m.setdefault("source", "motion")
        if "polygon" in m and "roi" not in m:
            m["roi"] = m.pop("polygon")
        if m.get("cycle") is True:
            m["cycle"] = {}
        elif m.get("cycle") in (False, None):
            m.pop("cycle", None)
        machines.append(m)
    cam["machines"] = machines
    cam.pop("machine", None)
    for key in ("gathering", "inactivity"):  # `gathering: true` / `gathering: {}` = enabled with defaults
        if cam.get(key) is True:
            cam[key] = {}
        elif key in cam and cam[key] in (False, None):
            cam.pop(key)
    return cam


def _poly_problem(p, min_pts=3):
    if not isinstance(p, (list, tuple)) or len(p) < min_pts:
        return f"needs at least {min_pts} [x, y] points"
    for pt in p:
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2 and all(isinstance(v, (int, float)) for v in pt)):
            return f"point {pt!r} is not [x, y]"
        if not all(0 <= v <= 1 for v in pt):
            return f"point {pt!r} is outside 0..1 (coordinates are normalised, not pixels)"
    return None


def check_config(cfg: dict) -> None:
    """Raises ConfigError listing every problem found."""
    errs: list[str] = []

    def unknown(where, d, allowed):
        for k in d:
            if k not in allowed:
                hint = difflib.get_close_matches(str(k), allowed, n=1)
                errs.append(f"{where}: unknown key '{k}'" + (f" (did you mean '{hint[0]}'?)" if hint else ""))

    def schedule(where, s):
        if not isinstance(s, dict):
            errs.append(f"{where}: must be a mapping")
            return
        unknown(where, s, SCHEDULE)
        for key in ("work", "breaks"):
            for pair in s.get(key) or []:
                try:
                    a, b = pair
                    parse_hm(a), parse_hm(b)
                except (ValueError, TypeError):
                    errs.append(f"{where}.{key}: {pair!r} is not a ['HH:MM', 'HH:MM'] pair")

    if not isinstance(cfg, dict):
        raise ConfigError("config must be a mapping with a 'cameras' section")
    unknown("config", cfg, ROOT)
    unknown("options", cfg.get("options") or {}, set(DEFAULTS))
    if cfg.get("schedule") is not None:
        schedule("schedule", cfg["schedule"])
    cams = cfg.get("cameras")
    if not isinstance(cams, dict) or not cams:
        errs.append("cameras: at least one camera is required")
        cams = {}
    for cname, raw in cams.items():
        w = f"cameras.{cname}"
        raw = raw or {}
        unknown(w, raw, CAMERA)
        if raw.get("schedule") is not None:
            schedule(f"{w}.schedule", raw["schedule"])
        cam = normalize_camera(raw)
        znames = []
        for i, z in enumerate(cam.get("zones") or []):
            zw = f"{w}.zones[{i}]"
            unknown(zw, z, ZONE)
            if not z.get("name"):
                errs.append(f"{zw}: 'name' is required")
            if z.get("name") in znames:
                errs.append(f"{zw}: duplicate zone name '{z.get('name')}'")
            znames.append(z.get("name"))
            if (p := _poly_problem(z.get("polygon"))):
                errs.append(f"{zw}.polygon: {p}")
        for i, ln in enumerate(cam.get("lines") or []):
            lw = f"{w}.lines[{i}]"
            unknown(lw, ln, LINE)
            for key in ("name", "p1", "p2"):
                if key not in ln:
                    errs.append(f"{lw}: '{key}' is required")
            for key in ("p1", "p2"):
                if key in ln and (p := _poly_problem([ln[key]], 1)):
                    errs.append(f"{lw}.{key}: {p}")
        mnames = []
        for i, m in enumerate(cam["machines"]):
            mw = f"{w}.machines[{i}]"
            unknown(mw, m, MACHINE)
            if m["name"] in mnames:
                errs.append(f"{mw}: duplicate machine name '{m['name']}'")
            mnames.append(m["name"])
            if m["source"] not in SOURCES:
                errs.append(f"{mw}.source: '{m['source']}' must be one of {sorted(SOURCES)}")
            if (p := _poly_problem(m.get("roi"))):
                errs.append(f"{mw}.roi: {p}")
            if m.get("zone") and m["zone"] not in znames:
                errs.append(f"{mw}.zone: '{m['zone']}' is not a zone of this camera ({znames})")
            if isinstance(m.get("cycle"), dict):
                unknown(f"{mw}.cycle", m["cycle"], CYCLE)
        if cam.get("gathering") is not None:
            unknown(f"{w}.gathering", cam["gathering"] or {}, GATHERING)
        ina = cam.get("inactivity")
        if ina is not None:
            unknown(f"{w}.inactivity", ina, INACTIVITY)
            if ina.get("method", "motion") not in ("pose", "motion"):
                errs.append(f"{w}.inactivity.method: must be 'pose' or 'motion'")
            for zn in ina.get("zones") or []:
                if zn not in znames:
                    errs.append(f"{w}.inactivity.zones: '{zn}' is not a zone of this camera ({znames})")
        wp = cam.get("workpieces")
        if wp is not None:
            unknown(f"{w}.workpieces", wp, WORKPIECES)
            if "line" not in wp:
                errs.append(f"{w}.workpieces.line: required (p1/p2 of the counting line)")
            else:
                unknown(f"{w}.workpieces.line", wp["line"], {"name", "p1", "p2"})
            if wp.get("count", "out") not in ("in", "out", "both"):
                errs.append(f"{w}.workpieces.count: must be 'in', 'out' or 'both'")
            if wp.get("model", "nano") not in ("nano", "small", "medium", "large"):
                errs.append(f"{w}.workpieces.model: must be nano, small, medium or large (XL/2XL use a non-commercial-friendly licence)")
    if errs:
        raise ConfigError("invalid config:\n  " + "\n  ".join(errs))
