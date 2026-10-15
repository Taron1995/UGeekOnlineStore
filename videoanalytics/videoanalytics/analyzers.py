"""Per-camera analyzers. process_video() builds a fresh set for every recording pass and feeds each processed frame
through them: zones (time at the machine), counting lines, machines (lamp colour or motion, cycles) and gatherings."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .lamp import LampReader
from .video import fmt_ts, save_clip
from .zones import Line, MachineState, Motion, Zone


@dataclass
class Ctx:
    """Everything the analyzers share for one recording: current frame, tracks and the result accumulators."""
    src: dict
    cam: dict
    opts: dict
    out_dir: Path | None
    events: list = field(default_factory=list)  # (ts, type, subject, value, clip, thumb)
    minutes: dict = field(default_factory=dict)  # minute -> [covered_s, people_s, people_max, det_frames]
    zone_minutes: dict = field(default_factory=dict)  # (zone, minute) -> [covered_s, occupied_s, people_s]
    machine_minutes: dict = field(default_factory=dict)  # (machine, minute) -> [work, idle, alarm, wait, work_att, idle_att, cycles]
    n_clips: int = 0
    t_start: float = 0.0  # start of the current pass (file-relative seconds)
    frame: np.ndarray | None = None
    h: int = 0
    w: int = 0
    t: float = 0.0  # file-relative seconds of the current frame
    abs_t: float = 0.0  # wall-clock seconds
    dt: float = 0.0
    tracks: list = field(default_factory=list)  # confirmed tracks matched in this frame
    alive: set = field(default_factory=set)  # ids of all live tracks
    motion: Motion | None = None
    occupied: dict = field(default_factory=dict)  # zone name -> occupied now
    zone_counts: dict = field(default_factory=dict)  # zone name -> people inside now

    def minute(self, abs_t: float | None = None) -> int:
        return int((self.abs_t if abs_t is None else abs_t) // 60)

    def zone_row(self, zone: str):
        return self.zone_minutes.setdefault((zone, self.minute()), [0.0, 0.0, 0.0])

    def machine_row(self, name: str, abs_t: float | None = None):
        return self.machine_minutes.setdefault((name, self.minute(abs_t)), [0.0] * 6 + [0])

    def emit(self, type_: str, subject: str = "", value: str = "", clip: str = "", thumb: str = "", ts: float | None = None):
        self.events.append((self.abs_t if ts is None else ts, type_, subject, value, clip, thumb))

    def evidence(self, label: str, boxes=(), t_center: float | None = None):
        """Saves a frame (boxes drawn) and a clip around the event. Returns (clip, thumb), or ('', '') if disabled/capped."""
        if not self.opts["clips"] or self.out_dir is None or self.n_clips >= self.opts["max_clips_per_file"]:
            return "", ""
        self.n_clips += 1
        base = f"clips/{self.src['camera']}/{fmt_ts(self.abs_t)}_{re.sub(r'[^A-Za-z0-9_-]+', '_', label)}"
        (self.out_dir / base).parent.mkdir(parents=True, exist_ok=True)
        img = self.frame.copy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 3)
        cv2.imwrite(str(self.out_dir / (base + ".jpg")), cv2.resize(img, (480, max(1, int(self.h * 480 / self.w)))))
        save_clip(self.src["path"], self.t if t_center is None else t_center, self.out_dir / (base + ".mp4"),
                  self.opts["clip_pad_s"], self.opts["hw_decode"])
        return base + ".mp4", base + ".jpg"


class SegmentTracker:
    """Turns a boolean signal into segments that lasted >= min_duration_s; gaps up to gap_s are bridged.
    update() returns [('qualified', info)] once the threshold is crossed and [('ended', info)] when the segment is over."""

    def __init__(self, min_duration_s: float, gap_s: float):
        self.min_duration_s, self.gap_s = min_duration_s, gap_s
        self.start = self.last_on = None
        self.qualified, self.peak = False, 0

    def update(self, t: float, on: bool, value: float = 0):
        out = []
        if on:
            if self.start is None:
                self.start, self.peak, self.qualified = t, 0, False
            self.last_on, self.peak = t, max(self.peak, value)
            if not self.qualified and t - self.start >= self.min_duration_s:
                self.qualified = True
                out.append(("qualified", dict(start=self.start, peak=self.peak)))
        elif self.start is not None and t - self.last_on > self.gap_s:
            out += self.flush()
        return out

    def flush(self):
        out = [("ended", dict(start=self.start, end=self.last_on, peak=self.peak))] if self.start is not None and self.qualified else []
        self.start = self.last_on = None
        self.qualified, self.peak = False, 0
        return out


class CycleCounter:
    """Counts work segments (motion -> stop = one machining). A pause shorter than min_idle_s does not split a cycle;
    segments shorter than min_work_s are ignored."""

    def __init__(self, min_work_s: float = 3.0, min_idle_s: float = 1.0):
        self.min_work_s, self.min_idle_s = min_work_s, min_idle_s
        self.start = self.end = None

    def update(self, t: float, working: bool):
        done = []
        if working:
            if self.start is None:
                self.start = t
            self.end = None
        elif self.start is not None:
            if self.end is None:
                self.end = t
            if t - self.end >= self.min_idle_s:
                if self.end - self.start >= self.min_work_s:
                    done.append((self.start, self.end))
                self.start = self.end = None
        return done

    def flush(self, t: float):
        if self.start is None:
            return []
        end = self.end if self.end is not None else t
        out = [(self.start, end)] if end - self.start >= self.min_work_s else []
        self.start = self.end = None
        return out


def largest_group(tracks, dist_k: float):
    """Biggest set of people standing close together: foot points closer than dist_k x mean box height (single linkage)."""
    n = len(tracks)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            (xi, yi), (xj, yj) = tracks[i].pt, tracks[j].pt
            hi, hj = tracks[i].box[3] - tracks[i].box[1], tracks[j].box[3] - tracks[j].box[1]
            if math.hypot(xi - xj, yi - yj) <= dist_k * (hi + hj) / 2:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(tracks[i])
    return max(groups.values(), key=len, default=[])


class ZoneAnalyzer:
    """Zone enter/exit with dwell time, plus per-minute occupancy (time at the machine / absence are derived from it)."""

    def __init__(self, zones: list[Zone]):
        self.zones = zones
        self.in_zone: dict = {}
        self.last_seen = {z.name: -1e9 for z in zones}

    def _exit(self, ctx, key):
        ctx.emit("zone_exit", f"t{key[0]}", f"{key[1]} dwell={ctx.t - self.in_zone.pop(key):.0f}s")

    def update(self, ctx: Ctx):
        counts = {z.name: 0 for z in self.zones}
        for tr in ctx.tracks:
            for z in self.zones:
                key, inside = (tr.id, z.name), z.contains(*tr.pt, ctx.w, ctx.h)
                if inside:
                    counts[z.name] += 1
                    if key not in self.in_zone:
                        self.in_zone[key] = ctx.t
                        ctx.emit("zone_enter", f"t{tr.id}", z.name)
                elif key in self.in_zone:
                    self._exit(ctx, key)
        for key in [k for k in self.in_zone if k[0] not in ctx.alive]:
            self._exit(ctx, key)
        for z in self.zones:
            if counts[z.name]:
                self.last_seen[z.name] = ctx.t
            occ = ctx.t - self.last_seen[z.name] <= z.hold_s
            ctx.occupied[z.name], ctx.zone_counts[z.name] = occ, counts[z.name]
            row = ctx.zone_row(z.name)
            row[0] += ctx.dt
            row[1] += ctx.dt if occ else 0.0
            row[2] += ctx.dt * counts[z.name]

    def finish(self, ctx: Ctx):
        for key in list(self.in_zone):
            self._exit(ctx, key)


class LineAnalyzer:
    def __init__(self, lines_cfg: list[dict]):
        self.lines = [Line(l["name"], l["p1"], l["p2"], l.get("margin", 0.012)) for l in lines_cfg]

    def update(self, ctx: Ctx):
        for tr in ctx.tracks:
            for ln in self.lines:
                d = ln.update(tr.id, tr.pt, ctx.w, ctx.h)
                if d:
                    ctx.emit("line_cross", f"t{tr.id}", f"{ln.name}:{d}")

    def finish(self, ctx: Ctx):
        pass


class MachineAnalyzer:
    """WORK/IDLE of one machine from a signal lamp (colour) or from motion in a small ROI; optional cycle counting.
    If `zone` links an operator zone, work/idle time is also split into attended / unattended."""

    def __init__(self, cfg: dict):
        self.name, self.source, self.link = cfg["name"], cfg["source"], cfg.get("zone")
        self.roi = Zone(self.name, cfg["roi"])
        thr = cfg.get("motion_thr", 0.02)
        if self.source == "lamp":
            self.lamp = LampReader(self.roi, cfg.get("hues"), cfg.get("sat_min", 90), cfg.get("val_min", 140),
                                   cfg.get("min_frac", 0.08), cfg.get("window_s", 3.0), cfg.get("on_frac", 0.3))
            self.work_colors = set(cfg.get("work_colors") or ["green"])
            self.warm_s = self.lamp.window_s
        else:
            self.ms = MachineState(thr, cfg.get("window_s", 10.0), cfg.get("on_frac", 0.3), cfg.get("off_frac", 0.1))
            self.warm_s = self.ms.window_s
        cyc = cfg.get("cycle")
        self.cycles = CycleCounter(cyc.get("min_work_s", 3.0), cyc.get("min_idle_s", 1.0)) if cyc is not None else None
        # a motion machine needs a short window to resolve cycles; the 10 s window used for WORK/IDLE would blur them
        self.cs = MachineState(thr, cyc.get("window_s", 2.0), cyc.get("on_frac", 0.5), cyc.get("off_frac", 0.2)) \
            if cyc is not None and self.source == "motion" else None
        self.prev = self.colour = None

    def _cycle_events(self, ctx, done):
        for s, e in done:
            ctx.emit("cycle", self.name, f"{e - s:.1f}s", ts=ctx.src["start_ts"] + e)
            ctx.machine_row(self.name, ctx.src["start_ts"] + e)[6] += 1

    def update(self, ctx: Ctx):
        colour, cworking = None, None
        if self.source == "lamp":
            colour, _ = self.lamp.update(ctx.frame, ctx.t)
            state = "work" if colour in self.work_colors else "idle"
        else:
            ratio = ctx.motion.roi_ratio(self.roi, [t.box for t in ctx.tracks], (ctx.h, ctx.w))
            state, _ = self.ms.update(ratio, ctx.t)
            if self.cs:
                cworking = self.cs.update(ratio, ctx.t)[0] == "work"
        if ctx.t - ctx.t_start < self.warm_s:
            return  # warm-up: the readers fill their windows, nothing is counted or reported
        changed, self.prev = self.prev is not None and state != self.prev, state
        if colour is not None:
            if self.colour is not None and colour != self.colour:
                ctx.emit("lamp_color", self.name, colour)
            self.colour = colour
        row = ctx.machine_row(self.name)
        att = bool(self.link and ctx.occupied.get(self.link))
        if state == "work":
            row[0] += ctx.dt
            row[4] += ctx.dt if att else 0.0
        else:
            row[1] += ctx.dt
            row[5] += ctx.dt if att else 0.0
        if colour == "red":
            row[2] += ctx.dt
        elif colour == "yellow":
            row[3] += ctx.dt
        if changed:
            clip, thumb = ctx.evidence(f"{self.name}_{state}")
            ctx.emit("machine_state", self.name, state, clip, thumb)
        if self.cycles:
            self._cycle_events(ctx, self.cycles.update(ctx.t, cworking if cworking is not None else state == "work"))

    def finish(self, ctx: Ctx):
        if self.cycles:
            self._cycle_events(ctx, self.cycles.flush(ctx.t))


class GatheringAnalyzer:
    """2+ people standing close together outside the work zones for longer than min_duration_s."""

    def __init__(self, cfg: dict, zones: list[Zone]):
        self.min_people, self.dist_k = int(cfg.get("min_people", 2)), float(cfg.get("dist_k", 1.0))
        self.exclude = [z for z in zones if z.kind == "workstation"]
        self.seg = SegmentTracker(float(cfg.get("min_duration_s", 180)), float(cfg.get("gap_s", 10)))
        self.n, self.ev = 0, ("", "")

    def _report(self, ctx, info):
        self.n += 1
        ctx.emit("gathering", f"g{self.n}", f"people={info['peak']} duration={info['end'] - info['start']:.0f}s",
                 *self.ev, ts=ctx.src["start_ts"] + info["start"])
        self.ev = ("", "")

    def update(self, ctx: Ctx):
        cand = [t for t in ctx.tracks if not any(z.contains(*t.pt, ctx.w, ctx.h) for z in self.exclude)]
        group = largest_group(cand, self.dist_k)
        for kind, info in self.seg.update(ctx.t, len(group) >= self.min_people, len(group)):
            if kind == "qualified":
                self.ev = ctx.evidence("gathering", [t.box for t in group])
            else:
                self._report(ctx, info)

    def finish(self, ctx: Ctx):
        for _, info in self.seg.flush():
            self._report(ctx, info)


def build_analyzers(cam: dict, zones: list[Zone], pose=None, workpiece_detect=None, fps: float = 3.0) -> list:
    """cam must be normalised (config.normalize_camera). Order matters: zones first (machines read ctx.occupied)."""
    out: list = [ZoneAnalyzer(zones)]
    if cam.get("lines"):
        out.append(LineAnalyzer(cam["lines"]))
    out += [MachineAnalyzer(m) for m in cam["machines"]]
    if cam.get("gathering") is not None:
        out.append(GatheringAnalyzer(cam["gathering"], zones))
    if cam.get("inactivity") is not None:
        from .inactivity import InactivityAnalyzer  # optional heavy dependency (MediaPipe) only when configured
        out.append(InactivityAnalyzer(cam["inactivity"], zones, pose))
    if cam.get("workpieces") is not None:
        from .workpieces import WorkpieceAnalyzer  # optional heavy dependencies (RF-DETR, supervision)
        out.append(WorkpieceAnalyzer(cam["workpieces"], workpiece_detect, fps))
    return out
