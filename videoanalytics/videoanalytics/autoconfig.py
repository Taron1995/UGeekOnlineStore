"""Scene learning: find work stations, machines and signal lamps from the footage itself (no cameras.yaml needed).

A few minutes sampled across a camera's footage are enough to see
* where people stay put      -> station zones      (stationary foot points, clustered)
* where something keeps moving -> motion machines   (people masked out; clock overlays and borders ignored)
* where small coloured lights switch -> lamp machines (a stack light cycles colours, a blinking lamp is on part of the time)
Everything is a heuristic: the result is written as an editable cameras.auto.yaml and drawn on a snapshot so it can be checked."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .lamp import HUES
from .tracker import Tracker
from .video import active_intervals, open_capture
from .zones import Motion

GRID = (64, 36)  # dwell map cells (x, y)
MOTION_GRID = (160, 90)  # Motion mask size
LAMP_RES = (320, 180)  # lamp colour maps
MIN_VISIBLE = 8  # frames a pixel must be visible (not behind a person) before its statistics mean anything


def sample_windows(srcs, total_s: float = 360.0, k: int = 6, hours=None):
    """[(src, start_s, length_s)] spread evenly over the (active) footage of one camera; the whole footage if it is short."""
    # a file that does not say how long it is is only trusted for its first minute (its duration is a guess)
    spans = [(s, a, b) for s in srcs for a, b in active_intervals(s["start_ts"], s["duration"] if s.get("duration_known", True) else 60.0, hours) if b > a]
    total = sum(b - a for _, a, b in spans)
    if total <= 0:
        return []
    if total <= total_s:
        return [(s, a, b - a) for s, a, b in spans]
    w, out = total_s / k, []
    for i in range(k):
        start, acc = max(0.0, (i + 0.5) * total / k - w / 2), 0.0
        for s, a, b in spans:
            if start < acc + (b - a):
                t0 = max(a, min(a + (start - acc), b - w))  # keep the whole window inside this recording when it fits
                out.append((s, t0, min(w, b - t0)))
                break
            acc += b - a
    return out


def window_frames(src: dict, start_s: float, length_s: float, fps: float):
    """Yields (t_rel, frame, dt) at about `fps` from one window of a recording."""
    cap = open_capture(src["path"])
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if 1 <= src_fps <= 120 else 25.0
    step = max(1, round(src_fps / fps))
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
    fi = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    while cap.grab():
        fi += 1
        if (fi - 1) % step:
            continue
        t = (fi - 1) / src_fps
        if t >= start_s + length_s:
            break
        ok, frame = cap.retrieve()
        if ok:
            yield t, frame, step / src_fps
    cap.release()


def _clear_boxes(mask: np.ndarray, boxes, frame_hw, pad: float = 0.15):
    """Zero the part of `mask` (any resolution) covered by person boxes, so people never count as machines or lamps."""
    mh, mw = mask.shape[:2]
    fh, fw = frame_hw
    for x1, y1, x2, y2, *_ in boxes:
        px, py = pad * (x2 - x1), pad * (y2 - y1)
        cv2.rectangle(mask, (int((x1 - px) / fw * mw), int((y1 - py) / fh * mh)),
                      (int((x2 + px) / fw * mw) + 1, int((y2 + py) / fh * mh) + 1), 0, -1)


def _norm_poly(x0, y0, x1, y1, w, h, pad_x=0.0, pad_y=0.0):
    """Cell bbox (x1, y1 exclusive) on a w x h grid -> normalised rectangle polygon, padded and clipped to 0..1."""
    cl = lambda v: round(float(min(1.0, max(0.0, v))), 4)  # plain floats: the config is dumped to YAML
    ax, bx, ay, by = cl(x0 / w - pad_x), cl(x1 / w + pad_x), cl(y0 / h - pad_y), cl(y1 / h + pad_y)
    return [[ax, ay], [bx, ay], [bx, by], [ax, by]]


class SceneLearner:
    """Accumulates evidence over all sampled frames of one camera."""

    def __init__(self, dwell_s: float = 15.0, radius_k: float = 1.0):
        self.dwell_s, self.radius_k = dwell_s, radius_k
        self.heat = np.zeros(GRID[::-1], np.float32)  # seconds of 'standing about' per cell
        self.hsum = np.zeros(GRID[::-1], np.float32)  # same, weighted by the person's height (share of frame height)
        self.windows: list = []  # per window: (motion frequency map, frames)
        self.lamp = {c: np.zeros(LAMP_RES[::-1], np.float32) for c in HUES}
        self.lamp_valid = np.zeros(LAMP_RES[::-1], np.float32)  # frames in which the pixel was not hidden behind a person
        self.lamp_n = 0
        self.frames = 0
        self.person_ids = 0
        self.frame_hw = (0, 0)

    def window(self, frames, detect, fps: float):
        """frames: iterable of (t, frame, dt)."""
        tracker, motion = Tracker(0.2, max(2, int(2 * fps)), 2), Motion()
        anchors: dict = {}
        mfreq, mvalid, mn = np.zeros(MOTION_GRID[::-1], np.float32), np.zeros(MOTION_GRID[::-1], np.float32), 0
        for t, frame, dt in frames:
            H, W = frame.shape[:2]
            self.frame_hw = (H, W)
            boxes = detect(frame)
            tracks = tracker.update(boxes)
            for tr in tracks:  # a person who stays within ~one body height for dwell_s is 'at a station'
                x, y = tr.pt
                h = tr.box[3] - tr.box[1]
                a = anchors.get(tr.id)
                if a is None or math.hypot(x - a[0], y - a[1]) > self.radius_k * max(h, a[3]):
                    anchors[tr.id] = (x, y, t, h)
                elif t - a[2] >= self.dwell_s:
                    cx, cy = min(GRID[0] - 1, int(a[0] / W * GRID[0])), min(GRID[1] - 1, int(max(0, a[1]) / H * GRID[1]))
                    self.heat[cy, cx] += dt
                    self.hsum[cy, cx] += dt * a[3] / H
            first = motion.prev is None
            motion.step(frame)
            if not first:  # pixels hidden behind a person are unknown: counted neither as moving nor as still
                seen = np.full(MOTION_GRID[::-1], 255, np.uint8)
                _clear_boxes(seen, boxes, (H, W))
                mfreq += (motion.mask > 0) & (seen > 0)
                mvalid += seen > 0
                mn += 1
            small = cv2.resize(frame, LAMP_RES, interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            lit = (hsv[..., 1] >= 140) & (hsv[..., 2] >= 200)
            seen = np.full(LAMP_RES[::-1], 255, np.uint8)
            _clear_boxes(seen, boxes, (H, W))
            ok = seen > 0
            for colour, ranges in HUES.items():
                sel = np.zeros(lit.shape, bool)
                for lo, hi in ranges:
                    sel |= (hsv[..., 0] >= lo) & (hsv[..., 0] <= hi)
                self.lamp[colour] += lit & sel & ok
            self.lamp_valid += ok
            self.lamp_n += 1
            self.frames += 1
        self.person_ids += tracker._next - 1
        if mn >= 3:
            f = mfreq / np.maximum(mvalid, 1)
            f[mvalid < max(MIN_VISIBLE, 0.05 * mn)] = 0.0  # hardly ever visible: no evidence
            self.windows.append((f, mn))

    # ---- results ---------------------------------------------------------------------------------------------------
    def spots(self, min_total_s: float = 30.0, max_spots: int = 8):
        if self.heat.max() <= 0:
            return []
        blur = cv2.GaussianBlur(self.heat, (0, 0), 1.2)
        mask = (blur >= 0.2 * blur.max()).astype(np.uint8)
        n, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        H, W = self.frame_hw
        out = []
        for i in range(1, n):
            comp = labels == i
            total = float(self.heat[comp].sum())
            if total < min_total_s:
                continue
            ys, xs = np.where(comp)
            hnorm = float(self.hsum[comp].sum() / max(total, 1e-6))  # mean person height / frame height
            out.append(dict(total_s=round(total, 1),
                            polygon=_norm_poly(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, *GRID, 0.5 * hnorm * H / W, 0.5 * hnorm)))
        return sorted(out, key=lambda s: -s["total_s"])[:max_spots]

    def motion_regions(self, min_freq: float = 0.2, max_regions: int = 4):
        """Regions that keep moving (outside people) in at least half of the sampled windows."""
        maps = [f for f, _ in self.windows]
        if not maps:
            return []
        f = np.median(maps, axis=0) if len(maps) >= 3 else np.mean(maps, axis=0)
        mask = cv2.morphologyEx((f >= min_freq).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        gw, gh = MOTION_GRID
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 8 or area > 0.25 * gw * gh or w * h > 0.35 * gw * gh:
                continue
            in_band = y + h <= 0.12 * gh or y >= 0.88 * gh  # time stamp / channel name overlays live at the top and bottom
            if in_band and w >= 3 * h:
                continue
            out.append(dict(freq=round(float(f[labels == i].mean()), 3), area=int(area),
                            polygon=_norm_poly(x, y, x + w, y + h, gw, gh, 1.0 / gw, 1.0 / gh)))
        return sorted(out, key=lambda r: -r["freq"] * math.sqrt(r["area"]))[:max_regions]

    def lamps(self, min_freq: float = 0.02, max_lamps: int = 4):
        """Small areas where red / yellow / green switch on and off: a stack light, or a blinking lamp."""
        if self.lamp_n < 5:
            return []
        trusted = self.lamp_valid >= max(MIN_VISIBLE, 0.05 * self.lamp_n)
        freq = {c: np.where(trusted, a / np.maximum(self.lamp_valid, 1), 0.0) for c, a in self.lamp.items()}
        any_mask = (np.max(list(freq.values()), axis=0) >= min_freq).astype(np.uint8)
        joined = cv2.dilate(any_mask, np.ones((15, 7), np.uint8))  # a stack light is a vertical row of lenses
        n, labels, _, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
        gw, gh = LAMP_RES
        out = []
        for i in range(1, n):
            cells = (labels == i) & (any_mask > 0)
            ys, xs = np.where(cells)
            if len(xs) < 3:
                continue
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            if (x1 - x0) * (y1 - y0) > 0.03 * gw * gh:
                continue
            peak = {c: float(freq[c][cells].max()) for c in freq}
            seen = [c for c, v in peak.items() if v >= 0.03]
            changing = any(0.03 <= peak[c] <= 0.95 for c in seen)  # something switches: always-on LEDs and static signs say nothing
            blinking = any(0.05 <= peak[c] <= 0.95 for c in seen)
            in_osd_band = y1 <= 0.10 * gh or y0 >= 0.92 * gh  # a blinking red REC dot sits in the overlay area
            if not (changing and (len(seen) >= 2 or (blinking and not in_osd_band))):
                continue
            out.append(dict(colours=seen, peak={c: round(v, 3) for c, v in peak.items()}, polygon=_norm_poly(x0, y0, x1, y1, gw, gh, 2.0 / gw, 2.0 / gh)))
        return sorted(out, key=lambda m: -len(m["colours"]))[:max_lamps]


def _centre(poly):
    a = np.array(poly)
    return a.mean(axis=0)


def build_camera_config(spots, motion, lamps, link_dist: float = 0.3) -> dict:
    """Camera config from what was found. Machines are linked to the nearest station zone (so the report can show work
    without an operator). People counting, gatherings and inactivity are always on, with conservative durations."""
    cam: dict = {"zones": [{"name": f"spot{i + 1}", "kind": "workstation", "polygon": s["polygon"]} for i, s in enumerate(spots)]}
    machines = [{"name": f"lamp{i + 1}", "source": "lamp", "roi": m["polygon"], "work_colors": ["green"]} for i, m in enumerate(lamps)]
    machines += [{"name": f"motion{i + 1}", "source": "motion", "roi": m["polygon"], "motion_thr": 0.02, "window_s": 10} for i, m in enumerate(motion)]
    for m in machines:
        if spots:
            # the machine sits above the operator's feet, so vertical distance counts half
            d = [float(np.hypot(*((_centre(m["roi"]) - _centre(s["polygon"])) * [1.0, 0.5]))) for s in spots]
            if min(d) <= link_dist:
                m["zone"] = f"spot{int(np.argmin(d)) + 1}"
    cam["machines"] = machines
    cam["gathering"] = {"min_people": 2, "min_duration_s": 180}
    cam["inactivity"] = {"method": "motion", "min_duration_s": 300}
    return cam


def learn_scene(srcs, detect, *, minutes: float = 6.0, fps: float = 2.0, windows: int = 6, hours=None, dwell_s: float = 15.0):
    """Learns one camera's setup from its recordings. Returns (camera_config, info, preview) where preview = (source, second)
    of a frame to draw the overlay on."""
    wins = sample_windows(srcs, minutes * 60.0, windows, hours)
    if not wins:
        raise ValueError("no footage inside the analysed hours")
    learner = SceneLearner(dwell_s)
    for src, start, length in wins:
        learner.window(window_frames(src, start, length, fps), detect, fps)
    spots, motion, lamps = learner.spots(), learner.motion_regions(), learner.lamps()
    cfg = build_camera_config(spots, motion, lamps)
    info = dict(sampled_s=round(sum(w[2] for w in wins)), windows=len(wins), frames=learner.frames, person_tracks=learner.person_ids,
                spots=spots, motion=motion, lamps=lamps)
    src0, start0, length0 = wins[len(wins) // 2]
    return cfg, info, (src0, start0 + length0 / 2)
