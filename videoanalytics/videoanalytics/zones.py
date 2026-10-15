"""Zones, counting lines (normalized 0-1 coordinates) and frame-difference motion / machine state."""
from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np


class Zone:
    def __init__(self, name: str, polygon: list[list[float]], kind: str = "workstation"):
        self.name, self.kind = name, kind
        self.poly = np.array(polygon, np.float32)

    def contains(self, x: float, y: float, w: int, h: int) -> bool:
        pts = (self.poly * [w, h]).astype(np.float32)
        return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0

    def mask(self, w: int, h: int) -> np.ndarray:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [(self.poly * [w, h]).astype(np.int32)], 255)
        return m


class Line:
    """Counting line p1->p2 (normalized). 'in' = a track moves from the right side to the left side of p1->p2."""

    def __init__(self, name: str, p1, p2, margin: float = 0.012):
        self.name, self.p1, self.p2, self.margin = name, p1, p2, margin
        self.state: dict = {}  # track id -> last side seen clearly away from the line (+1 right / -1 left)

    def dist(self, pt, w, h) -> float:
        """Signed distance in px; positive = right of p1->p2 (image y axis points down)."""
        (x1, y1), (x2, y2) = (self.p1[0] * w, self.p1[1] * h), (self.p2[0] * w, self.p2[1] * h)
        return ((x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)) / max(1e-6, math.hypot(x2 - x1, y2 - y1))

    def update(self, tid: int, pt, w: int, h: int):
        """Returns 'in', 'out' or None. A dead-band (margin x frame size) stops jitter on the line from counting."""
        d = self.dist(pt, w, h)
        if abs(d) < self.margin * max(w, h):
            return None
        side, prev = (1 if d > 0 else -1), self.state.get(tid)
        self.state[tid] = side
        if prev is None or prev == side:
            return None
        return "in" if prev > 0 else "out"


class Motion:
    """Frame-to-frame difference on a small grayscale image (works at any processing fps)."""

    def __init__(self, size=(160, 90), pix_thr: int = 20):
        self.size, self.pix_thr, self.prev, self.mask = size, pix_thr, None, None

    def step(self, frame: np.ndarray) -> float:
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA),
                                          cv2.COLOR_BGR2GRAY), (5, 5), 0)
        if self.prev is None:
            self.mask = np.zeros_like(g)
        else:
            self.mask = (cv2.absdiff(g, self.prev) > self.pix_thr).astype(np.uint8) * 255
        self.prev = g
        return float(np.count_nonzero(self.mask)) / self.mask.size

    def roi_ratio(self, roi: Zone, exclude_boxes=(), frame_hw=None) -> float:
        """Share of changed pixels inside roi, ignoring person boxes (so walking people != machine work)."""
        sw, sh = self.size
        m = self.mask.copy()
        if frame_hw:
            fh, fw = frame_hw
            for x1, y1, x2, y2 in exclude_boxes:
                cv2.rectangle(m, (int(x1 / fw * sw), int(y1 / fh * sh)),
                              (int(x2 / fw * sw) + 1, int(y2 / fh * sh) + 1), 0, -1)
        rm = roi.mask(sw, sh)
        area = max(1, int(np.count_nonzero(rm)))
        return float(np.count_nonzero(cv2.bitwise_and(m, rm))) / area


class MachineState:
    """WORK/IDLE from the share of 'active' motion samples in a sliding window (with hysteresis)."""

    def __init__(self, motion_thr: float = 0.02, window_s: float = 10.0, on_frac: float = 0.3,
                 off_frac: float = 0.1):
        self.motion_thr, self.window_s, self.on_frac, self.off_frac = motion_thr, window_s, on_frac, off_frac
        self.buf: deque = deque()  # (t, active)
        self.state = "idle"

    def update(self, ratio: float, t: float):
        """Returns (state, changed)."""
        self.buf.append((t, ratio > self.motion_thr))
        while self.buf and t - self.buf[0][0] > self.window_s:
            self.buf.popleft()
        frac = sum(a for _, a in self.buf) / len(self.buf)
        new = self.state
        if self.state == "idle" and frac >= self.on_frac:
            new = "work"
        elif self.state == "work" and frac <= self.off_frac:
            new = "idle"
        changed, self.state = new != self.state, new
        return self.state, changed
