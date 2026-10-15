"""Signal-lamp (stack light) colour reader: HSV classification of a small ROI plus temporal smoothing."""
from __future__ import annotations

from collections import Counter, deque

import cv2
import numpy as np

from .zones import Zone

HUES = {"red": [(0, 8), (170, 179)], "yellow": [(15, 38)], "green": [(40, 90)]}  # OpenCV hue scale 0..179


class LampReader:
    """A pixel counts as 'lit' when it is both saturated and bright; the lit pixels vote for red/yellow/green.

    The label is the dominant non-off colour of the last `window_s` seconds (it needs `on_frac` of the samples, so a
    blinking lamp still reads as its colour). Tune sat_min / val_min / min_frac with `snap`, which prints the scores."""

    def __init__(self, roi: Zone, hues=None, sat_min: int = 90, val_min: int = 140, min_frac: float = 0.08,
                 window_s: float = 3.0, on_frac: float = 0.3):
        self.roi, self.sat_min, self.val_min, self.min_frac = roi, sat_min, val_min, min_frac
        self.window_s, self.on_frac = window_s, on_frac
        self.hues = {c: [tuple(r) for r in rs] for c, rs in (hues or HUES).items()}
        self.buf: deque = deque()
        self.label = "off"
        self._cache = None  # (w, h, (x0, y0, x1, y1), mask crop)

    def _geometry(self, w: int, h: int):
        if self._cache is None or self._cache[:2] != (w, h):
            pts = (self.roi.poly * [w, h]).astype(np.int32)
            x0, y0 = max(0, int(pts[:, 0].min())), max(0, int(pts[:, 1].min()))
            x1, y1 = min(w, int(pts[:, 0].max()) + 1), min(h, int(pts[:, 1].max()) + 1)
            self._cache = (w, h, (x0, y0, x1, y1), self.roi.mask(w, h)[y0:y1, x0:x1])
        return self._cache[2], self._cache[3]

    def scores(self, frame: np.ndarray) -> dict:
        """Share of the ROI area that is lit, per colour."""
        (x0, y0, x1, y1), mask = self._geometry(frame.shape[1], frame.shape[0])
        if x1 <= x0 or y1 <= y0:
            return {c: 0.0 for c in self.hues}
        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        lit = (hsv[..., 1] >= self.sat_min) & (hsv[..., 2] >= self.val_min) & (mask > 0)
        area = max(1, int(np.count_nonzero(mask)))
        out = {}
        for colour, ranges in self.hues.items():
            sel = np.zeros_like(lit)
            for lo, hi in ranges:
                sel |= (hsv[..., 0] >= lo) & (hsv[..., 0] <= hi)
            out[colour] = float(np.count_nonzero(lit & sel)) / area
        return out

    def classify(self, frame: np.ndarray) -> str:
        s = self.scores(frame)
        best = max(s, key=s.get)
        return best if s[best] >= self.min_frac else "off"

    def update(self, frame: np.ndarray, t: float):
        """Returns (smoothed label, changed)."""
        self.buf.append((t, self.classify(frame)))
        while self.buf and t - self.buf[0][0] > self.window_s:
            self.buf.popleft()
        counts = Counter(lab for _, lab in self.buf)
        lit = [c for c in counts if c != "off"]
        top = max(lit, key=lambda c: counts[c], default=None)
        new = top if top and counts[top] >= self.on_frac * len(self.buf) else "off"
        changed, self.label = new != self.label, new
        return new, changed
