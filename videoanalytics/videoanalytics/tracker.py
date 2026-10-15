"""IoU tracker with constant-velocity prediction and a centre-distance second pass.

No Kalman filter and no Hungarian solver: greedy matching is enough for sparse scenes and keeps the
code dependency-free. At 2-3 fps (needed for multi-day footage) plain IoU matching fragments tracks badly,
so each track is moved along its smoothed velocity before matching, and leftovers are matched by distance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def foot(box):
    return ((box[0] + box[2]) / 2, box[3])


def centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


@dataclass
class Track:
    id: int
    box: tuple
    hits: int = 1
    missed: int = 0
    pt: tuple = (0.0, 0.0)  # current foot point
    v: tuple = (0.0, 0.0)  # smoothed centre velocity, px per update

    def predicted(self):
        k, b = self.missed + 1, self.box
        return (b[0] + self.v[0] * k, b[1] + self.v[1] * k, b[2] + self.v[0] * k, b[3] + self.v[1] * k)


class Tracker:
    def __init__(self, iou_thr: float = 0.2, max_missed: int = 30, min_hits: int = 2, dist_k: float = 1.0):
        self.iou_thr, self.max_missed, self.min_hits, self.dist_k = iou_thr, max_missed, min_hits, dist_k
        self.tracks: list[Track] = []
        self._next = 1

    def _match(self, t: Track, box: tuple):
        c0, c1, m = centre(t.box), centre(box), t.missed + 1
        meas = ((c1[0] - c0[0]) / m, (c1[1] - c0[1]) / m)
        t.v = meas if t.hits == 1 else (0.5 * t.v[0] + 0.5 * meas[0], 0.5 * t.v[1] + 0.5 * meas[1])
        t.box, t.hits, t.missed = box, t.hits + 1, 0
        t.pt = foot(box)

    def update(self, dets) -> list[Track]:
        """dets: [(x1,y1,x2,y2,score)]. Returns confirmed tracks matched in this update."""
        dets = sorted(dets, key=lambda d: -d[4])
        preds = [t.predicted() for t in self.tracks]
        used_t, used_d, matched = set(), set(), []

        def take(ti, di):
            self._match(self.tracks[ti], tuple(dets[di][:4]))
            used_t.add(ti), used_d.add(di), matched.append(self.tracks[ti])

        pairs = sorted(((iou(preds[ti], d[:4]), ti, di) for ti in range(len(self.tracks)) for di, d in enumerate(dets)),
                       reverse=True)
        for s, ti, di in pairs:
            if s < self.iou_thr:
                break
            if ti not in used_t and di not in used_d:
                take(ti, di)
        if self.dist_k > 0:  # second pass: same-sized box within dist_k box-heights of the predicted position
            cand = []
            for ti in range(len(self.tracks)):
                if ti in used_t:
                    continue
                pc, ph = centre(preds[ti]), max(1.0, preds[ti][3] - preds[ti][1])
                for di, d in enumerate(dets):
                    if di in used_d:
                        continue
                    dc = centre(d)
                    dist, hr = math.hypot(pc[0] - dc[0], pc[1] - dc[1]) / ph, (d[3] - d[1]) / ph
                    if dist < self.dist_k and 0.6 < hr < 1.6:
                        cand.append((dist, ti, di))
            for _, ti, di in sorted(cand):
                if ti not in used_t and di not in used_d:
                    take(ti, di)
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.missed += 1
        for di, d in enumerate(dets):
            if di not in used_d:
                t = Track(self._next, tuple(d[:4]), pt=foot(d))
                self._next += 1
                self.tracks.append(t)
                matched.append(t)
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return [t for t in matched if t.hits >= self.min_hits]
