"""'Seated without movement': per person, body activity over time with MediaPipe Pose (or a frame-difference fallback)."""
from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .analyzers import Ctx, SegmentTracker
from .zones import Zone

# MediaPipe pose landmark ids. Legs are usually hidden behind a desk or machine, so activity uses the upper body.
NOSE, SHOULDERS, ELBOWS, WRISTS, HIPS, KNEES = 0, (11, 12), (13, 14), (15, 16), (23, 24), (25, 26)
UPPER_BODY = (NOSE, *SHOULDERS, *ELBOWS, *WRISTS, *HIPS)
DEFAULT_POSE_MODEL = "models/pose_landmarker_lite.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"


def landmark_activity(prev: np.ndarray, cur: np.ndarray, box_h: float, vis: float = 0.5, top_k: int = 4):
    """Mean of the top_k largest upper-body landmark displacements, as a share of the person's height.
    A typing or turning person moves a few landmarks a lot, so the mean over all 33 would hide it."""
    d = [math.hypot(cur[i][0] - prev[i][0], cur[i][1] - prev[i][1]) / max(box_h, 1.0)
         for i in UPPER_BODY if prev[i][2] >= vis and cur[i][2] >= vis]
    return float(np.mean(sorted(d)[-top_k:])) if len(d) >= 3 else None


def posture(lm: np.ndarray, vis: float = 0.5):
    """'seated' when the knees are not clearly below the hips (relative to torso length), else 'standing'; None if unseen."""
    def mean_y(ids):
        ys = [lm[i][1] for i in ids if lm[i][2] >= vis]
        return float(np.mean(ys)) if ys else None
    sh, hip, knee = mean_y(SHOULDERS), mean_y(HIPS), mean_y(KNEES)
    if sh is None or hip is None or knee is None or hip - sh < 1e-3:
        return None
    return "seated" if (knee - hip) < 0.6 * (hip - sh) else "standing"


def crop_box(frame: np.ndarray, box, pad: float = 0.15):
    x1, y1, x2, y2 = box
    p = pad * (y2 - y1)
    h, w = frame.shape[:2]
    cx1, cy1, cx2, cy2 = max(0, int(x1 - p)), max(0, int(y1 - p)), min(w, int(x2 + p)), min(h, int(y2 + p))
    return (cx1, cy1, cx2, cy2), frame[cy1:cy2, cx1:cx2]


class PoseBackend:
    """MediaPipe PoseLandmarker (Tasks API) on a padded person crop. The .task model is not bundled with the wheel:
    `python -m videoanalytics models --pose` downloads it to models/."""

    def __init__(self, model_path: str | None = None):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as e:
            raise RuntimeError("inactivity.method 'pose' needs MediaPipe: pip install mediapipe "
                               "(or use method: motion)") from e
        path = Path(model_path or os.environ.get("VIDAN_POSE_MODEL") or DEFAULT_POSE_MODEL)
        if not path.exists():
            raise FileNotFoundError(f"pose model not found: {path}. Run: python -m videoanalytics models --pose")
        self._mp = mp
        self._lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)), running_mode=vision.RunningMode.IMAGE, num_poses=1,
            min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3))

    def landmarks(self, frame: np.ndarray, box):
        """(33, 3) array of x, y (frame pixels) and visibility, or None when no pose is found."""
        (cx1, cy1, cx2, cy2), crop = crop_box(frame, box)
        if crop.shape[0] < 24 or crop.shape[1] < 12:
            return None
        scale = max(1.0, 256.0 / crop.shape[0])  # small people (typical for CCTV) need upscaling for the pose model
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        res = self._lm.detect(img)
        if not res.pose_landmarks:
            return None
        arr = np.array([(l.x, l.y, l.visibility or 0.0) for l in res.pose_landmarks[0]], np.float32)
        arr[:, 0] = cx1 + arr[:, 0] * (cx2 - cx1)
        arr[:, 1] = cy1 + arr[:, 1] * (cy2 - cy1)
        return arr

    def close(self):
        self._lm.close()


class _Person:
    def __init__(self, min_duration_s: float, gap_s: float):
        self.seg = SegmentTracker(min_duration_s, gap_s)
        self.last_sample = -1e9
        self.prev = None  # previous landmarks (pose) or grayscale crop (motion)
        self.postures: Counter = Counter()
        self.ev = ("", "")
        self.zone = ""


class InactivityAnalyzer:
    """A person is 'inactive' while their activity stays below `motion_thr`; reported once it lasted min_duration_s.
    activity = share of body height the upper-body landmarks move per sample (pose) or share of changed pixels in the
    person's box (motion). Defaults are starting points: calibrate on your cameras."""

    def __init__(self, cfg: dict, zones: list[Zone], pose: PoseBackend | None):
        self.method = cfg.get("method", "motion")
        if self.method == "pose" and pose is None:
            raise RuntimeError("inactivity.method 'pose' needs a PoseBackend (MediaPipe)")
        self.pose = pose
        names = set(cfg.get("zones") or [])
        self.zones = [z for z in zones if z.name in names] if names else None
        self.sample_s = float(cfg.get("sample_period_s", 1.0))
        # measured on 60-83 px tall pedestrians with compression noise added: pose activity of a *still* person had
        # median 0.02 / p95 0.25 (useless at that size), the pixel-change share was exactly 0 -> motion is the default
        self.thr = float(cfg.get("motion_thr", 0.05 if self.method == "pose" else 0.01))
        self.min_duration_s, self.gap_s = float(cfg.get("min_duration_s", 300)), float(cfg.get("gap_s", 10))
        self.require_seated = bool(cfg.get("require_seated", False))
        self.people: dict[int, _Person] = {}
        self.n = 0

    def _zone_of(self, ctx, tr):
        if self.zones is None:
            return ""
        for z in self.zones:
            if z.contains(*tr.pt, ctx.w, ctx.h):
                return z.name
        return None

    def _measure(self, ctx, tr, p: _Person):
        """(activity or None, posture or None)"""
        box_h = tr.box[3] - tr.box[1]
        if self.method == "pose":
            lm = self.pose.landmarks(ctx.frame, tr.box)
            prev, p.prev = p.prev, lm
            if lm is None:
                return None, None
            return (landmark_activity(prev, lm, box_h) if prev is not None else None), posture(lm)
        _, crop = crop_box(ctx.frame, tr.box, 0.0)
        if crop.size == 0:
            return None, None
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(crop, (32, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY), (3, 3), 0)
        prev, p.prev = p.prev, g
        return (float((cv2.absdiff(g, prev) > 20).mean()) if prev is not None else None), None

    def _report(self, ctx, p: _Person, tid: int, info):
        self.n += 1
        post = p.postures.most_common(1)[0][0] if p.postures else "unknown"
        ctx.emit("inactive", f"t{tid}", f"duration={info['end'] - info['start']:.0f}s posture={post}"
                 + (f" zone={p.zone}" if p.zone else ""), *p.ev, ts=ctx.src["start_ts"] + info["start"])
        p.ev = ("", "")

    def _handle(self, ctx, p, tid, events, tr=None):
        for kind, info in events:
            if kind == "qualified" and tr is not None:
                p.ev = ctx.evidence(f"inactive_t{tid}", [tr.box])
            elif kind == "ended":
                self._report(ctx, p, tid, info)

    def update(self, ctx: Ctx):
        for tr in ctx.tracks:
            zone = self._zone_of(ctx, tr)
            if zone is None:
                continue
            p = self.people.setdefault(tr.id, _Person(self.min_duration_s, self.gap_s))
            if ctx.t - p.last_sample < self.sample_s:
                continue
            p.last_sample, p.zone = ctx.t, zone
            act, post = self._measure(ctx, tr, p)
            if post:
                p.postures[post] += 1
            still = act is not None and act < self.thr and (not self.require_seated or post == "seated")
            self._handle(ctx, p, tr.id, p.seg.update(ctx.t, still, 1), tr)
        for tid in [i for i in self.people if i not in ctx.alive]:  # person left the scene: close their segment
            p = self.people.pop(tid)
            self._handle(ctx, p, tid, p.seg.flush())

    def finish(self, ctx: Ctx):
        for tid, p in list(self.people.items()):
            self._handle(ctx, p, tid, p.seg.flush())
        self.people.clear()
