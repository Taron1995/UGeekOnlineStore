"""Workpiece counting: RF-DETR detector (fine-tuned on your parts) -> own tracker -> supervision LineZone."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .analyzers import Ctx
from .tracker import Tracker

SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}  # Apache-2.0 sizes


class RFDETRDetector:
    """Thin adapter: frame (BGR) -> [(x1, y1, x2, y2, score)] for the wanted classes.

    weights: a checkpoint you trained (`python -m videoanalytics train ...`, e.g. checkpoint_best_total.pth); without it the
    COCO-pretrained model of the given size is used (downloaded by rfdetr to ~/.roboflow/models on first use)."""

    def __init__(self, weights: str | None = None, model: str = "nano", classes=None, threshold: float = 0.5,
                 device: str | None = None, trust_checkpoint: bool = False):
        try:
            import rfdetr
        except ImportError as e:
            raise RuntimeError("workpiece counting needs: pip install rfdetr supervision (PyTorch is a dependency)") from e
        kw = {"device": device} if device else {}
        if weights:
            self.model = rfdetr.RFDETR.from_checkpoint(weights, trust_checkpoint=trust_checkpoint, **kw)
        else:
            self.model = getattr(rfdetr, SIZES[model])(**kw)
        self.classes = set(classes) if classes else None
        self.threshold = threshold

    def detect(self, frame):
        det = self.model.predict(np.ascontiguousarray(frame[:, :, ::-1]), threshold=self.threshold, include_source_image=False)
        names = det.data.get("class_name") if det.data else None
        out = []
        for i, (box, score) in enumerate(zip(det.xyxy, det.confidence)):
            if self.classes is None or (names is not None and str(names[i]) in self.classes):
                out.append((*map(float, box), float(score)))
        return out


class WorkpieceAnalyzer:
    """Counts workpieces crossing a line (e.g. the exit of the machine zone). `count`: 'out' (default), 'in' or 'both' --
    'in' means crossing from the right to the left side of p1 -> p2, like the people lines. Needs enough fps for the part
    speed: set a higher camera-level `fps` (4-6) for these cameras."""

    def __init__(self, cfg: dict, detect, fps: float = 3.0):
        try:
            import supervision as sv
        except ImportError as e:
            raise RuntimeError("workpiece counting needs: pip install supervision rfdetr") from e
        if detect is None:
            raise RuntimeError("workpieces: no detector (RF-DETR) available")
        self.sv, self.detect = sv, detect
        ln = cfg["line"]
        self.name, self.p1, self.p2 = ln.get("name", "workpiece_line"), ln["p1"], ln["p2"]
        self.count = cfg.get("count", "out")
        # supervision's own ByteTrack is deprecated (removed in 0.31); parts are small and fast, hence the looser matching
        self.tracker = Tracker(iou_thr=0.1, max_missed=max(2, int(fps)), min_hits=int(cfg.get("min_hits", 2)), dist_k=1.5)
        self.zone = None

    def update(self, ctx: Ctx):
        sv = self.sv
        if self.zone is None:
            self.zone = sv.LineZone(start=sv.Point(self.p1[0] * ctx.w, self.p1[1] * ctx.h),
                                    end=sv.Point(self.p2[0] * ctx.w, self.p2[1] * ctx.h),
                                    triggering_anchors=[sv.Position.CENTER])
        tracks = self.tracker.update(self.detect(ctx.frame))
        if not tracks:
            return
        det = sv.Detections(xyxy=np.array([t.box for t in tracks], dtype=float), confidence=np.ones(len(tracks)),
                            class_id=np.zeros(len(tracks), dtype=int), tracker_id=np.array([t.id for t in tracks]))
        crossed_in, crossed_out = self.zone.trigger(det)
        for i, t in enumerate(tracks):
            if crossed_in[i] and self.count in ("in", "both"):
                ctx.emit("workpiece", self.name, "in")
            if crossed_out[i] and self.count in ("out", "both"):
                ctx.emit("workpiece", self.name, "out")

    def finish(self, ctx: Ctx):
        pass


def iou_xyxy(a, b) -> float:
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match_boxes(gt, dets, iou_thr: float = 0.5):
    """Greedy one-to-one matching, most confident detection first. gt [(x1,y1,x2,y2)], dets [(x1,y1,x2,y2,score)] -> (tp, fp, fn)."""
    free, tp = list(range(len(gt))), 0
    for d in sorted(dets, key=lambda d: -d[4]):
        best = max(free, key=lambda i: iou_xyxy(gt[i], d), default=None)
        if best is not None and iou_xyxy(gt[best], d) >= iou_thr:
            free.remove(best)
            tp += 1
    return tp, len(dets) - tp, len(free)


def evaluate_detector(detect, dataset_dir: str, split: str = "valid", max_images: int = 40, iou_thr: float = 0.5) -> dict:
    """Recall / precision of detect(frame) -> [(x1,y1,x2,y2,score)] on the annotated images of one dataset split."""
    root = Path(dataset_dir) / split
    data = json.loads((root / "_annotations.coco.json").read_text(encoding="utf-8"))
    boxes: dict = {}
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]
        boxes.setdefault(a["image_id"], []).append((x, y, x + w, y + h))
    tp = fp = fn = n = 0
    for im in data["images"][:max_images]:
        frame = cv2.imread(str(root / im["file_name"]))
        if frame is None:
            continue
        a, b, c = match_boxes(boxes.get(im["id"], []), detect(frame), iou_thr)
        tp, fp, fn, n = tp + a, fp + b, fn + c, n + 1
    rec, prec = tp / max(1, tp + fn), tp / max(1, tp + fp)
    return dict(images=n, parts=tp + fn, recall=rec, precision=prec, f1=2 * rec * prec / max(1e-9, rec + prec))
