"""Person detector: YOLOX (Apache 2.0) via ONNX Runtime, fully offline."""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

PERSON_CLASS = 0


def _grids(size: int, strides=(8, 16, 32)):
    gs, ss = [], []
    for s in strides:
        gy, gx = np.meshgrid(np.arange(size // s), np.arange(size // s), indexing="ij")
        gs.append(np.stack([gx, gy], -1).reshape(-1, 2))
        ss.append(np.full((gs[-1].shape[0], 1), s))
    return np.concatenate(gs).astype(np.float32), np.concatenate(ss).astype(np.float32)


def postprocess(raw: np.ndarray, grid, stride, ratio: float, frame_hw, conf: float, nms: float):
    """raw: (N, 85) YOLOX head output -> [(x1, y1, x2, y2, score)] persons in frame pixels."""
    h, w = frame_hw
    xy = (raw[:, :2] + grid) * stride
    wh = np.exp(raw[:, 2:4]) * stride
    scores = raw[:, 4] * raw[:, 5 + PERSON_CLASS]
    keep = scores > conf
    if not keep.any():
        return []
    xy, wh, scores = xy[keep], wh[keep], scores[keep]
    boxes = np.concatenate([xy - wh / 2, wh], 1) / ratio  # x, y, w, h in frame px
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf, nms)
    res = []
    for i in np.array(idx).flatten():
        x, y, bw, bh = boxes[i]
        res.append((max(0.0, float(x)), max(0.0, float(y)), min(float(w), float(x + bw)),
                    min(float(h), float(y + bh)), float(scores[i])))
    return res


class PersonDetector:
    def __init__(self, model_path: str, conf: float = 0.35, nms: float = 0.45, threads: int = 1):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads  # 1 thread x N worker processes scales best on CPU
        self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape  # [1, 3, H, W]; 416 for tiny/nano, 640 for s/m/l/x
        self.size = int(shape[2]) if isinstance(shape[2], int) else 640
        self.conf, self.nms = conf, nms
        self.grid, self.stride = _grids(self.size)

    def detect(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        r = min(self.size / h, self.size / w)
        resized = cv2.resize(frame, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)  # YOLOX: BGR, 0-255
        raw = self.sess.run(None, {self.inp: blob})[0][0]
        return postprocess(raw, self.grid, self.stride, r, (h, w), self.conf, self.nms)
