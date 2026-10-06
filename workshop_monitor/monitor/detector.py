"""Один детектор для классов: человек, телефон, заготовка. RF-DETR (PyTorch) или ONNX Runtime/OpenVINO."""
from __future__ import annotations

import numpy as np
import supervision as sv

from .config import DetectorCfg


class RfDetrDetector:
    def __init__(self, cfg: DetectorCfg, conf: float):
        from rfdetr import RFDETRMedium, RFDETRSmall
        cls = {"small": RFDETRSmall, "medium": RFDETRMedium}[cfg.size]
        self.model = cls(pretrain_weights=cfg.weights) if cfg.weights else cls()
        self.conf = conf

    def detect(self, bgr: np.ndarray) -> sv.Detections:
        from PIL import Image
        img = Image.fromarray(bgr[:, :, ::-1])
        return self.model.predict(img, threshold=self.conf)


class OnnxDetector:
    """RF-DETR, экспортированный в ONNX (training/export_onnx.py).
    Провайдеры: CUDAExecutionProvider (GPU) или OpenVINOExecutionProvider (Intel CPU/iGPU,
    пакет onnxruntime-openvino). Выходы: dets (1,Q,4) cxcywh в долях кадра, labels (1,Q,C) логиты.
    Сверьте результат с RfDetrDetector на одном кадре после экспорта."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, cfg: DetectorCfg, conf: float):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(cfg.onnx_path, providers=cfg.providers)
        inp = self.sess.get_inputs()[0]
        self.name = inp.name
        self.h, self.w = int(inp.shape[2]), int(inp.shape[3])
        self.conf = conf

    def detect(self, bgr: np.ndarray) -> sv.Detections:
        import cv2
        H, W = bgr.shape[:2]
        x = cv2.resize(bgr, (self.w, self.h))[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        boxes, logits = self.sess.run(None, {self.name: x})[:2]
        probs = 1.0 / (1.0 + np.exp(-logits[0]))
        cls = probs.argmax(1)
        score = probs.max(1)
        keep = score >= self.conf
        cx, cy, bw, bh = boxes[0][keep].T
        xyxy = np.stack([(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H], 1)
        return sv.Detections(xyxy=xyxy.astype(np.float32), confidence=score[keep], class_id=cls[keep])


def build_detector(cfg: DetectorCfg, conf: float):
    return OnnxDetector(cfg, conf) if cfg.backend == "onnx" else RfDetrDetector(cfg, conf)
