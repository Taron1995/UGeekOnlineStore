"""Работа станка по движению в области machine_roi (OpenCV), без учёта людей в кадре."""
from __future__ import annotations

import cv2
import numpy as np


class MotionMeter:
    def __init__(self, polygon: list[list[int]], frame_shape: tuple[int, int]):
        self.mask = np.zeros(frame_shape[:2], np.uint8)
        cv2.fillPoly(self.mask, [np.array(polygon, np.int32)], 255)
        self.area = max(int(cv2.countNonZero(self.mask)), 1)
        self.prev: np.ndarray | None = None

    def measure(self, frame_bgr: np.ndarray, person_boxes: np.ndarray) -> float | None:
        """Доля изменившихся пикселей ROI. Рамки людей исключаются, чтобы не принять человека за станок."""
        gray = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        prev, self.prev = self.prev, gray
        if prev is None:
            return None
        diff = cv2.threshold(cv2.absdiff(gray, prev), 25, 255, cv2.THRESH_BINARY)[1]
        roi = self.mask.copy()
        for x1, y1, x2, y2 in person_boxes.astype(int):
            cv2.rectangle(roi, (x1, y1), (x2, y2), 0, -1)
        valid = max(cv2.countNonZero(roi), 1)
        return cv2.countNonZero(cv2.bitwise_and(diff, roi)) / valid
