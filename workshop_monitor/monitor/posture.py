"""Определение «сидит» и «не двигается» по скелету MediaPipe Pose (работает на CPU)."""
from __future__ import annotations

import numpy as np

# индексы landmarks MediaPipe Pose
L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE = 11, 12, 23, 24, 25, 26
MIN_VIS = 0.5


def is_seated(pts: np.ndarray, vis: np.ndarray, ratio_thr: float = 0.5) -> bool | None:
    """pts: (33,2) нормированные y вниз. Сидя колено примерно на уровне бедра:
    (knee_y - hip_y) / torso < ratio_thr. None — если ключевые точки не видны."""
    need = [L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE]
    if (vis[need] < MIN_VIS).any():
        return None
    sh = pts[[L_SH, R_SH], 1].mean()
    hip = pts[[L_HIP, R_HIP], 1].mean()
    knee = pts[[L_KNEE, R_KNEE], 1].mean()
    torso = hip - sh
    if torso <= 1e-3:
        return None  # человек лежит / плохая поза
    return bool((knee - hip) / torso < ratio_thr)


def pose_motion(prev: np.ndarray | None, cur: np.ndarray,
                prev_vis: np.ndarray | None, cur_vis: np.ndarray) -> float | None:
    """Среднее смещение видимых точек между двумя замерами (в долях размера кропа человека)."""
    if prev is None:
        return None
    m = (prev_vis >= MIN_VIS) & (cur_vis >= MIN_VIS)
    if m.sum() < 8:
        return None
    return float(np.linalg.norm(cur[m] - prev[m], axis=1).mean())


class PoseAnalyzer:
    def __init__(self, complexity: int = 0):
        import mediapipe as mp  # ленивый импорт: тесты не требуют mediapipe
        self._pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=complexity)

    def analyze(self, crop_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        import cv2
        if crop_bgr.size == 0:
            return None
        res = self._pose.process(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks.landmark
        pts = np.array([[p.x, p.y] for p in lm], dtype=np.float32)
        vis = np.array([p.visibility for p in lm], dtype=np.float32)
        return pts, vis
