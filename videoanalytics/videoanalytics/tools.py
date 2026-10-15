"""Helpers for setting up the config: snapshots with a coordinate grid and the configured zones drawn on top."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def snapshot(video: str, out_path: Path, cam_cfg: dict, at_s: float = 30.0):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000)
    ok, f = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_MSEC, 0)
        ok, f = cap.read()
    cap.release()
    if not ok:
        raise OSError(f"cannot read {video}")
    h, w = f.shape[:2]
    for i in range(1, 10):  # 10% grid with normalized coordinates
        x, y = int(w * i / 10), int(h * i / 10)
        cv2.line(f, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.line(f, (0, y), (w, y), (255, 255, 255), 1)
        cv2.putText(f, f"{i / 10:.1f}", (x + 3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(f, f"{i / 10:.1f}", (3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    def poly(pts, color, label):
        p = (np.array(pts) * [w, h]).astype(np.int32)
        cv2.polylines(f, [p], True, color, 2)
        cv2.putText(f, label, tuple(p[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for z in cam_cfg.get("zones") or []:
        poly(z["polygon"], (0, 200, 0), z["name"])
    if cam_cfg.get("machine"):
        poly(cam_cfg["machine"]["polygon"], (0, 140, 255), "machine")
    for ln in cam_cfg.get("lines") or []:
        a, b = (int(ln["p1"][0] * w), int(ln["p1"][1] * h)), (int(ln["p2"][0] * w), int(ln["p2"][1] * h))
        cv2.arrowedLine(f, a, b, (255, 0, 255), 2, tipLength=0.05)
        cv2.putText(f, ln["name"], a, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), f)
