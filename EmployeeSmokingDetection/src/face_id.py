"""Employee identification from face crops using OpenCV's LBPH face recognizer.

This intentionally avoids heavy external dependencies (dlib / face_recognition):
opencv-contrib-python ships everything needed (Haar cascade detector + LBPHFaceRecognizer).
"""
import json
import os

import cv2
import numpy as np

from config import FACE_LABELS_PATH, FACE_RECOGNIZER_PATH

_FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def get_face_detector() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(_FACE_CASCADE_PATH)


def detect_faces(gray_frame: np.ndarray, detector: cv2.CascadeClassifier, scale_factor=1.1, min_neighbors=5):
    return detector.detectMultiScale(gray_frame, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(60, 60))


class EmployeeFaceRecognizer:
    """Thin wrapper around cv2.face.LBPHFaceRecognizer with a label<->name mapping."""

    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels: dict[str, str] = {}  # str(label_id) -> employee name
        self._trained = False

    def train(self, faces: list[np.ndarray], labels: list[int], label_names: dict[int, str]):
        self.recognizer.train(faces, np.array(labels))
        self.labels = {str(k): v for k, v in label_names.items()}
        self._trained = True

    def predict(self, face_gray: np.ndarray) -> tuple[int, float]:
        """Returns (label_id, distance). Lower distance == more confident match."""
        if not self._trained:
            raise RuntimeError("Recognizer has not been trained/loaded yet.")
        return self.recognizer.predict(face_gray)

    def name_for_label(self, label_id: int) -> str:
        return self.labels.get(str(label_id), "Unknown")

    def save(self, model_path: str = FACE_RECOGNIZER_PATH, labels_path: str = FACE_LABELS_PATH):
        self.recognizer.save(model_path)
        with open(labels_path, "w") as f:
            json.dump(self.labels, f, indent=2)

    def load(self, model_path: str = FACE_RECOGNIZER_PATH, labels_path: str = FACE_LABELS_PATH) -> bool:
        if not (os.path.exists(model_path) and os.path.exists(labels_path)):
            return False
        self.recognizer.read(model_path)
        with open(labels_path) as f:
            self.labels = json.load(f)
        self._trained = True
        return True
