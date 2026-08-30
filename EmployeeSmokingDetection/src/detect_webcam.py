"""Real-time camera pipeline: detect a face -> identify the employee -> classify
smoking vs. not-smoking on the region around the face -> debounce and log confirmed events.

Run:
    python -m src.detect_webcam
    python -m src.detect_webcam --show-gradcam   # overlay Grad-CAM heatmap
"""
import argparse
import os
import time

import cv2
import numpy as np
import tensorflow as tf

from config import (
    CAMERA_INDEX,
    CLASS_NAMES,
    CLASSIFIER_MODEL_PATH,
    CONSECUTIVE_FRAMES_TO_CONFIRM,
    FACE_DETECTOR_MIN_NEIGHBORS,
    FACE_DETECTOR_SCALE_FACTOR,
    FACE_RECOGNITION_CONFIDENCE_THRESHOLD,
    IMG_SIZE,
    MIN_SECONDS_BETWEEN_LOGS_PER_EMPLOYEE,
    SMOKING_CONFIDENCE_THRESHOLD,
    SNAPSHOTS_DIR,
)
from src import db
from src.face_id import EmployeeFaceRecognizer, detect_faces, get_face_detector
from src.gradcam import make_gradcam_heatmap, overlay_heatmap


def preprocess_for_classifier(bgr_crop: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    return np.expand_dims(resized.astype(np.float32), axis=0)


def expand_face_to_body_region(x, y, w, h, frame_shape, upward=0.6, downward=2.2, side=0.8):
    """The mouth/hand/cigarette are below the face, so widen the crop downward from the face box."""
    fh, fw = frame_shape[:2]
    x1 = max(0, int(x - side * w))
    y1 = max(0, int(y - upward * h))
    x2 = min(fw, int(x + w + side * w))
    y2 = min(fh, int(y + h + downward * h))
    return x1, y1, x2, y2


class DetectionState:
    """Debounces per-employee predictions so a single noisy frame can't log a false event."""

    def __init__(self):
        self.consecutive_smoking = {}

    def update(self, employee_key: str, is_smoking: bool) -> bool:
        """Returns True the first time `is_smoking` has been observed for enough
        consecutive frames in a row (i.e. right when the event should be logged)."""
        count = self.consecutive_smoking.get(employee_key, 0)
        if is_smoking:
            count += 1
        else:
            count = 0
        self.consecutive_smoking[employee_key] = count
        return count == CONSECUTIVE_FRAMES_TO_CONFIRM


def run(camera_index: int = CAMERA_INDEX, show_gradcam: bool = False, headless: bool = False):
    db.init_db()

    if not os.path.exists(CLASSIFIER_MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {CLASSIFIER_MODEL_PATH}. Run `python -m src.train` first."
        )
    classifier = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)

    face_recognizer = EmployeeFaceRecognizer()
    have_face_id = face_recognizer.load()
    if not have_face_id:
        print("No face-recognition model found; detections will be logged as 'Unknown'. "
              "Run `python scripts/register_employee.py` to enroll employees.")

    face_detector = get_face_detector()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    state = DetectionState()
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    print("Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed; stopping.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray, face_detector, FACE_DETECTOR_SCALE_FACTOR, FACE_DETECTOR_MIN_NEIGHBORS)

            for (x, y, w, h) in faces:
                employee_name = "Unknown"
                employee_db_id = None
                if have_face_id:
                    label_id, distance = face_recognizer.predict(cv2.resize(gray[y:y + h, x:x + w], (200, 200)))
                    if distance <= FACE_RECOGNITION_CONFIDENCE_THRESHOLD:
                        employee_name = face_recognizer.name_for_label(label_id)
                        row = db.get_employee_by_label(label_id)
                        employee_db_id = row["id"] if row else None

                x1, y1, x2, y2 = expand_face_to_body_region(x, y, w, h, frame.shape)
                region = frame[y1:y2, x1:x2]
                if region.size == 0:
                    continue

                model_input = preprocess_for_classifier(region)
                smoking_prob = float(classifier.predict(model_input, verbose=0)[0][0])
                is_smoking = smoking_prob >= SMOKING_CONFIDENCE_THRESHOLD

                should_log = state.update(employee_name, is_smoking)
                if should_log:
                    last_ts = db.last_detection_time(employee_name)
                    if time.time() - last_ts >= MIN_SECONDS_BETWEEN_LOGS_PER_EMPLOYEE:
                        snapshot_path = os.path.join(
                            SNAPSHOTS_DIR, f"{employee_name}_{int(time.time())}.jpg"
                        )
                        cv2.imwrite(snapshot_path, frame)
                        db.log_detection(employee_db_id, employee_name, smoking_prob, snapshot_path)
                        print(f"[LOGGED] {employee_name} smoking (confidence={smoking_prob:.2f})")

                color = (0, 0, 255) if is_smoking else (0, 200, 0)
                label = f"{employee_name}: {CLASS_NAMES[int(is_smoking)]} ({smoking_prob:.2f})"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if show_gradcam and is_smoking:
                    heatmap = make_gradcam_heatmap(model_input, classifier)
                    overlay = overlay_heatmap(region.copy(), heatmap)
                    frame[y1:y2, x1:x2] = cv2.addWeighted(overlay, 0.5, region, 0.5, 0)

            if not headless:
                cv2.imshow("Employee Smoking Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX)
    parser.add_argument("--show-gradcam", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Don't open a display window (e.g. on a headless server).")
    args = parser.parse_args()
    run(args.camera_index, args.show_gradcam, args.headless)
