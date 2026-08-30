"""Enroll an employee's face so the live pipeline can attribute detections to a name.

Usage:
    python scripts/register_employee.py --name "Jane Doe" --code E1234 --samples 30

Captures face crops from the webcam, then retrains the LBPH recognizer over
every enrolled employee's saved samples (so re-running this script for a new
hire doesn't erase previously enrolled employees).
"""
import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CAMERA_INDEX, FACE_DETECTOR_MIN_NEIGHBORS, FACE_DETECTOR_SCALE_FACTOR, MODELS_DIR
from src import db
from src.face_id import EmployeeFaceRecognizer, detect_faces, get_face_detector

ENROLLMENT_DIR = os.path.join(MODELS_DIR, "enrollment_samples")


def capture_samples(name: str, num_samples: int, camera_index: int) -> str:
    person_dir = os.path.join(ENROLLMENT_DIR, name.replace(" ", "_"))
    os.makedirs(person_dir, exist_ok=True)

    detector = get_face_detector()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    print(f"Capturing {num_samples} face samples for '{name}'. Look at the camera. Press 'q' to abort.")
    captured = 0
    try:
        while captured < num_samples:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray, detector, FACE_DETECTOR_SCALE_FACTOR, FACE_DETECTOR_MIN_NEIGHBORS)
            for (x, y, w, h) in faces:
                face_crop = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                cv2.imwrite(os.path.join(person_dir, f"{captured:03d}.png"), face_crop)
                captured += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                break  # one face per frame
            cv2.putText(frame, f"Captured {captured}/{num_samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            cv2.imshow("Enrollment", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Captured {captured} samples to {person_dir}")
    return person_dir


def retrain_recognizer():
    """Rebuilds the LBPH recognizer from every enrolled employee's sample images.

    Label IDs must stay stable across retrains: `db.upsert_employee` matches rows by
    `label_id`, so re-deriving IDs from an alphabetically sorted folder listing would
    silently reassign an existing employee's ID to someone else as soon as a new
    employee sorts ahead of them. Reuse each existing employee's stored label_id and
    only mint new ones for folders that aren't in the DB yet.
    """
    if not os.path.isdir(ENROLLMENT_DIR):
        raise RuntimeError("No enrollment samples found yet.")

    db.init_db()
    existing_labels = {row["name"].replace(" ", "_"): row["label_id"] for row in db.list_employees()}
    next_label_id = max(existing_labels.values(), default=-1) + 1

    faces, labels, label_names = [], [], {}
    for person_name in sorted(os.listdir(ENROLLMENT_DIR)):
        person_dir = os.path.join(ENROLLMENT_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        if person_name in existing_labels:
            label_id = existing_labels[person_name]
        else:
            label_id = next_label_id
            next_label_id += 1
        label_names[label_id] = person_name.replace("_", " ")
        for fname in os.listdir(person_dir):
            img = cv2.imread(os.path.join(person_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(label_id)

    if not faces:
        raise RuntimeError("No face samples available to train on.")

    recognizer = EmployeeFaceRecognizer()
    recognizer.train(faces, labels, label_names)
    recognizer.save()

    for label_id, name in label_names.items():
        db.upsert_employee(label_id, name)

    print(f"Retrained face recognizer over {len(label_names)} employees, {len(faces)} samples.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Employee full name.")
    parser.add_argument("--code", default=None, help="Optional employee ID/code.")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX)
    args = parser.parse_args()

    db.init_db()
    capture_samples(args.name, args.samples, args.camera_index)
    retrain_recognizer()

    if args.code:
        row = db.list_employees()
        for r in row:
            if r["name"] == args.name:
                db.upsert_employee(r["label_id"], args.name, args.code)
                break

    print("Done. Restart src/detect_webcam.py to pick up the updated face model.")


if __name__ == "__main__":
    main()
