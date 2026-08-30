"""Central configuration for the Employee Smoking Detection system."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
SPLIT_DATA_DIR = os.path.join(DATA_DIR, "split")

MODELS_DIR = os.path.join(BASE_DIR, "models")
CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, "smoking_classifier.keras")
FACE_RECOGNIZER_PATH = os.path.join(MODELS_DIR, "face_recognizer.yml")
FACE_LABELS_PATH = os.path.join(MODELS_DIR, "face_labels.json")

LOGS_DIR = os.path.join(BASE_DIR, "logs")
SNAPSHOTS_DIR = os.path.join(LOGS_DIR, "snapshots")
DB_PATH = os.path.join(LOGS_DIR, "detections.db")
TRAINING_HISTORY_DIR = os.path.join(LOGS_DIR, "training")

CLASS_NAMES = ["not_smoking", "smoking"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Detection / inference settings
CAMERA_INDEX = 0
FACE_DETECTOR_SCALE_FACTOR = 1.1
FACE_DETECTOR_MIN_NEIGHBORS = 5
FACE_RECOGNITION_CONFIDENCE_THRESHOLD = 70.0  # lower LBPH distance = better match
SMOKING_CONFIDENCE_THRESHOLD = 0.65
CONSECUTIVE_FRAMES_TO_CONFIRM = 5  # debounce: require N consecutive smoking frames before logging an event
MIN_SECONDS_BETWEEN_LOGS_PER_EMPLOYEE = 60

for _dir in (MODELS_DIR, LOGS_DIR, SNAPSHOTS_DIR, TRAINING_HISTORY_DIR, RAW_DATA_DIR, SPLIT_DATA_DIR):
    os.makedirs(_dir, exist_ok=True)
