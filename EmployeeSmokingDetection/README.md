# Employee Smoking Detection (ML/CV)

Camera-based system that flags employees smoking on-site, using a transfer-learning
deep learning classifier for the smoking/not-smoking decision and classic computer
vision (face detection + LBPH face recognition) to attribute a detection to a specific
enrolled employee. Ships with a training pipeline, an explainability tool (Grad-CAM), a
real-time webcam pipeline, and a Flask dashboard for reviewing logged events.

**Read "Legal and ethical considerations" below before deploying this anywhere.**

## How it works

```
 webcam frame
      |
      v
 Haar-cascade face detector  --> face box
      |                              |
      v                              v
 LBPH face recognizer          crop widened region below the face
 (who is this employee?)       (mouth / hand / cigarette area)
      |                              |
      |                              v
      |                    MobileNetV2 transfer-learning
      |                    classifier -> P(smoking)
      |                              |
      +--------------+---------------+
                     v
      debounce (N consecutive frames)
      + per-employee cooldown
                     v
        SQLite log + snapshot image
                     v
            Flask review dashboard
```

- **Classifier** (`src/model.py`, `src/train.py`): MobileNetV2 pretrained on ImageNet,
  fine-tuned in two stages (frozen backbone warmup, then partial unfreeze at a low
  learning rate) for binary `smoking` vs. `not_smoking` classification. Uses data
  augmentation, class weighting, early stopping, LR scheduling, and TensorBoard logging.
- **Explainability** (`src/gradcam.py`): Grad-CAM heatmaps show which pixels drove a
  "smoking" prediction, so you can sanity-check the model isn't keying on a spurious
  correlation (e.g. a specific background) instead of the cigarette/gesture itself.
- **Employee identification** (`src/face_id.py`, `scripts/register_employee.py`):
  OpenCV's LBPH face recognizer, trained from a handful of enrollment photos per
  employee. No cloud face-ID API and no biometric templates leave the machine.
- **Event logging** (`src/db.py`): SQLite database of confirmed detections
  (debounced: requires several consecutive positive frames, rate-limited per employee)
  with a saved snapshot image for human review.
- **Dashboard** (`dashboard/app.py`): browse detection history, per-employee totals,
  and enrolled employees.

## Project layout

```
EmployeeSmokingDetection/
  config.py                 # paths, thresholds, camera settings
  data/raw/{smoking,not_smoking}/   # put your training images here
  src/
    dataset.py               # train/val/test split + tf.data augmentation pipeline
    model.py                 # MobileNetV2 transfer-learning architecture
    train.py                 # two-stage training script
    evaluate.py              # confusion matrix / ROC / classification report
    gradcam.py                # Grad-CAM explainability
    face_id.py                # LBPH employee face recognizer
    detect_webcam.py          # real-time detection + logging loop
    db.py                     # SQLite persistence
  scripts/
    register_employee.py      # enroll an employee's face from the webcam
    prepare_dataset.py         # standalone train/val/test split
  dashboard/                  # Flask review UI
  models/                      # trained model artifacts (git-ignored)
  logs/                        # detections.db + snapshots (git-ignored)
  tests/                       # unit tests (model, Grad-CAM, DB)
```

## Setup

```bash
cd EmployeeSmokingDetection
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 1. Collect training data

You must supply your own labeled images — none are bundled with this project (see
ethics note below on why: real employee photos are sensitive personal data).

Put images into:

```
data/raw/smoking/*.jpg
data/raw/not_smoking/*.jpg
```

Aim for at least a few hundred images per class, varied lighting/angles/people, for a
model that generalizes. Public "smoker detection" datasets on Kaggle/Roboflow are a
reasonable starting point for a proof of concept; add site-specific footage before
relying on this in production.

## 2. Train the classifier

```bash
python -m src.train --epochs 15 --fine-tune-epochs 10
```

This splits `data/raw` into `data/split/{train,val,test}`, trains the two-stage
MobileNetV2 model, and saves the best checkpoint to `models/smoking_classifier.keras`.
TensorBoard logs land in `logs/training/<run>/` (`tensorboard --logdir logs/training`).

## 3. Evaluate

```bash
python -m src.evaluate
```

Prints a classification report + ROC AUC and saves `confusion_matrix.png` /
`roc_curve.png` to `logs/training/`.

## 4. Enroll employees (optional, for attributing detections to a name)

```bash
python scripts/register_employee.py --name "Jane Doe" --code E1234 --samples 30
```

Captures ~30 face crops from the webcam and (re)trains the LBPH recognizer over every
enrolled employee. Skip this step if you only need anonymous "smoking detected" alerts.

## 5. Run live detection

```bash
python -m src.detect_webcam            # opens a display window with live overlay
python -m src.detect_webcam --show-gradcam   # + Grad-CAM heatmap on positive detections
python -m src.detect_webcam --headless       # no display window (e.g. a headless box)
```

Confirmed events (several consecutive positive frames, rate-limited per employee — see
`CONSECUTIVE_FRAMES_TO_CONFIRM` and `MIN_SECONDS_BETWEEN_LOGS_PER_EMPLOYEE` in
`config.py`) are written to `logs/detections.db` with a snapshot image.

## 6. Review detections

```bash
cd dashboard && python app.py
```

Open `http://localhost:5000`.

## Tuning

All thresholds live in `config.py`:

| Setting | Purpose |
|---|---|
| `SMOKING_CONFIDENCE_THRESHOLD` | Minimum model probability to count a frame as "smoking". |
| `CONSECUTIVE_FRAMES_TO_CONFIRM` | Debounce: consecutive positive frames required before logging. |
| `MIN_SECONDS_BETWEEN_LOGS_PER_EMPLOYEE` | Cooldown to avoid re-logging one continuous smoking session repeatedly. |
| `FACE_RECOGNITION_CONFIDENCE_THRESHOLD` | LBPH distance cutoff; lower = stricter identity match. |

## Testing

```bash
pip install pytest
pytest tests/
```

## Legal and ethical considerations

This system monitors employees with a camera and logs a health/behavior-adjacent
inference tied to their identity. Before deploying it:

- **Check local law first.** Many jurisdictions (e.g. GDPR in the EU/UK, and various
  US/state workplace-surveillance and biometric-privacy statutes) restrict or require
  specific safeguards for camera monitoring, biometric identification (face
  recognition counts), and processing health-related inferences about employees.
  Get sign-off from legal/HR before deploying — this is not legal advice.
- **Tell employees.** Disclose that cameras run this analysis, what is logged, how
  long it's retained, and who can see it. Covert workplace surveillance is illegal
  in many places.
- **Minimize retention.** `MIN_SECONDS_BETWEEN_LOGS_PER_EMPLOYEE` limits duplicate
  logs, but you should also add a scheduled job to purge `logs/detections.db` and
  `logs/snapshots/` past your retention policy.
- **Expect errors.** Both the smoking classifier and the face recognizer will
  misfire sometimes (false positives/negatives). Treat detections as a prompt for a
  human to review the snapshot, not as automatic grounds for discipline.
- **Secure the data.** Face embeddings, snapshots, and detection logs are sensitive.
  Restrict filesystem/dashboard access and consider encrypting `models/` and `logs/`
  at rest if deployed beyond a local proof of concept.
