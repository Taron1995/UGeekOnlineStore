"""Evaluate the trained classifier: confusion matrix, classification report, ROC curve."""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from config import CLASS_NAMES, CLASSIFIER_MODEL_PATH, TRAINING_HISTORY_DIR
from src.dataset import load_datasets


def evaluate(model_path: str = CLASSIFIER_MODEL_PATH, out_dir: str = TRAINING_HISTORY_DIR):
    model = tf.keras.models.load_model(model_path)
    _, _, test_ds, class_names = load_datasets()

    y_true, y_prob = [], []
    for images, labels in test_ds:
        y_true.extend(labels.numpy().flatten().tolist())
        y_prob.extend(model.predict(images, verbose=0).flatten().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred, target_names=class_names))

    auc = roc_auc_score(y_true, y_prob)
    print(f"ROC AUC: {auc:.4f}")

    os.makedirs(out_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()

    print(f"Saved confusion matrix to {cm_path}")
    print(f"Saved ROC curve to {roc_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=CLASSIFIER_MODEL_PATH)
    args = parser.parse_args()
    evaluate(args.model_path)
