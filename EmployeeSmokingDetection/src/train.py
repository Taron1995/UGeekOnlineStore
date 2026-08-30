"""Two-stage transfer-learning training: frozen-backbone warmup, then fine-tuning.

Usage:
    python -m src.train --epochs 15 --fine-tune-epochs 10
"""
import argparse
import datetime
import os

import tensorflow as tf

from config import CLASSIFIER_MODEL_PATH, TRAINING_HISTORY_DIR
from src.dataset import compute_class_weights, load_datasets, split_dataset
from src.model import build_model, compile_model


def _callbacks(run_name: str):
    log_dir = os.path.join(TRAINING_HISTORY_DIR, run_name)
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(CLASSIFIER_MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.CSVLogger(os.path.join(log_dir, "history.csv")),
    ]


def main():
    parser = argparse.ArgumentParser(description="Train the employee smoking-detection classifier.")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for the frozen-backbone warmup stage.")
    parser.add_argument("--fine-tune-epochs", type=int, default=10, help="Epochs for the fine-tuning stage.")
    parser.add_argument("--fine-tune-from-layer", type=int, default=100,
                         help="MobileNetV2 layer index to unfreeze from during fine-tuning.")
    parser.add_argument("--skip-split", action="store_true", help="Reuse an existing data/split directory.")
    args = parser.parse_args()

    if not args.skip_split:
        split_dataset()

    train_ds, val_ds, test_ds, class_names = load_datasets()
    class_weights = compute_class_weights()
    print(f"Class weights (balances class imbalance): {class_weights}")

    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Stage 1: train the classification head with a frozen MobileNetV2 backbone.
    model, base_model = build_model(base_trainable=False)
    compile_model(model, learning_rate=1e-3)
    model.summary()

    print("\n=== Stage 1: training classification head (frozen backbone) ===")
    model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        class_weight=class_weights, callbacks=_callbacks(f"{run_name}-stage1"),
    )

    # Stage 2: unfreeze the top of the backbone and fine-tune at a low learning rate.
    print("\n=== Stage 2: fine-tuning top backbone layers ===")
    base_model.trainable = True
    for layer in base_model.layers[:args.fine_tune_from_layer]:
        layer.trainable = False
    compile_model(model, learning_rate=1e-5)

    model.fit(
        train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
        class_weight=class_weights, callbacks=_callbacks(f"{run_name}-stage2"),
    )

    model.save(CLASSIFIER_MODEL_PATH)
    print(f"\nSaved trained model to {CLASSIFIER_MODEL_PATH}")

    print("\n=== Test set evaluation ===")
    results = model.evaluate(test_ds, return_dict=True)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
