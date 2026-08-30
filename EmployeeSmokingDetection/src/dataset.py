"""Data loading, splitting and augmentation pipeline for the smoking classifier."""
import os
import shutil
from pathlib import Path

import tensorflow as tf

from config import BATCH_SIZE, CLASS_NAMES, IMG_SIZE, RAW_DATA_DIR, SPLIT_DATA_DIR


def split_dataset(raw_dir: str = RAW_DATA_DIR, out_dir: str = SPLIT_DATA_DIR,
                   val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
    """Splits data/raw/<class>/*.jpg into data/split/{train,val,test}/<class>/*.jpg."""
    import random
    random.seed(seed)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    for class_name in CLASS_NAMES:
        class_dir = Path(raw_dir) / class_name
        images = [p for p in class_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not images:
            raise ValueError(
                f"No images found in {class_dir}. Add training photos before running the split."
            )
        random.shuffle(images)

        n = len(images)
        n_val = max(1, int(n * val_ratio))
        n_test = max(1, int(n * test_ratio))
        n_train = n - n_val - n_test
        if n_train < 1:
            raise ValueError(f"Not enough images in {class_dir} to split into train/val/test.")

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }
        for split_name, split_images in splits.items():
            dest_dir = Path(out_dir) / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(img_path, dest_dir / img_path.name)

    print(f"Dataset split written to {out_dir}")


def _augmentation_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomBrightness(0.1),
    ], name="augmentation")


def load_datasets(split_dir: str = SPLIT_DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """Returns (train_ds, val_ds, test_ds, class_names) as tf.data.Dataset pipelines."""
    train_dir = os.path.join(split_dir, "train")
    val_dir = os.path.join(split_dir, "val")
    test_dir = os.path.join(split_dir, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size, label_mode="binary", class_names=CLASS_NAMES,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=img_size, batch_size=batch_size, label_mode="binary", class_names=CLASS_NAMES,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=img_size, batch_size=batch_size, label_mode="binary", class_names=CLASS_NAMES,
        shuffle=False,
    )

    augment = _augmentation_layer()
    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=autotune)
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds, CLASS_NAMES


def compute_class_weights(split_dir: str = SPLIT_DATA_DIR):
    """Balances loss contribution when smoking/not_smoking sample counts differ."""
    counts = {}
    for i, class_name in enumerate(CLASS_NAMES):
        class_dir = Path(split_dir) / "train" / class_name
        counts[i] = len(list(class_dir.glob("*")))

    total = sum(counts.values())
    n_classes = len(counts)
    return {i: total / (n_classes * max(c, 1)) for i, c in counts.items()}
