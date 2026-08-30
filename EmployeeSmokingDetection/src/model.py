"""Model architecture: MobileNetV2 transfer learning for binary smoking classification."""
import tensorflow as tf
from tensorflow.keras import layers, models

from config import IMG_SIZE


def build_model(img_size=IMG_SIZE, base_trainable: bool = False, fine_tune_from: int | None = None) -> tf.keras.Model:
    """Builds a MobileNetV2-based binary classifier.

    Args:
        base_trainable: whether the backbone is trainable at all.
        fine_tune_from: if set (and base_trainable=True), freezes all backbone
            layers before this index and fine-tunes the rest. Use this for the
            second-stage fine-tuning pass after the classification head has converged.
    """
    inputs = layers.Input(shape=(*img_size, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)  # MobileNetV2 expects [-1, 1]

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*img_size, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = base_trainable
    if base_trainable and fine_tune_from is not None:
        for layer in base_model.layers[:fine_tune_from]:
            layer.trainable = False

    x = base_model(x, training=False if not base_trainable else None)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="smoking_probability")(x)

    model = models.Model(inputs, outputs, name="smoking_classifier")
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float = 1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model
