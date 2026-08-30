"""Grad-CAM heatmaps: visualize which image regions drove the smoking prediction.

Useful for auditing the model (e.g. confirming it keys on a cigarette/hand-to-mouth
gesture rather than a spurious correlation like a particular background).
"""
import cv2
import numpy as np
import tensorflow as tf


def _find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def _find_backbone(model: tf.keras.Model) -> tf.keras.Model | None:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    return None


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str | None = None) -> np.ndarray:
    """img_array: shape (1, H, W, 3), unscaled RGB pixel values (0-255).

    `model` is expected to be built as Input -> ... -> backbone submodel -> ... -> sigmoid
    (see src/model.py). Grad-CAM needs gradients of the output w.r.t. the backbone's last
    conv activations; because the backbone is a *nested* functional model, we replay the
    pre-backbone and post-backbone layers manually inside the tape rather than relying on
    Keras to trace a single combined graph (which nested functional models don't support
    reliably across Keras versions).
    """
    backbone = _find_backbone(model)
    conv_source = backbone if backbone is not None else model

    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv_layer(conv_source)

    conv_layer = conv_source.get_layer(last_conv_layer_name)
    backbone_grad_model = tf.keras.Model(conv_source.input, [conv_layer.output, conv_source.output])

    pre_layers, post_layers = [], []
    seen_backbone = False
    for layer in model.layers:
        if layer is conv_source:
            seen_backbone = True
            continue
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        (post_layers if seen_backbone else pre_layers).append(layer)

    x = tf.convert_to_tensor(img_array)
    for layer in pre_layers:
        x = layer(x)

    with tf.GradientTape() as tape:
        conv_output, backbone_output = backbone_grad_model(x)
        tape.watch(conv_output)
        y = backbone_output
        for layer in post_layers:
            y = layer(y)
        loss = y[:, 0]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(frame_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap_resized = cv2.resize(heatmap, (frame_bgr.shape[1], frame_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(colored, alpha, frame_bgr, 1 - alpha, 0)
