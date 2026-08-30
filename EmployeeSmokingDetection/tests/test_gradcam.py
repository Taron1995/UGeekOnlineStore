import numpy as np

from config import IMG_SIZE
from src.gradcam import make_gradcam_heatmap, overlay_heatmap
from src.model import build_model, compile_model


def test_gradcam_heatmap_shape_and_overlay():
    model, _ = build_model(base_trainable=False)
    compile_model(model)

    img = np.random.randint(0, 255, size=(1, *IMG_SIZE, 3)).astype("float32")
    heatmap = make_gradcam_heatmap(img, model)

    assert heatmap.ndim == 2
    assert np.all(heatmap >= 0) and np.all(heatmap <= 1)

    frame = np.random.randint(0, 255, size=(*IMG_SIZE, 3)).astype("uint8")
    overlay = overlay_heatmap(frame, heatmap)
    assert overlay.shape == frame.shape
