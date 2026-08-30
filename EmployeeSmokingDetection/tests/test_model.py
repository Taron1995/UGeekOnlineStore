import numpy as np

from config import IMG_SIZE
from src.model import build_model, compile_model


def test_build_model_output_shape():
    model, base_model = build_model(base_trainable=False)
    compile_model(model)

    dummy_batch = np.random.randint(0, 255, size=(2, *IMG_SIZE, 3)).astype("float32")
    predictions = model.predict(dummy_batch, verbose=0)

    assert predictions.shape == (2, 1)
    assert np.all((predictions >= 0) & (predictions <= 1))
    assert base_model.trainable is False


def test_fine_tune_freezes_lower_layers():
    model, base_model = build_model(base_trainable=True, fine_tune_from=100)
    frozen = [l for l in base_model.layers[:100] if not l.trainable]
    assert len(frozen) == 100
