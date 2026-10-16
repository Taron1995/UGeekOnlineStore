"""Where models live. Not the current directory: a drag-and-drop launch has an unpredictable one."""
from __future__ import annotations

import os
from pathlib import Path


def models_dir() -> Path:
    """$VIDAN_MODELS, else <project>/models for a source checkout / editable install, else ~/.videoanalytics/models."""
    env = os.environ.get("VIDAN_MODELS")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parent.parent
    if (root / "pyproject.toml").exists() and os.access(root, os.W_OK):
        return root / "models"
    return Path.home() / ".videoanalytics" / "models"


def find_model(name: str):
    """The model file if it exists in the models dir (or, for older layouts, ./models), else None."""
    for d in (models_dir(), Path("models")):
        p = d / name
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    return None
