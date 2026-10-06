"""Нарезка клипов FFmpeg без перекодирования (-c copy), поэтому подходит сборка FFmpeg без GPL (LGPL)."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def cut_clip(src: str, offset_s: float, duration_s: float, out: Path, pad_s: float = 3.0) -> bool:
    if not shutil.which("ffmpeg"):
        log.warning("ffmpeg не найден — клип пропущен")
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    start = max(offset_s - pad_s, 0.0)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", src,
           "-t", f"{duration_s + 2 * pad_s:.2f}", "-c", "copy", "-an", str(out)]
    try:
        subprocess.run(cmd, check=True, timeout=120)
        return True
    except (subprocess.SubprocessError, OSError) as e:
        log.warning("ffmpeg: %s", e)
        return False
