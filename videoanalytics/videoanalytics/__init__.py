"""Offline video analytics MVP: people counting and machine work/idle from camera videos."""

import os

# FFmpeg (inside OpenCV) prints a warning for every frame it cannot decode yet, e.g. after a jump into the middle of a long-GOP stream: harmless,
# but hundreds of lines. Must be set before OpenCV opens its first video; a value set by the user is kept.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
