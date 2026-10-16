"""PyCharm launcher: edit the constants below, pick STEP, press Run. Everything also works from a terminal:
python -m videoanalytics <command> --help"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # works even when the project root is the parent folder

from videoanalytics.cli import main  # noqa: E402

# ---- edit these ------------------------------------------------------------------------------------------------------
VIDEOS = r"D:\footage"  # a video file, or the folder with the recordings (searched recursively)
CONFIG = "cameras.yaml"  # only for the manual route: a copy of cameras.example.yaml with your cameras, zones and lamps
MODEL = "models/yolox_tiny.onnx"  # person detector
OUT = "out"
STEP = "analyze"  # analyze = everything in one step (just set VIDEOS); doctor = check the files first.  Manual route: models -> scan -> snap -> trial -> run
# ---------------------------------------------------------------------------------------------------------------------

COMMON = ["--config", CONFIG, "--videos", VIDEOS]
STEPS = {
    "doctor": ["doctor", VIDEOS],  # 10-60 s check of the recordings; prints a text (and saves doctor.txt) to read or to send if something is wrong
    "analyze": ["analyze", VIDEOS],  # learns the scene by itself, analyses, opens the report (results: <folder>/_analysis)
    "gui": ["gui", VIDEOS],  # the same in a window
    "models": ["models", "--pose"],  # downloads yolox_tiny.onnx (+ the MediaPipe pose model into models/)
    "scan": ["scan", *COMMON, "--model", MODEL],  # files per camera, timestamps, run-time estimate
    "snap": ["snap", *COMMON, "--out", OUT],  # out/snap/<camera>.jpg with grid, zones and lamp colour readout
    "trial": ["run", *COMMON, "--model", MODEL, "--out", OUT, "--limit-files", "4"],
    "run": ["run", *COMMON, "--model", MODEL, "--out", OUT],  # resumable: stop and start again any time
    "report": ["report", "--config", CONFIG, "--out", OUT],
}

if __name__ == "__main__":  # the guard is required: worker processes import this file on Windows
    main(STEPS[STEP])
