"""PyCharm launcher: edit the constants below, pick STEP, press Run. Everything also works from a terminal:
python -m videoanalytics <command> --help"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # works even when the project root is the parent folder

from videoanalytics.cli import main  # noqa: E402

# ---- edit these ------------------------------------------------------------------------------------------------------
VIDEOS = r"D:\footage"  # root folder with the recordings (searched recursively)
CONFIG = "cameras.yaml"  # copy of cameras.example.yaml with your cameras, zones and lamps
MODEL = "models/yolox_tiny.onnx"  # person detector
OUT = "out"
STEP = "scan"  # models -> scan -> snap -> trial -> run   (report rebuilds the HTML/CSV from finished files)
# ---------------------------------------------------------------------------------------------------------------------

COMMON = ["--config", CONFIG, "--videos", VIDEOS]
STEPS = {
    "models": ["models", "--pose"],  # downloads yolox_tiny.onnx (+ the MediaPipe pose model into models/)
    "scan": ["scan", *COMMON, "--model", MODEL],  # files per camera, timestamps, run-time estimate
    "snap": ["snap", *COMMON, "--out", OUT],  # out/snap/<camera>.jpg with grid, zones and lamp colour readout
    "trial": ["run", *COMMON, "--model", MODEL, "--out", OUT, "--limit-files", "4"],
    "run": ["run", *COMMON, "--model", MODEL, "--out", OUT],  # resumable: stop and start again any time
    "report": ["report", "--config", CONFIG, "--out", OUT],
}

if __name__ == "__main__":  # the guard is required: worker processes import this file on Windows
    main(STEPS[STEP])
