"""Double-click (or Run in PyCharm): opens the window where you choose the videos and press Start."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from videoanalytics.gui import main  # noqa: E402

if __name__ == "__main__":  # the guard is required: worker processes import this file on Windows
    main(sys.argv[1:])
