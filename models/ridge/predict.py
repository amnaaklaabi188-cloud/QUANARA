"""Ridge Regression — predict. Run with --help for options; see README.md in this folder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from models.workflow import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("ridge", "predict"))
