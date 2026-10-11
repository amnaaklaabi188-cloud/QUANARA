"""Shared helpers for the evaluation scripts (path setup, output folders)."""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
warnings.filterwarnings("ignore")
RESULTS = ROOT / "results" / "model_benchmarks"
FIGURES = ROOT / "figures" / "model_evaluation"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
