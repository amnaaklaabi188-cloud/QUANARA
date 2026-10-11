"""Central configuration: paths, dataset identifiers and the modelling calendar.

All dates below were verified in the data audit (reports/audit_report.md), not assumed:
  * hotel files are daily; train 2022-01-01..2025-07-31, test 2025-08-01..2026-02-28
  * flight_data.xlsx is MONTHLY for 2022 (one row per route on the 1st of each month)
    and DAILY from 2023-01-01 to 2026-02-28
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw inputs. By default the app reads its own read-only copies in data/raw; set
# QUANARA_DATA_DIR to read the original folder instead. Originals are never written.
RAW_DIR = Path(os.environ.get("QUANARA_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
SCENARIOS_DIR = ARTIFACTS_DIR / "scenarios"
REPORTS_DIR = PROJECT_ROOT / "reports"
ACTIVE_MODEL_FILE = MODELS_DIR / "active_model.json"

# Logical dataset id -> filename regex (case-insensitive). Tolerates spaces, underscores
# and numbered suffixes such as "flight_data (1).xlsx".
DATASET_PATTERNS = {
    "intl_train": r"international[ _-]*train.*\.xlsx$",
    "intl_test": r"international[ _-]*test.*\.xlsx$",
    "dom_train": r"domestic[ _-]*train.*\.xlsx$",
    "dom_test": r"domestic[ _-]*test.*\.xlsx$",
    "flight": r"flight[ _-]*data.*\.xlsx$",
}
DICTIONARY_PATTERN = r"data[ _-]*dictionary.*\.pdf$"
SHEET_NAME = "Export"

DATASET_LABELS = {
    "intl_train": "International hotel guests — train",
    "intl_test": "International hotel guests — test",
    "dom_train": "Domestic hotel guests — train",
    "dom_test": "Domestic hotel guests — test",
    "flight": "Flight operations (arrivals into AUH)",
}

# ---- Modelling calendar (verified) -------------------------------------------------
HOTEL_TRAIN_START = pd.Timestamp("2022-01-01")
HOTEL_TRAIN_END = pd.Timestamp("2025-07-31")
HOTEL_TEST_START = pd.Timestamp("2025-08-01")
HOTEL_TEST_END = pd.Timestamp("2026-02-28")
FLIGHT_DAILY_START = pd.Timestamp("2023-01-01")
FLIGHT_END = pd.Timestamp("2026-02-28")

# Trailing aviation features use up to 28 days plus a 28-day growth lag (56 days), so the
# first date with every feature defined is 2023-02-26. Every model uses the same fixed
# start so results and ablations are comparable.
FEATURE_WARMUP_DAYS = 56
MODEL_START = pd.Timestamp("2023-03-01")
MODEL_END = HOTEL_TRAIN_END
DEFAULT_HOLDOUT_START = pd.Timestamp("2025-02-01")  # last 181 observed days

# Same weekday one year earlier (52 weeks).
SEASONAL_LAG_DAYS = 364

RANDOM_STATE = 42

SCOPES = {
    "international_total": "International — all 45 reported nationalities (daily total)",
    "international_market": "International — by source market (daily × nationality)",
    "domestic": "Domestic (daily total)",
    "overall": "Overall — international + domestic (daily total)",
}
MODES = {
    "whatif": "What-if Forecasting (scenario-time features only)",
    "competition": "Competition Test Prediction (may use observed test-time hotel fields)",
}


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, CACHE_DIR, MODELS_DIR, SCENARIOS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
