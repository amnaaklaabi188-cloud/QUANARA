"""One naming system for saved models, used by every page (Training Studio, Evaluation, Simulator,
Copilot, Demo) so a model is described identically wherever it appears.

Every value is read from the model's own metadata; nothing is inferred from another model.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

import pandas as pd

ALGORITHMS = {"seasonal_naive": "Seasonal naive baseline", "ridge": "Ridge Regression", "random_forest": "Random Forest",
              "extra_trees": "Extra Trees", "catboost": "CatBoost", "xgboost": "XGBoost", "svr": "Support Vector Regression",
              "lightgbm": "LightGBM"}
TARGETS = {"international_market": "International hotel guests, by nationality", "international_total": "International hotel guests",
           "domestic": "Domestic hotel guests", "overall": "All hotel guests (international + domestic)"}
TARGETS_SHORT = {"international_market": "Intl guests by nationality", "international_total": "Intl guests",
                 "domestic": "Domestic guests", "overall": "All guests"}
SCOPES = {"international_market": "Daily, per nationality", "international_total": "Daily total", "domestic": "Daily total",
          "overall": "Daily total"}
UNITS = {"international_market": "guests per nationality-day", "international_total": "guests per day", "domestic": "guests per day",
         "overall": "guests per day"}
ROWS = {"international_market": "nationality-days", "international_total": "days", "domestic": "days", "overall": "days"}
MODES = {"whatif": "What-if", "competition": "Competition"}
MODE_NOTES = {"whatif": "Aviation, calendar and last year's guests only — usable for scenarios.",
              "competition": "Also uses same-day hotel arrivals — scores the official test files, cannot run scenarios."}


def algorithm(md: dict) -> str:
    return ALGORITHMS.get(md.get("algorithm"), md.get("algorithm_label", "Model").split(" (")[0])


def target(md: dict, short: bool = False) -> str:
    return (TARGETS_SHORT if short else TARGETS)[md["scope"]]


def scope(md: dict) -> str:
    return SCOPES[md["scope"]]


def unit(md: dict) -> str:
    return UNITS[md["scope"]]


def row_unit(md: dict) -> str:
    return ROWS[md["scope"]]


def mode(md: dict) -> str:
    return MODES.get(md.get("mode"), md.get("mode", ""))


def validation(md: dict) -> str:
    h0, h1 = (pd.Timestamp(x) for x in md["periods"]["holdout"])
    return f"Chronological holdout · {h0:%b %Y} – {h1:%b %Y}"


def name(md: dict) -> str:
    """Short display name used in selectors: 'Random Forest · Intl guests'."""
    ref = " · reference" if md.get("reference") else ""
    return f"{algorithm(md)} · {target(md, short=True)}{'' if md.get('mode') == 'whatif' else ' · competition'}{ref}"


def facts(md: dict) -> dict[str, str]:
    """The four facts shown wherever a model is introduced."""
    return {"Algorithm": algorithm(md), "Prediction target": target(md), "Model scope": scope(md), "Validation": validation(md)}


def comparable(a: dict, b: dict) -> bool:
    """Metrics are only comparable for the same target, grain and holdout window."""
    return a["scope"] == b["scope"] and a["periods"]["holdout"] == b["periods"]["holdout"]


def labels(mds: list[dict]) -> dict[str, str]:
    """Selector labels for a list of models; identical names are told apart by their training time."""
    out = {m["model_id"]: name(m) for m in mds}
    for fmt in ("%d %b %H:%M", "%d %b %H:%M:%S"):
        dup = Counter(out.values())
        for m in mds:
            if dup[out[m["model_id"]]] > 1:
                out[m["model_id"]] = f"{name(m)} · trained {pd.Timestamp(m['created_at']).strftime(fmt)}"
    dup = Counter(out.values())
    for m in mds:  # same second: fall back to the end of the model ID
        if dup[out[m["model_id"]]] > 1:
            out[m["model_id"]] += f" · {m['model_id'][-6:]}"
    return out


@lru_cache(maxsize=512)
def _saved_negatives(model_id: str, stamp: float) -> int:
    from . import registry
    n = 0
    for t in ("predictions_holdout", "predictions_test"):
        df = registry.load_table(model_id, t)
        if df is not None and len(df):
            n += int((df["y_pred"] < 0).sum())
    return n


def saved_negative_predictions(md: dict) -> int:
    """Negative guest counts in the model's own saved holdout and test-period predictions (0 when a zero floor is applied)."""
    if md.get("negative_policy") == "floor_zero":
        return 0
    from . import registry
    meta = registry.model_dir(md["model_id"]) / "metadata.json"
    return _saved_negatives(md["model_id"], meta.stat().st_mtime if meta.exists() else 0.0)


def not_ready_reason(md: dict) -> str | None:
    """Why a model cannot drive the What-if Simulator (None when it can)."""
    from . import registry
    from .features import CATALOG_BY_NAME
    if md.get("mode") != "whatif":
        return "Competition model: it needs same-day hotel arrivals that do not exist on scenario dates."
    if md.get("algorithm") == "seasonal_naive":
        return "Seasonal baseline: it uses no aviation input, so it cannot respond to a scenario."
    if not all("whatif" in CATALOG_BY_NAME[f].modes for f in md.get("features", []) if f in CATALOG_BY_NAME):
        return "Uses a feature that cannot be rebuilt on a scenario date."
    if not (registry.model_dir(md["model_id"]) / "model_final.joblib").exists():
        return "The saved pipeline file is missing."
    n = saved_negative_predictions(md)
    if n:
        return (f"Its saved predictions include {n:,} negative guest counts (trained before the zero floor existed); "
                "retrain it with “Floor at zero” to use it in the simulator.")
    return None


def scenario_ready(md: dict) -> bool:
    """A model can drive the What-if Simulator only if it is a What-if model that responds to aviation inputs, every
    feature it needs can be rebuilt on a scenario date, its saved pipeline exists, and it cannot output negative guests."""
    return not_ready_reason(md) is None
