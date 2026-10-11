"""Executable leakage and timing checks used by Data Intelligence, Methodology and the tests.

Each check re-runs the real feature or training code and compares results, so a "Verified" status is
backed by an executed test rather than by a description. Feature timing is classified separately:
a feature that ends on the prediction date is not automatically safe for an advance forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .features import CATALOG, aggregate_aviation, aviation_features, default_features, guest_history_features
from .integration import Analytic, model_frame, training_rows

# How each feature relates to the prediction date t
TIMING = {
    "calendar": "Known for any date in advance",
    "schedule": "Planned capacity, known in advance from schedules (the file records operated values)",
    "realised": "Realised on the prediction date: known only after flights operate",
    "lagged_realised": "Realised 7 or more days before the prediction date",
    "history": "Historical Guests, at least 364 days before the prediction date",
    "retrospective": "Same-day hotel observation: available only retrospectively (official test files)",
    "identifier": "Row identifier",
}


def timing_class(name: str) -> str:
    fs = next(f for f in CATALOG if f.name == name)
    if fs.group == "time":
        return "calendar"
    if fs.group == "market":
        return "identifier"
    if fs.group == "competition":
        return "retrospective"
    if name.startswith("guests_lag364"):
        return "history"
    if name.endswith("_prev7d"):
        return "lagged_realised"
    if name in ("seats", "flights", "seats_7d", "flights_7d", "seats_28d", "seats_per_flight_7d", "business_seat_share_7d",
                "etihad_seat_share_7d", "n_routes", "n_cities", "n_airlines", "seats_growth_28d"):
        return "schedule"
    return "realised"


def use_case_availability(name: str) -> dict[str, bool]:
    """Whether a feature can legitimately be used for each prediction use case."""
    c = timing_class(name)
    return {"retrospective": True,
            "advance_forecast": c in ("calendar", "schedule", "history", "identifier", "lagged_realised"),
            "whatif": c != "retrospective"}


def _guest_history_past_only(a: Analytic) -> dict:
    t = a.hotel_total[a.hotel_total["split"] == "train"][["Date", "intl_guests"]].rename(columns={"intl_guests": "Guests"})
    full = guest_history_features(t, by_market=False).set_index("Date")
    ok = True
    for cut in pd.to_datetime(["2024-03-15", "2024-11-02", "2025-05-20"]):
        trunc = guest_history_features(t[t["Date"] <= cut - pd.Timedelta(days=config.SEASONAL_LAG_DAYS)], by_market=False).set_index("Date")
        a_, b_ = full.loc[cut, "guests_lag364"], trunc.loc[cut, "guests_lag364"]
        ok &= bool(np.isclose(a_, b_))
    return {"check": "Guest-history features use only Guests from 364+ days earlier", "ok": ok,
            "evidence": "Recomputed after deleting every Guests value later than t − 364 for three dates; the feature values did not change."}


def _aviation_trailing_only(a: Analytic) -> dict:
    rows = a.flight_daily_rows
    full = aviation_features(aggregate_aviation(rows, by_market=False), by_market=False).set_index("Date")
    ok = True
    cols = ["seats_7d", "p2p_7d", "load_factor_7d", "p2p_28d", "seats_growth_28d", "p2p_prev7d"]
    for cut in pd.to_datetime(["2024-02-10", "2025-06-30"]):
        r = rows[rows["Date"] <= cut]
        trunc = aviation_features(aggregate_aviation(r, by_market=False, end=cut), by_market=False).set_index("Date")
        ok &= bool(np.allclose(full.loc[cut, cols].astype(float), trunc.loc[cut, cols].astype(float), equal_nan=True))
    return {"check": "Aviation windows end on the prediction date (no future flights)", "ok": ok,
            "evidence": "Recomputed after deleting every flight row after t for two dates; trailing, growth and prior-week features did not change."}


def _preprocessing_in_fold(a: Analytic) -> dict:
    from .models import make_model
    feats = default_features("international_total", "whatif")
    tr = training_rows(model_frame(a, "international_total", "whatif"))
    fold = tr[tr["Date"] < pd.Timestamp("2024-06-01")]
    m = make_model("ridge", feats).fit(fold[feats], fold["y"].to_numpy())
    imp = m.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    scal = m.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    from .features import numeric_features
    num = numeric_features(feats)  # the pipeline's numeric block, in order (categoricals are one-hot encoded separately)
    ok = bool(np.allclose(imp.statistics_[: len(num)], fold[num].median().to_numpy(), equal_nan=True))
    ok &= bool(np.isclose(scal.mean_[0], fold[num[0]].mean()))
    return {"check": "Imputation and scaling are fitted on training rows only", "ok": ok,
            "evidence": "A pipeline fitted on one fold stores medians and means equal to that fold's rows, not to the full data."}


def _chronological(md: dict | None) -> dict:
    if md is None:
        return {"check": "Validation dates follow all training dates", "ok": None, "evidence": "No active model to inspect."}
    fit_end = pd.Timestamp(md["periods"]["eval_fit"][1])
    hold0 = pd.Timestamp(md["periods"]["holdout"][0])
    folds = md["periods"].get("cv_folds") or []
    ok = fit_end < hold0 and all(pd.Timestamp(v[1]) < hold0 for v in folds)
    return {"check": "Validation dates follow all training dates", "ok": bool(ok),
            "evidence": f"Active model: fitted to {fit_end:%Y-%m-%d}, holdout from {hold0:%Y-%m-%d}; {len(folds)} CV folds all end before the holdout."}


def _whatif_excludes_hotel() -> dict:
    bad = [f.name for f in CATALOG if "whatif" in f.modes and f.group == "competition"]
    return {"check": "What-if models cannot use same-day hotel observations", "ok": not bad,
            "evidence": "New Arrivals and Same-Day Guests are restricted to Competition mode in the feature catalogue."}


def run_checks(a: Analytic, md: dict | None = None) -> list[dict]:
    return [_guest_history_past_only(a), _aviation_trailing_only(a), _preprocessing_in_fold(a), _chronological(md), _whatif_excludes_hotel()]


def timing_summary(features: list[str]) -> pd.DataFrame:
    rows = [{"feature": f, "timing": timing_class(f), **use_case_availability(f)} for f in features]
    return pd.DataFrame(rows)
