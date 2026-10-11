"""Section 6 — Model Evaluation: integrity checks, diagnostics, insights and importance on saved artifacts."""
import numpy as np
import pandas as pd
import pytest

from quanara import config, registry
from quanara.evaluation import diagnostics, fold_table, insights, metrics_from_predictions, permutation_table, verify_artifacts

needs_models = pytest.mark.skipif(not registry.list_models(), reason="reference models not trained")


def _observed(analytic, scope):
    if scope == "international_market":
        hm = analytic.hotel_market[analytic.hotel_market["split"] == "train"]
        return hm[["Date", "market_key", "Guests"]]
    col = {"international_total": "intl_guests", "domestic": "dom_guests", "overall": "overall_guests"}[scope]
    ht = analytic.hotel_total[analytic.hotel_total["split"] == "train"]
    return ht[["Date", col]].rename(columns={col: "Guests"})


@needs_models
def test_every_saved_model_passes_the_integrity_checks(analytic):
    for m in registry.list_models():
        h, t = registry.load_table(m["model_id"], "predictions_holdout"), registry.load_table(m["model_id"], "predictions_test")
        checks = verify_artifacts(m, h, t, _observed(analytic, m["scope"]))
        assert all(c["ok"] for c in checks), (m["model_id"], [c for c in checks if not c["ok"]])
        assert t is None or "y_true" not in t.columns     # official-test forecasts are never labelled


@needs_models
def test_integrity_check_catches_a_tampered_actual(analytic):
    m = next(x for x in registry.list_models() if x["model_id"].startswith("DOM-WI-ridge"))
    h = registry.load_table(m["model_id"], "predictions_holdout")
    h.loc[0, "y_true"] += 1
    checks = {c["check"]: c["ok"] for c in verify_artifacts(m, h, None, _observed(analytic, m["scope"]))}
    assert not checks["Saved actuals equal the observed Guests in the training files"]


def test_residual_sign_convention():
    p = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=4), "y_true": [10.0, 10, 10, 10], "y_pred": [8.0, 9, 9, 12]})
    d = diagnostics(p)
    assert d["mean_residual"] == pytest.approx(0.5) and d["under_share"] == 0.75 and d["over_share"] == 0.25
    assert metrics_from_predictions(p)["WMAPE"] == pytest.approx((2 + 1 + 1 + 2) / 40)


@needs_models
def test_cv_folds_come_from_the_saved_model_not_defaults():
    m = next((x for x in registry.list_models() if x["config"].get("cv_folds") != 4 and x["periods"]["cv_folds"]), None)
    if m is None:
        pytest.skip("no saved model with a non-default fold configuration")
    ft = fold_table(registry.load_table(m["model_id"], "predictions_cv"), m)
    assert len(ft) == m["config"]["cv_folds"] and [list(x) for x in zip(ft["val_start"], ft["val_end"])] == m["periods"]["cv_folds"]


@needs_models
def test_insights_reflect_the_measured_bias_and_peaks():
    m = next(x for x in registry.list_models() if x["model_id"].startswith("DOM-WI-ridge"))
    h = registry.load_table(m["model_id"], "predictions_holdout")
    ins = {i["title"]: i["text"] for i in insights(m, h)}
    assert "under-predict" in ins["Systematic bias"] and "+813" in ins["Systematic bias"]
    assert "systematically missed" in ins["Demand spikes"]
    assert "Eligible for the What-if Simulator" in ins["Scenario suitability"]


def test_permutation_keeps_negative_values_and_reports_spread():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=500), "noise": rng.normal(size=500)})
    y = 3 * X["a"].to_numpy()

    class Lin:
        def predict(self, Z):
            return 3 * Z["a"].to_numpy() + 0.01 * Z["noise"].to_numpy()
    t = permutation_table(Lin(), X, y, {"a": ["a"], "noise": ["noise"]}, n_repeats=5).set_index("name")
    assert t.loc["a", "mean"] > 1 and t.loc["a", "sd"] >= 0 and abs(t.loc["noise", "mean"]) < 0.1
