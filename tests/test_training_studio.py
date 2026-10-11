"""Section 5 — ML Training Studio: algorithms, constraints, policies, persistence and comparison rules."""
import json

import numpy as np
import pandas as pd
import pytest

from quanara import config, registry
from quanara.evaluation import comparison_key, diagnostics, metrics_from_predictions
from quanara.features import MONOTONE_INCREASING, default_features
from quanara.integration import model_frame, training_rows
from quanara.models import NonNegative, make_model
from quanara.training import ABLATIONS_DIR, TrainConfig, ablation_sets, run_ablation, train

needs_models = pytest.mark.skipif(not registry.list_models(), reason="reference models not trained")


def _ref(prefix: str) -> str | None:
    return next((m["model_id"] for m in registry.list_models() if m["model_id"].startswith(prefix) and m.get("reference")), None)


def test_extra_trees_trains_with_monotone_constraint(analytic, tmp_registry):
    feats = default_features("international_total", "whatif")
    r = train(TrainConfig("international_total", "whatif", "extra_trees", feats, params={"n_estimators": 60}, cv_folds=1,
                          cv_horizon_days=91, permutation_importance=False), analytic, save=True)
    md = registry.load_metadata(r.model_id)
    assert md["monotone"] and set(md["monotone_features"]) == {f for f in feats if f in MONOTONE_INCREASING}
    est = registry.load_model(r.model_id, "eval")
    cst = dict(zip(est.names_, est.model_.monotonic_cst))
    assert all(cst[f"num__{f}"] == 1 for f in md["monotone_features"]) and sum(cst.values()) == len(md["monotone_features"])
    # holding everything else fixed, more seats/passengers/flights cannot lower the prediction
    X = training_rows(model_frame(analytic, "international_total", "whatif"))[feats].head(200).copy()
    base = est.predict(X)
    for f in md["monotone_features"]:
        X[f] = X[f] * 1.25
    assert (est.predict(X) >= base - 1e-9).all()
    assert md["cv_fold_details"][0]["train_rows"] > 0 and md["holdout_dates"] == 181


def test_negative_policy_floors_and_counts(analytic):
    feats = default_features("international_market", "whatif")
    raw = train(TrainConfig("international_market", "whatif", "ridge", feats, cv_folds=0, permutation_importance=False), analytic, save=False)
    flo = train(TrainConfig("international_market", "whatif", "ridge", feats, cv_folds=0, permutation_importance=False, negative_policy="floor_zero"),
                analytic, save=False)
    assert raw.metadata["negative_predictions_raw"]["holdout"] == int((raw.holdout["y_pred"] < 0).sum()) > 0
    assert (flo.holdout["y_pred"] >= 0).all()
    assert flo.metadata["negative_predictions_raw"]["holdout"] == raw.metadata["negative_predictions_raw"]["holdout"]
    assert flo.metadata["metrics_holdout"]["MAE"] <= raw.metadata["metrics_holdout"]["MAE"]
    assert isinstance(make_model("ridge", feats, negative_policy="floor_zero"), NonNegative)


@needs_models
def test_saved_pipelines_reload_and_reproduce_their_predictions(analytic):
    mid = _ref("INTL-WI-ridge")
    md = registry.load_metadata(mid)
    frame = model_frame(analytic, md["scope"], md["mode"])
    rows = training_rows(frame)
    hold = rows[rows["Date"] >= pd.Timestamp(md["config"]["holdout_start"])]
    saved = registry.load_table(mid, "predictions_holdout")
    assert np.allclose(registry.load_model(mid, "eval").predict(hold[md["features"]]), saved["y_pred"], rtol=1e-9, atol=1e-6)
    test = frame[frame["split"] == "test"]
    saved_t = registry.load_table(mid, "predictions_test")
    assert np.allclose(registry.load_model(mid, "final").predict(test[md["features"]]), saved_t["y_pred"], rtol=1e-9, atol=1e-6)


@needs_models
def test_screenshot_models_reproduce_from_artifacts():
    dom = _ref("DOM-WI-ridge")
    m = metrics_from_predictions(registry.load_table(dom, "predictions_holdout"))
    assert (round(m["MAE"], 1), round(m["RMSE"], 1), round(m["R2"], 3), round(m["WMAPE"], 3)) == (2143.5, 3311.4, 0.290, 0.135)
    d = diagnostics(registry.load_table(dom, "predictions_holdout"))
    assert round(d["mean_residual"], 1) == 812.8 and d["under_share"] > 0.5


@needs_models
def test_comparison_groups_require_identical_evaluation_rows():
    a, b, c = _ref("DOM-WI-ridge"), _ref("DOM-WI-catboost"), _ref("INTL-WI-ridge")
    key = lambda m: comparison_key(registry.load_metadata(m), registry.load_table(m, "predictions_holdout"))
    assert key(a) == key(b)          # same target, grain and holdout rows
    assert key(a) != key(c)          # domestic vs international totals are never ranked together


@needs_models
def test_reference_models_are_protected_and_competition_models_cannot_activate():
    ref = _ref("DOM-WI-ridge")
    with pytest.raises(PermissionError):
        registry.delete(ref)
    comp = _ref("DOM-COMP")
    with pytest.raises(ValueError):
        registry.activate(comp)


def test_ablation_records_its_source_and_identical_split(analytic, tmp_registry):
    feats = default_features("international_total", "whatif")
    sets = ablation_sets(feats, ["All selected features", "Aviation only", "Calendar only", "All minus guest history"])
    assert sets["All selected features"] == feats and all(f in feats for f in sets["Aviation only"])
    abl_id, summary = run_ablation(TrainConfig("international_total", "whatif", "ridge", feats), analytic,
                                   experiments=["Aviation only", "Calendar only"], source_model_id="TEST-MODEL")
    from quanara import training
    cfg = json.loads((training.ABLATIONS_DIR / abl_id / "config.json").read_text())
    pred = pd.read_csv(training.ABLATIONS_DIR / abl_id / "predictions_holdout.csv")
    assert cfg["source_model_id"] == "TEST-MODEL" and set(pred["feature_set"]) == {"All selected features", "Aviation only", "Calendar only"}
    assert pred.groupby("feature_set")["Date"].agg(["min", "max", "count"]).nunique().eq(1).all()
