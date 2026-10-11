"""Metrics, model factory and the chronological training protocol."""
import numpy as np
import pandas as pd
import pytest

from quanara import config, registry
from quanara.evaluation import metrics, metrics_from_predictions
from quanara.features import default_features
from quanara.training import TrainConfig, cv_folds, run_ablation, train


def test_metrics_known_values():
    m = metrics([10, 20, 30], [12, 18, 30])
    assert np.isclose(m["MAE"], 4 / 3)
    assert np.isclose(m["RMSE"], np.sqrt(8 / 3))
    assert np.isclose(m["WMAPE"], 4 / 60)
    assert np.isclose(m["R2"], 1 - 8 / 200)


def test_cv_folds_are_chronological_and_before_holdout():
    dates = pd.Series(pd.date_range("2023-03-01", "2025-01-31"))
    folds = cv_folds(dates, pd.Timestamp("2025-02-01"), 4, 91)
    assert len(folds) == 4
    assert folds[-1][1] < pd.Timestamp("2025-02-01")
    assert all(a[1] < b[0] for a, b in zip(folds, folds[1:]))


def test_invalid_feature_for_mode_rejected():
    with pytest.raises(ValueError):
        TrainConfig("international_total", "whatif", "ridge", ["new_arrivals"]).validate()


def test_train_save_reload_and_traceable_metrics(analytic, tmp_registry):
    feats = default_features("international_total", "whatif")
    r = train(TrainConfig("international_total", "whatif", "ridge", feats, cv_folds=2), analytic)
    md = registry.load_metadata(r.model_id)
    hold = registry.load_table(r.model_id, "predictions_holdout")
    # holdout strictly after the fit period
    assert hold["Date"].min() == pd.Timestamp(md["periods"]["holdout"][0]) >= pd.Timestamp(config.DEFAULT_HOLDOUT_START)
    assert pd.Timestamp(md["periods"]["eval_fit"][1]) < hold["Date"].min()
    # displayed metrics are recomputable from the saved predictions
    m = metrics_from_predictions(hold)
    for k in ("MAE", "RMSE", "R2", "WMAPE"):
        assert np.isclose(m[k], md["metrics_holdout"][k])
    # reload produces identical predictions
    model = registry.load_model(r.model_id, "eval")
    from quanara.integration import model_frame, training_rows
    rows = training_rows(model_frame(analytic, "international_total", "whatif"))
    h = rows[rows["Date"] >= config.DEFAULT_HOLDOUT_START]
    assert np.allclose(model.predict(h[feats]), hold["y_pred"])
    assert len(registry.load_table(r.model_id, "predictions_test")) == 212


def test_reproducible_with_seed(analytic, tmp_registry):
    feats = ["p2p_7d", "seats_7d", "dow", "doy_sin", "doy_cos", "guests_lag364_7d"]
    cfg = dict(scope="international_total", mode="whatif", algorithm="random_forest", features=feats, params={"n_estimators": 60},
               cv_folds=0, permutation_importance=False)
    a = train(TrainConfig(**cfg), analytic, save=False)
    b = train(TrainConfig(**cfg), analytic, save=False)
    assert np.allclose(a.holdout["y_pred"], b.holdout["y_pred"])


@pytest.mark.parametrize("algo", ["seasonal_naive", "catboost", "xgboost", "svr"])
def test_every_algorithm_trains(analytic, tmp_registry, algo):
    feats = default_features("international_total", "whatif")
    params = {"iterations": 100} if algo == "catboost" else {"n_estimators": 60} if algo == "xgboost" else {}
    r = train(TrainConfig("international_total", "whatif", algo, feats if algo != "seasonal_naive" else [], params=params,
                          cv_folds=0, permutation_importance=False), analytic, save=False)
    assert np.isfinite(r.metadata["metrics_holdout"]["MAE"])


def test_monotone_constraint_never_decreases_with_p2p(analytic, tmp_registry):
    feats = default_features("international_total", "whatif")
    r = train(TrainConfig("international_total", "whatif", "xgboost", feats, params={"n_estimators": 80}, cv_folds=0,
                          permutation_importance=False, monotone=True), analytic)
    model = registry.load_model(r.model_id)
    from quanara.integration import model_frame, training_rows
    X = training_rows(model_frame(analytic, "international_total", "whatif")).head(50)[feats].copy()
    base = model.predict(X)
    for c in ("p2p", "p2p_7d", "p2p_28d", "seats_7d", "transfer_7d", "flights_7d"):
        X[c] = X[c] * 1.3
    assert (model.predict(X) >= base - 1e-6).all()


def test_ablation_saves_traceable_predictions(analytic, tmp_registry):
    feats = default_features("international_total", "whatif")
    abl_id, summary = run_ablation(TrainConfig("international_total", "whatif", "ridge", feats), analytic)
    from quanara import training
    pred = pd.read_csv(training.ABLATIONS_DIR / abl_id / "predictions_holdout.csv")
    assert set(pred["feature_set"]) == set(summary.loc[summary["n_features"] > 0, "feature_set"])
    assert pred.groupby("feature_set")["Date"].nunique().nunique() == 1  # identical holdout for every variant
