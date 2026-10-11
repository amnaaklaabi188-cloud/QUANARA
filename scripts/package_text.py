"""Fixed descriptions used by scripts/build_research_docs.py — each one restates the code in quanara/models.py."""

_IMPUTE = "numeric features → median imputation fitted on the training rows, with a binary missing-indicator column per feature"
_ONEHOT = "categorical features (the market identifier) → one-hot encoding, unknown categories ignored"
_LOG = "With `log_target`, the estimator is wrapped in `TransformedTargetRegressor(log1p, expm1)`; every metric is reported in guests."
_FLOOR = "With the `floor_zero` policy the pipeline is wrapped in `NonNegative`, which floors predictions at 0 (raw negatives are still counted in the metadata)."

PREPROCESSING = {
    "catboost": ("`CatBoostPipeline`: selects the feature columns, passes the market identifier to CatBoost as a native categorical feature and leaves "
                 "missing numeric values to CatBoost's native handling (no imputation). Loss RMSE, seed 42. When the shape constraint is on, "
                 "monotone_constraints = +1 for the volume features in `features.MONOTONE_INCREASING`. " + _LOG + " " + _FLOOR),
    "ridge": (f"`Pipeline`: ColumnTransformer ({_IMPUTE}, then StandardScaler; {_ONEHOT}) → `Ridge(alpha)`. Linear, so it extrapolates beyond the "
              "training range and can predict below zero for small markets. " + _LOG + " " + _FLOOR),
    "random_forest": (f"`PrepModel`: ColumnTransformer ({_IMPUTE}, no scaling; {_ONEHOT}) → `RandomForestRegressor` (seed 42, all cores). When the "
                      "shape constraint is on, `monotonic_cst` is aligned to the transformed columns: +1 on volume features, 0 on indicators and one-hot "
                      "columns. Trees predict flat beyond the training range. " + _LOG + " " + _FLOOR),
    "xgboost": (f"`PrepModel`: ColumnTransformer ({_IMPUTE}, no scaling; {_ONEHOT}) → `XGBRegressor(tree_method='hist')` (seed 42). When the shape "
                "constraint is on, `monotone_constraints` is aligned to the transformed columns (+1 on volume features). " + _LOG + " " + _FLOOR),
    "svr": (f"`TransformedTargetRegressor` around `Pipeline`: ColumnTransformer ({_IMPUTE}, then StandardScaler; {_ONEHOT}) → `SVR(kernel='rbf', C, "
            "epsilon, gamma)`. The target is standardised (log1p first when `log_target` is on) and predictions are transformed back to guests. "
            "Kernel cost grows quadratically with rows: the 29k-row market panel takes several minutes. " + _FLOOR),
    "seasonal_naive": ("`SeasonalNaive`: no learned parameters except fallbacks. ŷ(t) = Guests(t − 364 days) — the same weekday last year — for the same "
                       "nationality (market scope) or the same total; missing values fall back to last year's 7-day mean, then the training median of "
                       "the market (or overall). Uses no aviation input, so it cannot respond to a scenario; it is the benchmark every model must beat."),
    "extra_trees": (f"`PrepModel`: ColumnTransformer ({_IMPUTE}; {_ONEHOT}) → `ExtraTreesRegressor` (seed 42)."),
}

RUN_COMMANDS = """
# recompute the metrics of every saved model of this algorithm from its saved predictions
python models/{folder}/evaluate.py
# predict the official test period with this algorithm's reference What-if model (writes results/predictions/)
python models/{folder}/predict.py --start 2025-08-01 --end 2026-02-28
# print the exact training configuration of a saved model without fitting anything
python models/{folder}/train.py --from {first} --dry-run
# refit that configuration in memory and compare with the recorded metrics (saves nothing)
python models/{folder}/train.py --from {first} --no-save
# refit and register it as a new model in artifacts/models/
python models/{folder}/train.py --from {first}
"""
