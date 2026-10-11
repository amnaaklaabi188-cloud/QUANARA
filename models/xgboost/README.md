# XGBoost

Gradient-boosted trees (histogram method).

## Pipeline

`PrepModel`: ColumnTransformer (numeric features → median imputation fitted on the training rows, with a binary missing-indicator column per feature, no scaling; categorical features (the market identifier) → one-hot encoding, unknown categories ignored) → `XGBRegressor(tree_method='hist')` (seed 42). When the shape constraint is on, `monotone_constraints` is aligned to the transformed columns (+1 on volume features). With `log_target`, the estimator is wrapped in `TransformedTargetRegressor(log1p, expm1)`; every metric is reported in guests. With the `floor_zero` policy the pipeline is wrapped in `NonNegative`, which floors predictions at 0 (raw negatives are still counted in the metadata).

Every saved artefact is the complete pipeline (preprocessing + estimator), so predicting needs only the feature columns listed in `config.json`.

## Hyperparameters

| Name | Meaning | Default | Range | Note |
|---|---|---|---|---|
| `n_estimators` | Trees | 600 | 50 – 3000 | Boosting rounds. |
| `learning_rate` | Learning rate | 0.05 | 0.005 – 0.3 | Shrinkage per round. |
| `max_depth` | Max depth | 5 | 2 – 12 | Tree depth. |
| `subsample` | Row subsample | 0.8 | 0.3 – 1.0 | Row sampling per tree. |
| `colsample_bytree` | Column subsample | 0.8 | 0.3 – 1.0 | Feature sampling per tree. |
| `min_child_weight` | Min child weight | 3.0 | 0.0 – 50.0 | Minimum hessian per leaf. |

## Saved models (3)

Metrics are on the chronological holdout recorded with each model and are recomputed from its saved predictions by `evaluate.py`. Compare only models with the same target and holdout (see `models/model_registry.csv`).

| Model ID | Target | Mode | Features | Holdout | MAE | RMSE | R² | WMAPE | Simulator |
|---|---|---|---|---|---|---|---|---|---|
| `MKT-COMP-xgboost-20261010-124113-edc9dd` (reference) | Intl guests by nationality | Competition | 20 | 2025-02-01 → 2025-07-31 | 76.5 | 160.7 | 0.963 | 13.1% | no |
| `INTL-WI-xgboost-20261010-123947-af9e7a` (reference) | Intl guests | What-if | 17 | 2025-02-01 → 2025-07-31 | 2,084.7 | 2,534.9 | 0.563 | 8.4% | ready |
| `MKT-WI-xgboost-20261010-123145-4d4b72` (reference) | Intl guests by nationality | What-if | 18 | 2025-02-01 → 2025-07-31 | 177.7 | 341.0 | 0.860 | 25.1% | ready |

Training, validation and deployment dates, the full feature list, preprocessing options and software versions of each model are in `config.json`.

## Run

```bash
# recompute the metrics of every saved model of this algorithm from its saved predictions
python models/xgboost/evaluate.py
# predict the official test period with this algorithm's reference What-if model (writes results/predictions/)
python models/xgboost/predict.py --start 2025-08-01 --end 2026-02-28
# print the exact training configuration of a saved model without fitting anything
python models/xgboost/train.py --from MKT-COMP-xgboost-20261010-124113-edc9dd --dry-run
# refit that configuration in memory and compare with the recorded metrics (saves nothing)
python models/xgboost/train.py --from MKT-COMP-xgboost-20261010-124113-edc9dd --no-save
# refit and register it as a new model in artifacts/models/
python models/xgboost/train.py --from MKT-COMP-xgboost-20261010-124113-edc9dd
```

## Reproducibility status

The prediction side is verified: `tests/test_legacy_artifacts.py` reloads every saved pipeline and reproduces its saved holdout predictions. Re-fitting was **not** re-run when this package was assembled (no model was retrained). `train.py --from <model_id> --no-save` refits the recorded configuration in memory and prints the recorded and new metrics side by side, so the training side can be checked on your machine. Tree ensembles are seeded (seed 42); small differences can still arise from library versions or thread scheduling.

