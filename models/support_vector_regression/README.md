# Support Vector Regression

Support Vector Regression with an RBF kernel on standardised inputs and target. Slow on the market panel.

## Pipeline

`TransformedTargetRegressor` around `Pipeline`: ColumnTransformer (numeric features → median imputation fitted on the training rows, with a binary missing-indicator column per feature, then StandardScaler; categorical features (the market identifier) → one-hot encoding, unknown categories ignored) → `SVR(kernel='rbf', C, epsilon, gamma)`. The target is standardised (log1p first when `log_target` is on) and predictions are transformed back to guests. Kernel cost grows quadratically with rows: the 29k-row market panel takes several minutes. With the `floor_zero` policy the pipeline is wrapped in `NonNegative`, which floors predictions at 0 (raw negatives are still counted in the metadata).

Every saved artefact is the complete pipeline (preprocessing + estimator), so predicting needs only the feature columns listed in `config.json`.

## Hyperparameters

| Name | Meaning | Default | Range | Note |
|---|---|---|---|---|
| `C` | C | 3.0 | 0.01 – 100.0 | Inverse regularisation (on the standardised target). |
| `epsilon` | ε | 0.05 | 0.0 – 1.0 | Insensitive-tube width (standardised target units). |
| `gamma` | γ (RBF) | scale | ['scale', 'auto'] | RBF kernel width. |

## Saved models (2)

Metrics are on the chronological holdout recorded with each model and are recomputed from its saved predictions by `evaluate.py`. Compare only models with the same target and holdout (see `models/model_registry.csv`).

| Model ID | Target | Mode | Features | Holdout | MAE | RMSE | R² | WMAPE | Simulator |
|---|---|---|---|---|---|---|---|---|---|
| `INTL-WI-svr-20261010-123947-3d1cd1` (reference) | Intl guests | What-if | 17 | 2025-02-01 → 2025-07-31 | 2,469.0 | 2,954.5 | 0.406 | 9.9% | ready |
| `MKT-WI-svr-20261010-123918-b5964e` (reference) | Intl guests by nationality | What-if | 18 | 2025-02-01 → 2025-07-31 | 183.2 | 365.1 | 0.840 | 25.9% | ready |

Training, validation and deployment dates, the full feature list, preprocessing options and software versions of each model are in `config.json`.

## Run

```bash
# recompute the metrics of every saved model of this algorithm from its saved predictions
python models/support_vector_regression/evaluate.py
# predict the official test period with this algorithm's reference What-if model (writes results/predictions/)
python models/support_vector_regression/predict.py --start 2025-08-01 --end 2026-02-28
# print the exact training configuration of a saved model without fitting anything
python models/support_vector_regression/train.py --from INTL-WI-svr-20261010-123947-3d1cd1 --dry-run
# refit that configuration in memory and compare with the recorded metrics (saves nothing)
python models/support_vector_regression/train.py --from INTL-WI-svr-20261010-123947-3d1cd1 --no-save
# refit and register it as a new model in artifacts/models/
python models/support_vector_regression/train.py --from INTL-WI-svr-20261010-123947-3d1cd1
```

## Reproducibility status

The prediction side is verified: `tests/test_legacy_artifacts.py` reloads every saved pipeline and reproduces its saved holdout predictions. Re-fitting was **not** re-run when this package was assembled (no model was retrained). `train.py --from <model_id> --no-save` refits the recorded configuration in memory and prints the recorded and new metrics side by side, so the training side can be checked on your machine. Tree ensembles are seeded (seed 42); small differences can still arise from library versions or thread scheduling.

