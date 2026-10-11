# Ridge Regression

Linear model with L2 penalty on standardised inputs. Fast and transparent; extrapolates linearly.

## Pipeline

`Pipeline`: ColumnTransformer (numeric features → median imputation fitted on the training rows, with a binary missing-indicator column per feature, then StandardScaler; categorical features (the market identifier) → one-hot encoding, unknown categories ignored) → `Ridge(alpha)`. Linear, so it extrapolates beyond the training range and can predict below zero for small markets. With `log_target`, the estimator is wrapped in `TransformedTargetRegressor(log1p, expm1)`; every metric is reported in guests. With the `floor_zero` policy the pipeline is wrapped in `NonNegative`, which floors predictions at 0 (raw negatives are still counted in the metadata).

Every saved artefact is the complete pipeline (preprocessing + estimator), so predicting needs only the feature columns listed in `config.json`.

## Hyperparameters

| Name | Meaning | Default | Range | Note |
|---|---|---|---|---|
| `alpha` | Regularisation α | 10.0 | 0.01 – 1000.0 | L2 penalty strength (log scale). |

## Saved models (8)

Metrics are on the chronological holdout recorded with each model and are recomputed from its saved predictions by `evaluate.py`. Compare only models with the same target and holdout (see `models/model_registry.csv`).

| Model ID | Target | Mode | Features | Holdout | MAE | RMSE | R² | WMAPE | Simulator |
|---|---|---|---|---|---|---|---|---|---|
| `INTL-WI-ridge-20261010-173514-8b8ec2` | Intl guests | What-if | 17 | 2025-02-01 → 2025-07-31 | 2,712.7 | 3,329.1 | 0.245 | 10.9% | ready |
| `ALL-COMP-ridge-20261010-124156-2feca1` (reference) | All guests | Competition | 19 | 2025-02-01 → 2025-07-31 | 1,729.1 | 2,290.1 | 0.841 | 4.2% | no |
| `DOM-COMP-ridge-20261010-124135-deb259` (reference) | Domestic guests | Competition | 19 | 2025-02-01 → 2025-07-31 | 780.7 | 1,165.1 | 0.912 | 4.9% | no |
| `INTL-COMP-ridge-20261010-124114-3b07cc` (reference) | Intl guests | Competition | 19 | 2025-02-01 → 2025-07-31 | 1,185.8 | 1,509.7 | 0.845 | 4.8% | no |
| `DOM-WI-ridge-20261010-124010-fd1946` (reference) | Domestic guests | What-if | 17 | 2025-02-01 → 2025-07-31 | 2,143.5 | 3,311.4 | 0.290 | 13.5% | ready |
| `ALL-WI-ridge-20261010-123947-282a1a` (reference) | All guests | What-if | 17 | 2025-02-01 → 2025-07-31 | 3,451.0 | 4,831.6 | 0.294 | 8.4% | ready |
| `INTL-WI-ridge-20261010-123919-3722ab` (reference) | Intl guests | What-if | 17 | 2025-02-01 → 2025-07-31 | 2,712.7 | 3,329.1 | 0.245 | 10.9% | ready |
| `MKT-WI-ridge-20261010-123104-b69aeb` (reference) | Intl guests by nationality | What-if | 18 | 2025-02-01 → 2025-07-31 | 206.8 | 359.9 | 0.844 | 29.2% | no |

Training, validation and deployment dates, the full feature list, preprocessing options and software versions of each model are in `config.json`.

## Run

```bash
# recompute the metrics of every saved model of this algorithm from its saved predictions
python models/ridge/evaluate.py
# predict the official test period with this algorithm's reference What-if model (writes results/predictions/)
python models/ridge/predict.py --start 2025-08-01 --end 2026-02-28
# print the exact training configuration of a saved model without fitting anything
python models/ridge/train.py --from INTL-WI-ridge-20261010-173514-8b8ec2 --dry-run
# refit that configuration in memory and compare with the recorded metrics (saves nothing)
python models/ridge/train.py --from INTL-WI-ridge-20261010-173514-8b8ec2 --no-save
# refit and register it as a new model in artifacts/models/
python models/ridge/train.py --from INTL-WI-ridge-20261010-173514-8b8ec2
```

## Reproducibility status

The prediction side is verified: `tests/test_legacy_artifacts.py` reloads every saved pipeline and reproduces its saved holdout predictions. Re-fitting was **not** re-run when this package was assembled (no model was retrained). `train.py --from <model_id> --no-save` refits the recorded configuration in memory and prints the recorded and new metrics side by side, so the training side can be checked on your machine. Tree ensembles are seeded (seed 42); small differences can still arise from library versions or thread scheduling.

