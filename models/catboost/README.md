# CatBoost

Gradient-boosted symmetric trees with native categorical handling.

## Pipeline

`CatBoostPipeline`: selects the feature columns, passes the market identifier to CatBoost as a native categorical feature and leaves missing numeric values to CatBoost's native handling (no imputation). Loss RMSE, seed 42. When the shape constraint is on, monotone_constraints = +1 for the volume features in `features.MONOTONE_INCREASING`. With `log_target`, the estimator is wrapped in `TransformedTargetRegressor(log1p, expm1)`; every metric is reported in guests. With the `floor_zero` policy the pipeline is wrapped in `NonNegative`, which floors predictions at 0 (raw negatives are still counted in the metadata).

Every saved artefact is the complete pipeline (preprocessing + estimator), so predicting needs only the feature columns listed in `config.json`.

## Hyperparameters

| Name | Meaning | Default | Range | Note |
|---|---|---|---|---|
| `iterations` | Iterations | 800 | 100 – 3000 | Boosting rounds. |
| `learning_rate` | Learning rate | 0.05 | 0.005 – 0.3 | Shrinkage per round. |
| `depth` | Depth | 6 | 2 – 10 | Symmetric tree depth. |
| `l2_leaf_reg` | L2 leaf regularisation | 3.0 | 0.1 – 30.0 | Leaf-value penalty. |

## Saved models (8)

Metrics are on the chronological holdout recorded with each model and are recomputed from its saved predictions by `evaluate.py`. Compare only models with the same target and holdout (see `models/model_registry.csv`).

| Model ID | Target | Mode | Features | Holdout | MAE | RMSE | R² | WMAPE | Simulator |
|---|---|---|---|---|---|---|---|---|---|
| `ALL-COMP-catboost-20261010-124220-62ae2d` (reference) | All guests | Competition | 19 | 2025-02-01 → 2025-07-31 | 2,402.0 | 3,338.6 | 0.663 | 5.9% | no |
| `DOM-COMP-catboost-20261010-124156-8da746` (reference) | Domestic guests | Competition | 19 | 2025-02-01 → 2025-07-31 | 1,081.9 | 1,384.8 | 0.876 | 6.8% | no |
| `INTL-COMP-catboost-20261010-124135-a7f113` (reference) | Intl guests | Competition | 19 | 2025-02-01 → 2025-07-31 | 1,778.6 | 2,357.2 | 0.622 | 7.1% | no |
| `MKT-COMP-catboost-20261010-124105-534ddb` (reference) | Intl guests by nationality | Competition | 20 | 2025-02-01 → 2025-07-31 | 81.0 | 183.7 | 0.952 | 13.9% | no |
| `DOM-WI-catboost-20261010-124029-d248ff` (reference) | Domestic guests | What-if | 17 | 2025-02-01 → 2025-07-31 | 2,822.8 | 4,022.6 | -0.048 | 17.8% | ready |
| `ALL-WI-catboost-20261010-124009-41b203` (reference) | All guests | What-if | 17 | 2025-02-01 → 2025-07-31 | 3,789.9 | 5,173.4 | 0.190 | 9.3% | ready |
| `INTL-WI-catboost-20261010-123941-734b77` (reference) | Intl guests | What-if | 17 | 2025-02-01 → 2025-07-31 | 2,109.0 | 2,796.7 | 0.467 | 8.5% | ready |
| `MKT-WI-catboost-20261010-123103-7d603b` (reference) | Intl guests by nationality | What-if | 18 | 2025-02-01 → 2025-07-31 | 160.1 | 317.1 | 0.879 | 22.6% | ready |

Training, validation and deployment dates, the full feature list, preprocessing options and software versions of each model are in `config.json`.

## Run

```bash
# recompute the metrics of every saved model of this algorithm from its saved predictions
python models/catboost/evaluate.py
# predict the official test period with this algorithm's reference What-if model (writes results/predictions/)
python models/catboost/predict.py --start 2025-08-01 --end 2026-02-28
# print the exact training configuration of a saved model without fitting anything
python models/catboost/train.py --from ALL-COMP-catboost-20261010-124220-62ae2d --dry-run
# refit that configuration in memory and compare with the recorded metrics (saves nothing)
python models/catboost/train.py --from ALL-COMP-catboost-20261010-124220-62ae2d --no-save
# refit and register it as a new model in artifacts/models/
python models/catboost/train.py --from ALL-COMP-catboost-20261010-124220-62ae2d
```

## Reproducibility status

The prediction side is verified: `tests/test_legacy_artifacts.py` reloads every saved pipeline and reproduces its saved holdout predictions. Re-fitting was **not** re-run when this package was assembled (no model was retrained). `train.py --from <model_id> --no-save` refits the recorded configuration in memory and prints the recorded and new metrics side by side, so the training side can be checked on your machine. Tree ensembles are seeded (seed 42); small differences can still arise from library versions or thread scheduling.

