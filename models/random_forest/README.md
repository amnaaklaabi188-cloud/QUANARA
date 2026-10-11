# Random Forest

Averaged decision trees on bootstrap samples. Robust; cannot extrapolate beyond the training range.

## Pipeline

`PrepModel`: ColumnTransformer (numeric features → median imputation fitted on the training rows, with a binary missing-indicator column per feature, no scaling; categorical features (the market identifier) → one-hot encoding, unknown categories ignored) → `RandomForestRegressor` (seed 42, all cores). When the shape constraint is on, `monotonic_cst` is aligned to the transformed columns: +1 on volume features, 0 on indicators and one-hot columns. Trees predict flat beyond the training range. With `log_target`, the estimator is wrapped in `TransformedTargetRegressor(log1p, expm1)`; every metric is reported in guests. With the `floor_zero` policy the pipeline is wrapped in `NonNegative`, which floors predictions at 0 (raw negatives are still counted in the metadata).

Every saved artefact is the complete pipeline (preprocessing + estimator), so predicting needs only the feature columns listed in `config.json`.

## Hyperparameters

| Name | Meaning | Default | Range | Note |
|---|---|---|---|---|
| `n_estimators` | Trees | 400 | 50 – 1000 | Number of trees. |
| `max_depth` | Max depth (0 = unlimited) | 12 | 0 – 40 | Maximum tree depth. |
| `min_samples_leaf` | Min samples per leaf | 3 | 1 – 50 | Larger values smooth predictions. |
| `max_features` | Feature fraction per split | 0.5 | 0.1 – 1.0 | Share of features considered at each split. |

## Saved models (8)

Metrics are on the chronological holdout recorded with each model and are recomputed from its saved predictions by `evaluate.py`. Compare only models with the same target and holdout (see `models/model_registry.csv`).

| Model ID | Target | Mode | Features | Holdout | MAE | RMSE | R² | WMAPE | Simulator |
|---|---|---|---|---|---|---|---|---|---|
| `INTL-WI-random_forest-20261010-190246-8c6b93` | Intl guests | What-if | 17 | 2025-02-01 → 2025-07-31 | 1,662.3 | 2,077.3 | 0.706 | 6.7% | ready |
| `INTL-COMP-random_forest-20261010-182951-c5bdb5` | Intl guests | Competition | 19 | 2025-02-01 → 2025-07-31 | 1,300.1 | 1,692.9 | 0.805 | 5.2% | no |
| `INTL-COMP-random_forest-20261010-182938-e99d18` | Intl guests | Competition | 19 | 2025-02-01 → 2025-07-31 | 1,352.4 | 1,732.5 | 0.796 | 5.4% | no |
| `INTL-COMP-random_forest-20261010-182935-e99d18` | Intl guests | Competition | 19 | 2025-02-01 → 2025-07-31 | 1,352.4 | 1,732.5 | 0.796 | 5.4% | no |
| `MKT-COMP-random_forest-20261010-153327-ed8a8d` | Intl guests by nationality | Competition | 20 | 2025-02-01 → 2025-07-31 | 96.3 | 251.8 | 0.909 | 16.5% | no |
| `ALL-WI-random_forest-20261010-123950-84eba1` (reference) | All guests | What-if | 17 | 2025-02-01 → 2025-07-31 | 3,697.6 | 5,151.7 | 0.197 | 9.1% | ready |
| `INTL-WI-random_forest-20261010-123923-f81968` (reference) | Intl guests | What-if | 17 | 2025-02-01 → 2025-07-31 | 1,807.6 | 2,244.8 | 0.657 | 7.2% | ready |
| `MKT-WI-random_forest-20261010-123137-c5fe30` (reference) | Intl guests by nationality | What-if | 18 | 2025-02-01 → 2025-07-31 | 193.3 | 425.9 | 0.782 | 27.3% | ready |

Training, validation and deployment dates, the full feature list, preprocessing options and software versions of each model are in `config.json`.

## Run

```bash
# recompute the metrics of every saved model of this algorithm from its saved predictions
python models/random_forest/evaluate.py
# predict the official test period with this algorithm's reference What-if model (writes results/predictions/)
python models/random_forest/predict.py --start 2025-08-01 --end 2026-02-28
# print the exact training configuration of a saved model without fitting anything
python models/random_forest/train.py --from INTL-WI-random_forest-20261010-190246-8c6b93 --dry-run
# refit that configuration in memory and compare with the recorded metrics (saves nothing)
python models/random_forest/train.py --from INTL-WI-random_forest-20261010-190246-8c6b93 --no-save
# refit and register it as a new model in artifacts/models/
python models/random_forest/train.py --from INTL-WI-random_forest-20261010-190246-8c6b93
```

## Reproducibility status

The prediction side is verified: `tests/test_legacy_artifacts.py` reloads every saved pipeline and reproduces its saved holdout predictions. Re-fitting was **not** re-run when this package was assembled (no model was retrained). `train.py --from <model_id> --no-save` refits the recorded configuration in memory and prints the recorded and new metrics side by side, so the training side can be checked on your machine. Tree ensembles are seeded (seed 42); small differences can still arise from library versions or thread scheduling.

