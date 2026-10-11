# Model training and benchmarking

## Algorithms

| Algorithm | Implementation (`quanara/models.py`) | Saved models |
|---|---|---|
| Seasonal naive baseline | `SeasonalNaive`: Guests on the same weekday 364 days earlier, with fallbacks | 5 |
| Ridge Regression | median imputation + indicators → standardisation → one-hot market → `Ridge` | 8 |
| Random Forest | median imputation + indicators → one-hot market → `RandomForestRegressor` | 8 |
| CatBoost | native categorical market, native missing handling → `CatBoostRegressor` (RMSE) | 8 |
| XGBoost | median imputation + indicators → one-hot market → `XGBRegressor(hist)` | 3 |
| Support Vector Regression | imputation → standardisation → one-hot → `SVR(rbf)` with standardised target | 2 |

`models/model_registry.csv` lists all 34 saved models; each algorithm folder documents its own pipeline, hyperparameters and models. Extra Trees is implemented but no Extra Trees model has been saved.

Options recorded with every model: log-transformed target (`log1p`/`expm1`, metrics always in guests), scenario-safe monotone constraint for tree models (predictions cannot fall when seats, passengers or flights rise, holding everything else fixed), negative-prediction policy (`none` or `floor_zero`), seed 42.

## Dataset preparation

`quanara.integration.model_frame(scope, mode)` builds one row per date (per date × nationality for the market scope) with the target `y` and every feature; `training_rows` keeps the modelling window 2023-03-01 → 2025-07-31 where the target is observed. Competition-mode models that use New Arrivals keep only nationality-days with New Arrivals ≥ 10, matching the official test extract. The training matrix of any configuration can be written with `features/build_features.py`.

## Feature selection

Each scope and mode has a recommended feature set (`quanara.features.default_features`), visible on the Feature Engineering page and in `models/<algorithm>/config.json`. Redundant pairs (e.g. P2P share and transfer share, which mirror each other by construction) are not used together. The Feature ablation study retrains the same algorithm with feature families removed or isolated on the identical holdout (`artifacts/ablations/`, summarised in `reports/reference_training_log.txt`).

## Hyperparameters

Defaults and ranges are in `quanara.models.HYPERPARAMS` and in each `models/<algorithm>/README.md`. The reference models use the defaults. **No tuning was done on the holdout**: tuning by watching holdout scores would leak it; settings are compared on the cross-validation mean.

## Validation

1. Expanding-window cross-validation folds (3 for the market panel, 4 otherwise, 91 days each) inside the fitting period.
2. Chronological holdout 2025-02-01 → 2025-07-31 (181 days), never used for fitting or selection.
3. Deployment refit on all 884 modelling days.

The holdout predictions, CV predictions, test-period forecasts and permutation importance are saved with each model; every displayed metric is recomputed from them.

## Metrics

- **MAE** — mean absolute error, in the target's unit (guests per day, or per nationality-day).
- **RMSE** — root mean squared error; penalises large misses.
- **WMAPE** — Σ|actual − predicted| ÷ Σ actual; scale-free, so it is the main comparison metric.
- **R²** — share of variance explained on the holdout; can be negative; not an accuracy percentage.
- Pearson and Spearman correlation between actual and predicted values.

## Comparison rules

Models are ranked only when they predict the same target at the same grain for exactly the same evaluation rows (`quanara.evaluation.comparison_key`), and What-if models are never ranked against Competition models, which see same-day hotel arrivals. `evaluation/compare_models.py` applies these rules; results are in `results/model_benchmarks/ranking_by_group.csv`.

## Results (What-if models, holdout 2025-02-01 → 2025-07-31)

| Target | Model | MAE | RMSE | R² | WMAPE |
|---|---|---|---|---|---|
| By nationality | CatBoost (reference) | 160.1 | 317.1 | 0.879 | 22.6% |
| By nationality | XGBoost (reference) | 177.7 | 341.0 | 0.860 | 25.1% |
| By nationality | Support Vector Regression (reference, **active in the Simulator**) | 183.2 | 365.1 | 0.840 | 25.9% |
| By nationality | Seasonal naive baseline | 192.7 | 387.1 | 0.820 | 27.2% |
| By nationality | Random Forest (reference) | 193.3 | 425.9 | 0.782 | 27.3% |
| By nationality | Ridge (reference; not simulator-ready: negative predictions) | 206.8 | 359.9 | 0.844 | 29.2% |
| International total | Random Forest | 1,662.3 | 2,077.3 | 0.706 | 6.7% |
| International total | Random Forest (reference) | 1,807.6 | 2,244.8 | 0.657 | 7.2% |
| International total | XGBoost (reference) | 2,084.7 | 2,534.9 | 0.563 | 8.4% |
| International total | CatBoost (reference) | 2,109.0 | 2,796.7 | 0.467 | 8.5% |
| International total | Support Vector Regression (reference) | 2,469.0 | 2,954.5 | 0.406 | 9.9% |
| International total | Ridge (reference) | 2,712.7 | 3,329.1 | 0.245 | 10.9% |
| International total | Seasonal naive baseline | 3,091.9 | 3,825.3 | 0.004 | 12.4% |
| All guests | Ridge (reference) | 3,451.0 | 4,831.6 | 0.294 | 8.4% |
| Domestic | Ridge (reference) | 2,143.5 | 3,311.4 | 0.290 | 13.5% |
| Domestic | Seasonal naive baseline | 2,628.9 | 4,295.1 | −0.195 | 16.5% |
| Domestic | CatBoost (reference) | 2,822.8 | 4,022.6 | −0.048 | 17.8% |

Readings: for nationality-level demand the learned models improve on the seasonal baseline by a few WMAPE points; for the international daily total the improvement is larger (12.4% → 6.7–7.2%). For domestic guests, the CatBoost model does not beat the seasonal baseline, which is consistent with domestic stays having no direct inbound-flight mechanism. The best R² is not automatically the best model for every use — bias, peak behaviour and simulator eligibility matter too.

## Re-running a model

```bash
.venv/bin/python models/<algorithm>/train.py --from <model_id> --dry-run    # print the recorded configuration
.venv/bin/python models/<algorithm>/train.py --from <model_id> --no-save    # refit in memory, compare metrics
.venv/bin/python scripts/train_reference_models.py                         # rebuild all reference models
```

Re-fitting was not re-run when this package was assembled; the saved pipelines' predictions are verified by `tests/test_legacy_artifacts.py`.
