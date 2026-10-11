# Models

Every algorithm that has at least one saved model has its own folder with `train.py`, `predict.py`, `evaluate.py`, `config.json` (features, hyperparameters, dates, target, preprocessing, artefact paths, verified metrics) and a README.

| Algorithm | Folder | Saved models | Reference models |
|---|---|---|---|
| Seasonal naive baseline | `models/seasonal_naive/` | 5 | 4 |
| Random Forest | `models/random_forest/` | 8 | 3 |
| Ridge Regression | `models/ridge/` | 8 | 7 |
| CatBoost | `models/catboost/` | 8 | 8 |
| XGBoost | `models/xgboost/` | 3 | 3 |
| Support Vector Regression | `models/support_vector_regression/` | 2 | 2 |

Extra Trees is implemented in `quanara/models.py` and offered in the ML Training Studio, but no Extra Trees model has been saved, so it has no folder.

## Prediction scopes

| Scope | Target | Grain |
|---|---|---|
| `international_total` | International hotel guests | Daily total |
| `international_market` | International hotel guests, by nationality | Daily, per nationality |
| `domestic` | Domestic hotel guests | Daily total |
| `overall` | All hotel guests (international + domestic) | Daily total |

Modes: **What-if** models use aviation, calendar and last-year guests and can drive the simulator. **Competition** models may also use same-day hotel arrivals (New Arrivals, Same-Day Guests) present in the official test files; they score the test period but cannot run scenarios.

## Artefacts

The trained pipelines (`artifacts/models/<model_id>/model_final.joblib`, `model_eval.joblib`) and their prediction files are not part of this package: they are derived from the restricted competition data. `models/model_registry.csv` and each `config.json` keep every configuration and verified metric. Rebuild the reference models with `python scripts/train_reference_models.py` after placing the data (see `data/README.md`).

