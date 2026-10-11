# Seasonal naive baseline

Benchmark: Guests on the same weekday last year. Uses no aviation input, so it cannot respond to a scenario.

## Pipeline

`SeasonalNaive`: no learned parameters except fallbacks. ŷ(t) = Guests(t − 364 days) — the same weekday last year — for the same nationality (market scope) or the same total; missing values fall back to last year's 7-day mean, then the training median of the market (or overall). Uses no aviation input, so it cannot respond to a scenario; it is the benchmark every model must beat.

Every saved artefact is the complete pipeline (preprocessing + estimator), so predicting needs only the feature columns listed in `config.json`.

## Hyperparameters

| Name | Meaning | Default | Range | Note |
|---|---|---|---|---|
| — | No hyperparameters | | | |

## Saved models (5)

Metrics are on the chronological holdout recorded with each model and are recomputed from its saved predictions by `evaluate.py`. Compare only models with the same target and holdout (see `models/model_registry.csv`).

| Model ID | Target | Mode | Features | Holdout | MAE | RMSE | R² | WMAPE | Simulator |
|---|---|---|---|---|---|---|---|---|---|
| `INTL-WI-seasonal_naive-20261010-215221-2da9d9` | Intl guests | What-if | 0 | 2025-02-01 → 2025-07-31 | 3,091.9 | 3,825.3 | 0.004 | 12.4% | no |
| `DOM-WI-seasonal_naive-20261010-124010-554899` (reference) | Domestic guests | What-if | 17 | 2025-02-01 → 2025-07-31 | 2,629.0 | 4,295.1 | -0.195 | 16.5% | no |
| `ALL-WI-seasonal_naive-20261010-123947-5d88af` (reference) | All guests | What-if | 17 | 2025-02-01 → 2025-07-31 | 4,329.0 | 6,209.0 | -0.166 | 10.6% | no |
| `INTL-WI-seasonal_naive-20261010-123918-4187e6` (reference) | Intl guests | What-if | 17 | 2025-02-01 → 2025-07-31 | 3,091.9 | 3,825.3 | 0.004 | 12.4% | no |
| `MKT-WI-seasonal_naive-20261010-123104-d90910` (reference) | Intl guests by nationality | What-if | 18 | 2025-02-01 → 2025-07-31 | 192.7 | 387.1 | 0.820 | 27.2% | no |

Training, validation and deployment dates, the full feature list, preprocessing options and software versions of each model are in `config.json`.

## Run

```bash
# recompute the metrics of every saved model of this algorithm from its saved predictions
python models/seasonal_naive/evaluate.py
# predict the official test period with this algorithm's reference What-if model (writes results/predictions/)
python models/seasonal_naive/predict.py --start 2025-08-01 --end 2026-02-28
# print the exact training configuration of a saved model without fitting anything
python models/seasonal_naive/train.py --from INTL-WI-seasonal_naive-20261010-215221-2da9d9 --dry-run
# refit that configuration in memory and compare with the recorded metrics (saves nothing)
python models/seasonal_naive/train.py --from INTL-WI-seasonal_naive-20261010-215221-2da9d9 --no-save
# refit and register it as a new model in artifacts/models/
python models/seasonal_naive/train.py --from INTL-WI-seasonal_naive-20261010-215221-2da9d9
```

## Reproducibility status

The prediction side is verified: `tests/test_legacy_artifacts.py` reloads every saved pipeline and reproduces its saved holdout predictions. Re-fitting was **not** re-run when this package was assembled (no model was retrained). `train.py --from <model_id> --no-save` refits the recorded configuration in memory and prints the recorded and new metrics side by side, so the training side can be checked on your machine. Tree ensembles are seeded (seed 42); small differences can still arise from library versions or thread scheduling.

