# Methodology

QuanAra answers one question with six supplied files: *how might we build a tool that predicts how flight changes affect hotel visits in Abu Dhabi?* This document describes what the code does, step by step. Numbers quoted here are produced by the scripts in this repository.

## 1. Data sources

Five workbooks and `Data_Dictionary.pdf` (see `data/README.md`). The hotel files are daily: international guests per nationality per day and domestic guests per day, with train files to 2025-07-31 and official test files (no Guests) from 2025-08-01 to 2026-02-28. The flight file has one row per route (departure city × airline) — monthly for 2022 and daily from 2023-01-01. No external data, calendars or APIs are used, and the originals are never edited: the app reads byte-identical copies verified by SHA-256.

Where the dictionary and the delivered workbooks disagree, the workbooks are followed; the ten reconciliations are listed on the Data Intelligence page and in `reports/audit_report.md`.

## 2. Preparation (`quanara/data_loading.py`, `cleaning.py`, `integration.py`)

- Blank cells (the dictionary's “*” suppression marker) stay missing and are never read as zero. One explicit exception: summed Same-Day Guests count blanks as 0 and are labelled a lower bound.
- Daily flights are decoded exactly from Average Weekly Frequency (AWF × days in month ÷ 7).
- The 1,213 monthly 2022 flight rows are never spread over days; they appear only in monthly views.
- Seats, passengers and flights are summed per day (or per day × departure country) on a complete calendar; a day without a row is a genuine zero. Load factor and shares are recomputed from those sums, never averaged.
- Load factors above 100% are kept as recorded; Total Transit (non-zero on 3 daily route rows) and cabin-level counts that do not add up to the totals are not used as features.
- **Source markets** pair a guest nationality with the departure country of the same name: 33 of 45 nationalities, 92.2% of international guest-days. This is a proxy, not passenger attribution.
- Guests is a stock (guests in-house each day) fed by arrivals over several days, which is why trailing 7- and 28-day windows are used.

## 3. Features (`quanara/features.py`)

See `docs/feature_engineering.md` for every feature. In short: aviation volumes (same day, trailing 7 and 28 days, previous week), derived shares, seats per flight, active routes/cities/airlines and 28-day growth, calendar terms (day of week, weekend, month, quarter, seasonal sine/cosine), and Guests on the same weekday 364 days earlier and its 7-day mean. Same-day hotel observations (New Arrivals, Same-Day Guests) are restricted to Competition mode.

## 4. Prediction scopes and modes

| Scope | Target | Grain |
|---|---|---|
| `international_market` | International hotel guests, by nationality | daily × nationality (33 matched markets) |
| `international_total` | International hotel guests | daily total (45 nationalities) |
| `domestic` | Domestic hotel guests | daily total |
| `overall` | All hotel guests | daily total |

**What-if mode** uses only inputs that exist on a scenario date (aviation, calendar, last-year guests). **Competition mode** may add same-day hotel arrivals, which the official test files contain; such models score the test period but cannot drive scenarios.

## 5. Training and validation (`quanara/training.py`)

- Modelling window 2023-03-01 → 2025-07-31 (884 days, after a 56-day warm-up so every trailing feature is defined).
- **Chronological holdout:** fit on 2023-03-01 → 2025-01-31, score on 2025-02-01 → 2025-07-31 (181 days) — dates the model never saw. No random splits.
- **Expanding-window cross-validation** before the holdout (3–4 folds × 91 days); each fold trains on all earlier dates only.
- Imputation and scaling are fitted inside the training rows of each fold.
- **Deployment model:** refitted with identical settings on all modelling days, used for test-period forecasts and scenarios.
- Metrics (MAE, RMSE, R², WMAPE, Pearson, Spearman) are recomputed from saved predictions whenever they are shown; models are ranked only within the same target, grain and evaluation rows.

## 6. Simulation (`quanara/scenario.py`)

Seat capacity, flight frequency (aircraft size absorbs the difference, so seats are never double-counted), load factor and P2P share can be changed for one departure country or all, sustained across the 56-day look-back. Recomputed quantities: PAX = load factor × seats; P2P = P2P share × PAX; transit scales with PAX; transfer = PAX − P2P − transit. Calendar terms, last-year guests, route network and cabin mix are held fixed. Features are rebuilt with the training code and the saved deployment model predicts baseline and scenario on identical dates; a zero change reproduces the baseline. Inputs outside the training range are flagged.

## 7. Verification

- 118 automated tests (`tests/`), including leakage tests, artifact reproduction and scenario identities.
- `scripts/verify_pages.py` recomputes every figure on the Overview and Data Intelligence pages directly from the raw workbooks (125/125 checks).
- `features/feature_validation.py`, `evaluation/evaluate_models.py` and `simulator/scenario_validation.py` re-run the key checks from the command line.
