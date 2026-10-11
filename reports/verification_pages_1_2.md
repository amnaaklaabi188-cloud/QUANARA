# Verification — Pages 1 and 2
_2026-10-11T02:02:54 · active model `MKT-WI-svr-20261010-123918-b5964e` · **125/125 checks passed**_

| page | check | displayed | independent | status |
|---|---|---|---|---|
| 1+2 | Flight rows | 117608 | 117608 | PASS |
| 1+2 | Flight first date | 2022-01-01 | 2022-01-01 | PASS |
| 1+2 | Flight last date | 2026-02-28 | 2026-02-28 | PASS |
| 1+2 | Flight monthly-grain rows (2022) | 1213 | 1213 | PASS |
| 1+2 | Flight daily-grain rows | 116395 | 116395 | PASS |
| 1 | First daily-grain date | 2023-01-01 | 2023-01-01 | PASS |
| 1+2 | Departure countries (all rows) | 33 | 33 | PASS |
| 1+2 | Airlines (all rows) | 59 | 59 | PASS |
| 1+2 | Σ monthly PAX in sparkline = Σ Total PAX | 40279493.0 | 40279493 | PASS |
| 2 | Daily-row P2P passengers | 17507078.0 | 17507078.0 | PASS |
| 2 | Daily-row Transfer passengers | 17069665.0 | 17069665.0 | PASS |
| 2 | Daily-row Transit passengers | 164.0 | 164.0 | PASS |
| 2 | P2P share of daily PAX (shown 1 dp) | 50.6 | 50.6 | PASS |
| 2 | Transfer share of daily PAX (shown 1 dp) | 49.4 | 49.4 | PASS |
| 2 | Daily rows with Transit > 0 | 3 | 3 | PASS |
| 2 | PAX = P2P + Transfer + Transit on every row | True | True | PASS |
| 2 | Daily rows with load factor > 100% (shown 1 dp) | 8.5 | 8.5 | PASS |
| 2 | Flight card: monthly rows in text | True | True | PASS |
| 1+2 | International train rows | 58622 | 58622 | PASS |
| 1+2 | International nationalities | 45 | 45 | PASS |
| 1 | Days with observed Guests | 1308 | 1308 | PASS |
| 1+2 | Domestic train days | 1308 | 1308 | PASS |
| 1 | Labelled records (intl + domestic) | 59930 | 59930 | PASS |
| 1 | Guests observed from | 2022-01-01 | 2022-01-01 | PASS |
| 1 | Guests observed to | 2025-07-31 | 2025-07-31 | PASS |
| 1 | Unlabelled test rows (intl + domestic) | 9414 | 9414 | PASS |
| 1 | Test period start | 2025-08-01 | 2025-08-01 | PASS |
| 1 | Test period end | 2026-02-28 | 2026-02-28 | PASS |
| 1+2 | Test files contain a Guests column | False | False | PASS |
| 1 | Average international Guests per day (shown 0 dp) | 19441 | 19441 | PASS |
| 1 | Average domestic Guests per day (shown 0 dp) | 17737 | 17737 | PASS |
| 2 | intl_train: rows | 58,622 | 58,622 | PASS |
| 2 | intl_train: columns in text | True | True | PASS |
| 2 | intl_train: distinct days in text | True | True | PASS |
| 2 | intl_test: rows | 9,202 | 9,202 | PASS |
| 2 | intl_test: columns in text | True | True | PASS |
| 2 | intl_test: distinct days in text | True | True | PASS |
| 2 | dom_train: rows | 1,308 | 1,308 | PASS |
| 2 | dom_train: columns in text | True | True | PASS |
| 2 | dom_train: distinct days in text | True | True | PASS |
| 2 | dom_test: rows | 212 | 212 | PASS |
| 2 | dom_test: columns in text | True | True | PASS |
| 2 | dom_test: distinct days in text | True | True | PASS |
| 2 | Duplicate keys across the five files | 0 | 0 | PASS |
| 2 | Literal '*' cells across the five files | 0 | 0 | PASS |
| 2 | Missing nationality-day combinations | 238 | 238 | PASS |
| 2 | Nationalities with missing days | 8 | 8 | PASS |
| 2 | Same-Day Guests suppressed, intl train (shown 1 dp) | 59.7 | 59.7 | PASS |
| 2 | Same-Day Guests suppressed, intl test (shown 1 dp) | 47.8 | 47.8 | PASS |
| 2 | Train rows with New Arrivals < 10 | 5202 | 5202 | PASS |
| 2 | Test rows with New Arrivals < 10 | 0 | 0 | PASS |
| 1+2 | Nationalities matching a departure country | 33 | 33 | PASS |
| 2 | Guest-days in matched nationalities (shown 1 dp) | 92.2 | 92.2 | PASS |
| 2 | Rows checked for duplicate keys (all five files) | 186952 | 186952 | PASS |
| 2 | Duplicate keys (quality card) | 0 | 0 | PASS |
| 2 | Training rows with Guests: international | 58622 | 58622 | PASS |
| 2 | Training rows with Guests: domestic | 1308 | 1308 | PASS |
| 2 | Finding evidence — Nationalities with a matching departure country | 33 | 33 | PASS |
| 2 | Finding evidence — International guest-days in matched nationalities | 0.9218 | 0.9218 | PASS |
| 2 | Finding evidence — Arrivals from the United Kingdom that are transfer passengers | 0.7219 | 0.7219 | PASS |
| 2 | Finding evidence — Russian nationals: share of guest-days vs. share of arriving passengers | [0.1007, 0.024] | [0.1007, 0.024] | PASS |
| 2 | Finding evidence — Monthly-grain rows (2022) | 1213 | 1213 | PASS |
| 2 | Finding evidence — Daily-grain rows | 116395 | 116395 | PASS |
| 2 | Finding evidence — Average Weekly Frequency empty: monthly vs daily rows | [1213, 0] | [1213, 0] | PASS |
| 2 | Finding evidence — Training rows with New Arrivals below 10 | 5202 | 5202 | PASS |
| 2 | Finding evidence — Test rows with New Arrivals below 10 | 0 | 0 | PASS |
| 2 | Finding evidence — Share of training guest-days in those rows | 0.0087 | 0.0087 | PASS |
| 2 | Finding evidence — Nationality rows per day: training vs test | [44.8, 43.4] | [44.8, 43.4] | PASS |
| 2 | Finding evidence — Unobserved nationality-day combinations | 238 | 238 | PASS |
| 2 | Finding evidence — Most affected | {'FINLAND': 54, 'NORWAY': 44, 'AZERBAIJAN': 42, 'DENMARK': 42, 'CZECHIA': 27, 'ARMENIA': 24, 'ISRAEL': 4, 'QATAR': 1} | {'FINLAND': 54, 'NORWAY': 44, 'AZERBAIJAN': 42, 'DENMARK': 42, 'CZECHIA': 27, 'ARMENIA': 24, 'ISRAEL': 4, 'QATAR': 1} | PASS |
| 2 | Finding evidence — Smallest recorded international Guests value | 10 | 10 | PASS |
| 2 | Finding evidence — Blank Same-Day Guests, international training | 35026 | 35026 | PASS |
| 2 | Finding evidence — Blank Same-Day Guests, international test | 4402 | 4402 | PASS |
| 2 | Finding evidence — Literal “*” cells across the five files | 0 | 0 | PASS |
| 2 | Finding evidence — Daily rows above 100% | 9883 | 9883 | PASS |
| 2 | Finding evidence — Of these, still above 100% without infants | 2245 | 2245 | PASS |
| 2 | Finding evidence — 2022 monthly rows above 100% | 13 | 13 | PASS |
| 2 | Finding evidence — Largest gap between Load Factor and PAX ÷ Seats | 0.0 | 0.0 | PASS |
| 2 | Finding evidence — Unlabelled test rows (international + domestic) | 9414 | 9414 | PASS |
| 2 | Finding evidence — Guests column present in a test file | False | False | PASS |
| 2 | Finding evidence — Daily rows with Total Transit > 0 | 3 | 3 | PASS |
| 2 | Finding evidence — Transit passengers vs all arriving passengers | 164.0 | 164.0 | PASS |
| 2 | Finding evidence — Daily rows with cabin transit counts > 0 | 4207 | 4207 | PASS |
| 2 | Finding evidence — Rows where cabin transfer + transit ≤ Total Transfer | 1.0 | 1.0 | PASS |
| 2 | Finding evidence — Daily rows where cabin P2P = Total P2P | 33020 | 33020 | PASS |
| 2 | Finding evidence — Rows where cabin P2P + infants = Total P2P | 0.8667 | 0.8667 | PASS |
| 2 | Smallest recorded international Guests value | 10 | 10 | PASS |
| 2 | Empty New Arrivals cells, international training | 262 | 262 | PASS |
| 2 | Daily rows with Total Transit = 0 (genuine zeros) | 116392 | 116392 | PASS |
| 2 | Empty Guests in training files | 0 | 0 | PASS |
| 2 | Empty passenger/seat totals on daily rows | 0 | 0 | PASS |
| 2 | International test columns (no Guests) | Date, New Arrivals, Same-Day Guests, Nationality, Residence (groups) | Date, New Arrivals, Same-Day Guests, Nationality, Residence (groups) | PASS |
| 2 | Domestic test columns (no Guests) | Date, New Arrivals, Same-Day Guests, Residence (groups) | Date, New Arrivals, Same-Day Guests, Residence (groups) | PASS |
| 2 | Domestic training Guests minimum | 8827 | 8827 | PASS |
| 2 | Domestic training Guests maximum | 36385 | 36385 | PASS |
| 2 | Distinct column names across the five workbooks | 31 | 31 | PASS |
| 2 | Largest load factor (2022 monthly rows) | 1.857143 | 1.857143 | PASS |
| 2 | International test days | 212 | 212 | PASS |
| 2 | Dates stored as Excel date values (not strings) | True | True | PASS |
| 2 | Dictionary wording present in PDF — Target in the test files | True | True | PASS |
| 2 | Dictionary wording present in PDF — Flight-data grain | True | True | PASS |
| 2 | Dictionary wording present in PDF — Average Weekly Frequency | True | True | PASS |
| 2 | Dictionary wording present in PDF — Suppression marker | True | True | PASS |
| 2 | Dictionary wording present in PDF — Load Factor format | True | True | PASS |
| 2 | Dictionary wording present in PDF — Date format | True | True | PASS |
| 2 | Dictionary wording present in PDF — International test coverage | True | True | PASS |
| 2 | Dictionary wording present in PDF — Domestic Guests range | True | True | PASS |
| 2 | Dictionary wording present in PDF — Cabin-class columns | True | True | PASS |
| 2 | Dictionary wording present in PDF — Number of columns | True | True | PASS |
| 1 | Correlation: intl Guests vs trailing-7-day P2P (shown 2 dp) | 0.78 | 0.78 | PASS |
| 1 | Dictionary: Guests = guests recorded for nationality on date (intl) | True | True | PASS |
| 1 | Dictionary: New Arrivals are a subset of Guests (continuing stays excluded) | True | True | PASS |
| 1 | Dictionary: domestic Guests is a daily total | True | True | PASS |
| 1 | Active model ID resolved | MKT-WI-svr-20261010-123918-b5964e | MKT-WI-svr-20261010-123918-b5964e | PASS |
| 1 | Model metadata ID = folder ID | MKT-WI-svr-20261010-123918-b5964e | MKT-WI-svr-20261010-123918-b5964e | PASS |
| 1 | Holdout R² (shown 3 dp) | 0.84 | 0.84 | PASS |
| 1 | Holdout MAE (shown 0 dp) | 183 | 183 | PASS |
| 1 | Holdout WMAPE (shown 1 dp) | 25.9 | 25.9 | PASS |
| 1 | Holdout RMSE (metadata vs recomputed) | 365.096591 | 365.096591 | PASS |
| 1 | Holdout predictions count | 5973 | 5973 | PASS |
| 1 | Holdout start | 2025-02-01 | 2025-02-01 | PASS |
| 1 | Holdout end | 2025-07-31 | 2025-07-31 | PASS |
| 1 | Training (evaluation fit) ends the day before the holdout | 2025-01-31 | 2025-01-31 | PASS |
| 1 | Training starts at the modelling start | 2023-03-01 | 2023-03-01 | PASS |
| 1 | Every saved y_true equals the raw Guests value | 5973/5973 | 5973/5973 | PASS |
