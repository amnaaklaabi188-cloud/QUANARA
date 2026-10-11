# Data

QuanAra uses only the six files supplied for the DCT Abu Dhabi ChallengeON 2026. **They are not included in this package.** Place your authorised copies in `data/raw/` with their original names (or set `QUANARA_DATA_DIR` to the folder that holds them). The application reads them read-only and never modifies them.

| Dataset | File | Rows | Dates | Schema |
|---|---|---|---|---|
| International hotel guests — train | `data international_train.xlsx` | 58,622 | 2022-01-01 → 2025-07-31 | `schemas/intl_train.csv` |
| International hotel guests — test | `data international_test.xlsx` | 9,202 | 2025-08-01 → 2026-02-28 | `schemas/intl_test.csv` |
| Domestic hotel guests — train | `data domestic_train.xlsx` | 1,308 | 2022-01-01 → 2025-07-31 | `schemas/dom_train.csv` |
| Domestic hotel guests — test | `data domestic_test.xlsx` | 212 | 2025-08-01 → 2026-02-28 | `schemas/dom_test.csv` |
| Flight operations | `flight_data.xlsx` | 117,608 | 2022-01-01 → 2026-02-28 | `schemas/flight.csv` |
| Data dictionary | `Data_Dictionary.pdf` | — | — | quoted in the Data Intelligence page |

## Check your copies

`SHA256SUMS.txt` lists the checksums of the files this project was built and verified with:

```bash
cp data/SHA256SUMS.txt data/raw/ && cd data/raw && shasum -a 256 -c SHA256SUMS.txt
```

## Derived folders (created at run time, git-ignored)

- `data/processed/` — model matrices written by `features/build_features.py` and `scripts/build_dataset.py`
- `data/cache/` — cached analytical dataset
- `artifacts/` — trained pipelines, predictions and saved scenarios

