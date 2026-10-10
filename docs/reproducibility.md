# Reproducibility and private inputs

Python 3.12 was used. From the existing checkout (cloud tasks are already isolated; no extra Git worktree is needed):

```bash
cd /workspace/QUANARA
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/audit_inputs.py --inputs /path/to/authorized-inputs --output private-results
```

Supply exactly one of each of the five original filenames: `data domestic_train.xlsx`, `data domestic_test.xlsx`, `data international_train.xlsx`, `data international_test.xlsx`, and `flight_data.xlsx`, with sheet `Export`. Place `Data_Dictionary.pdf` alongside them for provenance hashing. Inputs may reside in separate subdirectories; each workbook filename must appear once. The audit reads inputs and does not modify or copy them. Additional workbooks or unexpected sheet lists are rejected. Output should be outside the input directory.

Generated files under the output root:

- `docs/results/dataset_inventory.csv`: workbook/PDF provenance and structure.
- `docs/results/column_quality.csv`: stored types, missingness and suppression.
- `docs/results/dataset_coverage.csv`: monthly coverage and absent hotel keys.
- `docs/results/quality_checks.csv`: consistency flags.
- `docs/results/feature_correlations.csv`: eligible observed-data correlations.
- `docs/results/feature_inventory.csv` and `model_benchmarking.csv`: explicit header-only unavailable model records.
`scripts/verify_outputs.py` independently verifies all original hashes, dimensions and each published correlation against pandas computations. Run it with the same `--inputs` and `--output` arguments.

- `docs/figures/`: observed trends, count/capacity correlation heatmaps, contemporaneous scatter plots, distributions and missing-data summary.

In this task the authorized originals remain under `/workspace/attachments`; locally generated results are under `/workspace/quanara-analysis`. They are not included in the source commit pending redistribution approval. Do not place credentials, originals or private models in Git. Ignore rules are an additional safeguard, not a substitute for inspecting staged files.

There is no supplied Streamlit entry point, app dependency specification, saved model, copilot or Judges' Demo. Application startup, model evaluation, simulator operation and screenshot reproduction are blocked on the recovery materials listed in [recovery status](recovery-status.md). No command is offered that pretends these components exist.
