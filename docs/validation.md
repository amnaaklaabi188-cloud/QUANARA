# Validation report

## Executed

- Inspected both referenced repositories and their available histories; neither contained application source or model artifacts.
- Fetched and fast-forwarded the selected checkout to include the owner's README and image commits; preserved the legacy image.
- Inspected the actual image; it says QUANTERRA, so official QuanAra banner validation is blocked.
- Created a fresh Python 3.12 virtual environment and installed all pinned analysis dependencies with normal TLS verification.
- Executed five numerical unit tests: suppressed/blank values stay missing; correlations use pairwise-valid counts; insufficient/constant inputs are excluded; Spearman ranks differ from nonlinear Pearson association; aviation column selection excludes country labels and ratios. All five passed.
- Ran the offline audit against all five supplied workbooks and hashed all six originals. Generated observed-data results locally; no inputs were modified.

- Independently recomputed all 124 reported Pearson/Spearman rows with pandas, including valid and missing pair counts; all matched. Verified all six original hashes and five workbook dimensions.
- Ran the audit more than once against unchanged originals; tabular correlation results remained reproducible.
- Checked all repository Markdown links and Git whitespace checks.

## Blocked rather than passed

All ten application pages, model loading, original feature calculations, model metrics, residual/prediction plots, simulator zero-change behavior, a saved-model scenario example, live copilot, Judges' Demo and application screenshots require unavailable source/artifacts. The official banner requires the correct image. No fabricated metrics, model rankings or model-based plots fill these gaps.

## Publication boundary

Source and documentation may be published under the owner's explicit instruction. Competition inputs, extracted dictionary text, numeric correlation results and data-derived figures remain local pending explicit redistribution approval. No dataset, private artifact, credential or log belongs in the source commit. Final staged-file inspection and remote push are checked separately; a local commit alone does not establish GitHub publication.
