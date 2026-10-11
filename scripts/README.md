# Scripts

Dataset, audit, reference-training, verification and documentation workflows.

- [`__init__.py`](__init__.py) — Python package marker.
- [`build_brand_assets.py`](build_brand_assets.py) — Write the QuanAra brand assets (sidebar logo, collapsed-sidebar mark, favicon) from quanara/ui/brand.py.
- [`build_dataset.py`](build_dataset.py) — Milestone 2: build and validate the analytical datasets (daily grain).
- [`build_research_docs.py`](build_research_docs.py) — Regenerate the documentation that describes features, models and data schemas from the code itself.
- [`package_text.py`](package_text.py) — Fixed descriptions used by scripts/build_research_docs.py — each one restates the code in quanara/models.py.
- [`run_audit.py`](run_audit.py) — Milestone 1: audit the five workbooks against Data_Dictionary.pdf.
- [`train_reference_models.py`](train_reference_models.py) — Train the verified reference models used by the Judges' Demo (reproducible, seed 42).
- [`verify_pages.py`](verify_pages.py) — Independent verification of every figure shown on Page 1 (Welcome & Overview) and Page 2 (Data Intelligence).
