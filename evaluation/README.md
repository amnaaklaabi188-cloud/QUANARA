# Evaluation

Saved-prediction metric recomputation, like-for-like comparison and diagnostics.

- [`common.py`](common.py) — Shared helpers for the evaluation scripts (path setup, output folders).
- [`compare_models.py`](compare_models.py) — Rank models only within a comparison group (same target, grain and evaluation rows), What-if and Competition apart.
- [`evaluate_models.py`](evaluate_models.py) — Recompute the holdout metrics of every saved model from its own saved predictions.
- [`prediction_diagnostics.py`](prediction_diagnostics.py) — Residual diagnostics and actual-versus-predicted figures for the What-if reference models and the active model.
