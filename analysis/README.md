# Analysis

Observed-data correlation analysis and scientific plots.

- [`__init__.py`](__init__.py) — Python package marker.
- [`correlation_heatmaps.py`](correlation_heatmaps.py) — Correlation heatmaps (Pearson and Spearman) for each prediction scope.
- [`correlations.py`](correlations.py) — Shared correlation analysis used by the scripts in this folder.
- [`pearson_correlation.py`](pearson_correlation.py) — Pearson correlations for every eligible feature, per prediction scope.
- [`plotstyle.py`](plotstyle.py) — Matplotlib style shared by the figure scripts (QuanAra emerald, charcoal and grey).
- [`scatter_plots.py`](scatter_plots.py) — Scatter plots of Guests against the most strongly correlated scenario-ready features, per scope.
- [`spearman_correlation.py`](spearman_correlation.py) — Spearman rank correlations for every eligible feature, per prediction scope.
- [`statistical_diagnostics.py`](statistical_diagnostics.py) — Diagnostics that decide how far the correlations can be trusted.
