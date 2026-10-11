# Correlation analysis

*Preliminary findings — ongoing investigation.*

Scripts: `analysis/pearson_correlation.py`, `analysis/spearman_correlation.py`, `analysis/correlation_heatmaps.py`, `analysis/scatter_plots.py`, `analysis/statistical_diagnostics.py`. Unit of observation: one calendar day; window 2023-03-01 → 2025-07-31 restricted to days with an observed target (n = 884 per scope). Tables for every eligible feature are in `results/correlations/` (valid n, Pearson and Spearman with p-values, effective n, differenced correlations, feature-to-feature matrices and pair lists).

## Two kinds of relationship

- **Feature → target** (a predictor against Guests): describes association with demand.
- **Feature ↔ feature** (two predictors): describes redundancy. A pair can be redundant because of an observed regularity (flights and seats) or because of how the quantities are defined (P2P share and transfer share).

## Feature → target, international daily total

| Feature | Pearson r | Spearman ρ | r on day-over-day changes | effective n |
|---|---|---|---|---|
| Guests same weekday last year, 7-day mean | 0.850 | 0.851 | 0.201 | ≈ 11 |
| Guests same weekday last year | 0.848 | 0.854 | 0.291 | ≈ 16 |
| Total P2P, trailing 7 days | 0.784 | 0.785 | 0.191 | ≈ 12 |
| Total PAX, trailing 7 days | 0.733 | 0.743 | 0.176 | ≈ 10 |
| Weekend indicator | −0.052 | −0.049 | — | — |

**Why the levels are strong but the changes are weak.** Arrivals and hotel guests both grew over 2023–2025 and both follow the seasons. Consecutive days are almost identical (lag-1 autocorrelation about 0.99 for P2P and 0.98 for Guests), so the 884 days carry roughly a dozen independent pieces of information for this pair, and classical p-values (around 10⁻¹⁸⁴) are far too optimistic. Correlations of day-over-day changes remove most of the shared trend and season and fall to about 0.2.

**Lag analysis.** For the international total, the correlation between 7-day P2P arrivals and Guests is highest at lag 0 (r = 0.784) and declines slowly with lag; neighbouring lags are highly correlated, so the peak is not a travel-time estimate (`results/correlations/international_total_lag_p2p_7d.csv`).

**Domestic guests** relate weakly or negatively to inbound aviation (largest What-if correlation: last-year guests, r = 0.372); the negative correlations with P2P reflect opposite seasonal patterns, not a mechanism.

## Feature ↔ feature

| Pair | Pearson r | Interpretation |
|---|---|---|
| Flights vs seats | 0.995 (Spearman 0.994) | Observed: the average aircraft size barely changes (220 seats per flight, coefficient of variation 1.4%). Flights come from Average Weekly Frequency, seats from Total Seats. The two inputs overlap heavily; correlation alone does not show that one adds nothing — the ablation study and permutation importance measure that. |
| P2P share vs transfer share | −1.000 | Accounting identity: PAX = P2P + transfer + transit on every row, so transfer share = 1 − P2P share − transit share. Transit is non-zero on 3 daily route rows, so the identity is broken on 7 of 884 days, by at most 0.0002. Using both repeats the same quantity; the recommended feature sets include P2P share only. |

The pair lists flag every predictor pair with |r| ≥ 0.95 (19 Pearson pairs for the international total, 18 for domestic and overall). They are flagged, never removed automatically.

## Interpretation rules used throughout

1. Correlation describes co-movement in historical data, not causation.
2. Strong correlation between two predictors is redundancy, not evidence that one is useless.
3. Associations driven by shared trend or season are distinguished from day-to-day co-movement by differencing.
4. Significance is read with the effective sample size, not the raw number of days.
