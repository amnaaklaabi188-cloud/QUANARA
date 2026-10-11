"""Pearson correlations for every eligible feature, per prediction scope.

    python analysis/pearson_correlation.py

Writes results/correlations/<scope>_feature_to_target_pearson.csv (with p-value, valid n, effective n and the
day-over-day Pearson), <scope>_feature_matrix_pearson.csv and <scope>_feature_pairs_pearson.csv.
p-values assume independent days; consecutive days are strongly autocorrelated, so read them with the effective n.
"""
from correlations import run_method

if __name__ == "__main__":
    run_method("pearson")
