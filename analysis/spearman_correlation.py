"""Spearman rank correlations for every eligible feature, per prediction scope.

    python analysis/spearman_correlation.py

Writes results/correlations/<scope>_feature_to_target_spearman.csv, <scope>_feature_matrix_spearman.csv and
<scope>_feature_pairs_spearman.csv. Rank correlation is robust to outliers and to monotone non-linear relationships.
"""
from correlations import run_method

if __name__ == "__main__":
    run_method("spearman")
