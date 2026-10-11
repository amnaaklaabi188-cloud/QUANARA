"""Correlation heatmaps (Pearson and Spearman) for each prediction scope.

    python analysis/correlation_heatmaps.py

Writes figures/correlations/heatmap_<scope>_<method>.png for the ten headline variables shown in the
Correlation Lab, and heatmap_<scope>_<method>_all.png for every eligible feature.
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from correlations import FIGURES, LEVELS, frame, features_of
from plotstyle import plt
from quanara import correlation as C

HEADLINE = ["Guests", "p2p_7d", "seats_7d", "transfer_7d", "flights_7d", "load_factor_7d", "p2p_share_7d", "guests_lag364_7d", "doy_cos", "is_weekend"]
CMAP = LinearSegmentedColormap.from_list("qa", ["#44403c", "#a8a29e", "#f5f5f4", "#6ee7b7", "#047857"])


def heatmap(m, labels, path, title, annotate):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * n + 2), max(5, 0.5 * n + 1.5)))
    im = ax.imshow(m, cmap=CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=8.5 if n > 14 else 9.5)
    ax.set_yticks(range(n), labels, fontsize=8.5 if n > 14 else 9.5)
    ax.grid(False)
    if annotate:
        for i in range(n):
            for j in range(n):
                if np.isfinite(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#101614")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    ax.set_title(title, loc="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    for scope, lv in LEVELS.items():
        df = frame(scope)
        obs = df[df["Guests"].notna()]
        for method in ("pearson", "spearman"):
            cols = [c for c in HEADLINE if c in obs]
            m = obs[cols].astype(float).corr(method=method).to_numpy()
            heatmap(m, [C.short_label(c) for c in cols], FIGURES / "correlations" / f"heatmap_{scope}_{method}.png",
                    f"{lv} — {method.title()} correlation, {len(obs)} observed days", True)
            allc = ["Guests"] + features_of(df)
            m2 = obs[allc].astype(float).corr(method=method).to_numpy()
            heatmap(m2, [C.short_label(c) for c in allc], FIGURES / "correlations" / f"heatmap_{scope}_{method}_all.png",
                    f"{lv} — {method.title()} correlation, all eligible features", False)
        print(f"{lv}: wrote heatmaps for Pearson and Spearman")
