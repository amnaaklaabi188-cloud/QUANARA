"""Scatter plots of Guests against the most strongly correlated scenario-ready features, per scope.

    python analysis/scatter_plots.py [--top 4]

Each point is one observed day; the line is the ordinary least-squares fit with its 95% confidence band for the
fitted mean. Writes figures/scatter_plots/<scope>__<feature>.png.
"""
import argparse

import numpy as np

from correlations import FIGURES, LEVELS, feature_to_target, frame
from plotstyle import CHARCOAL, EMERALD, plt
from quanara import correlation as C

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=4)
    args = ap.parse_args()
    for scope, lv in LEVELS.items():
        df = frame(scope)
        obs = df[df["Guests"].notna()]
        ft = feature_to_target(df, scope)
        pick = ft[ft["whatif_eligible"] & (ft["group"] != "time")].head(args.top)["feature"].tolist()
        for f in pick:
            s = C.pair_stats(obs[f], obs["Guests"])
            ok = obs[f].notna()
            x, y = obs.loc[ok, f].to_numpy(float), obs.loc[ok, "Guests"].to_numpy(float)
            xs = np.linspace(x.min(), x.max(), 100)
            lo, hi = C.mean_ci_band(s, xs)
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            ax.scatter(x, y, s=12, color=EMERALD, alpha=0.5, linewidths=0)
            ax.fill_between(xs, lo, hi, color=CHARCOAL, alpha=0.12, linewidth=0)
            ax.plot(xs, s["intercept"] + s["slope"] * xs, color=CHARCOAL, linewidth=2)
            ax.set_xlabel(C.label(f)); ax.set_ylabel(f"Guests per day ({lv.lower()})")
            ax.set_title(f"{C.label(f)} vs Guests — {lv}", loc="left")
            ax.text(0.01, 0.98, f"Pearson r = {s['pearson']:.3f} · Spearman ρ = {s['spearman']:.3f} · n = {s['n']} days (effective n ≈ {s['n_eff']:.0f})",
                    transform=ax.transAxes, va="top", fontsize=9, color=CHARCOAL)
            p = FIGURES / "scatter_plots" / f"{scope}__{f}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p); plt.close(fig)
        print(f"{lv}: scatter plots for {', '.join(pick)}")
