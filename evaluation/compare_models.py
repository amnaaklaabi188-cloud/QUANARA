"""Rank models only within a comparison group (same target, grain and evaluation rows), What-if and Competition apart.

    python evaluation/compare_models.py        (run evaluate_models.py first)

Writes results/model_benchmarks/ranking_by_group.csv and figures/model_evaluation/benchmark_<scope>.png for the
What-if models of each prediction scope. Competition models also see same-day hotel arrivals, so they are ranked
separately and never against What-if forecasts.
"""
import pandas as pd

from common import FIGURES, RESULTS
from plotstyle import EMERALD, GREY, plt

if __name__ == "__main__":
    df = pd.read_csv(RESULTS / "all_models.csv")
    df["rank_WMAPE"] = df.groupby(["comparison_group", "mode"])["WMAPE"].rank(method="min").astype(int)
    out = df.sort_values(["comparison_group", "mode", "rank_WMAPE"])[["comparison_group", "target", "grain", "mode", "holdout_start", "holdout_end",
                                                                      "rank_WMAPE", "model", "model_id", "MAE", "RMSE", "R2", "WMAPE", "simulator_ready"]]
    out.to_csv(RESULTS / "ranking_by_group.csv", index=False, float_format="%.6g")
    for (g, target), sub in df[df["mode"] == "What-if"].groupby(["comparison_group", "target"]):
        sub = sub.sort_values("WMAPE", ascending=False)
        fig, ax = plt.subplots(figsize=(7.6, 0.5 * len(sub) + 1.4))
        best = sub["WMAPE"].min()
        ax.barh(sub["model"].str.replace(" · reference", ""), sub["WMAPE"] * 100, color=[EMERALD if v == best else GREY for v in sub["WMAPE"]], height=0.6)
        for i, (v, r2) in enumerate(zip(sub["WMAPE"], sub["R2"])):
            ax.text(v * 100 + 0.3, i, f"{v:.1%} · R² {r2:.3f}", va="center", fontsize=9)
        ax.set_xlabel("Holdout WMAPE (%) — lower is better"); ax.grid(axis="y", visible=False)
        ax.set_title(f"{target} — What-if models, holdout {sub['holdout_start'].iloc[0]} → {sub['holdout_end'].iloc[0]}", loc="left", fontsize=11)
        ax.set_xlim(0, sub["WMAPE"].max() * 100 * 1.35)
        scope = target.lower().replace(" ", "_").replace(",", "").replace("(", "").replace(")", "").replace("+", "and")
        fig.savefig(FIGURES / f"benchmark_{g:02d}_{scope}.png"); plt.close(fig)
        print(f"group {g} · {target}: best What-if WMAPE {best:.1%} ({sub.loc[sub['WMAPE'].idxmin(), 'model']})")
    print("wrote results/model_benchmarks/ranking_by_group.csv")
