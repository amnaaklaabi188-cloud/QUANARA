"""Residual diagnostics and actual-versus-predicted figures for the What-if reference models and the active model.

    python evaluation/prediction_diagnostics.py

Residual = actual − predicted (positive = under-prediction). Writes results/model_benchmarks/residual_diagnostics.csv
and figures/model_evaluation/actual_vs_predicted__<model_id>.png.
"""
import numpy as np
import pandas as pd

from common import FIGURES, RESULTS
from plotstyle import CHARCOAL, EMERALD, plt
from quanara import registry
from quanara import model_identity as MI
from quanara.evaluation import daily_aggregate, diagnostics, metrics_from_predictions

if __name__ == "__main__":
    act = registry.active_model_id()
    pick = [m for m in registry.list_models() if (m.get("reference") and m["mode"] == "whatif") or m["model_id"] == act]
    rows = []
    for m in pick:
        mid = m["model_id"]
        hold = registry.load_table(mid, "predictions_holdout")
        mm, dg = metrics_from_predictions(hold), diagnostics(hold)
        rows.append({"model_id": mid, "model": MI.name(m), "active": mid == act, **{k: dg[k] for k in dg}, "MAE": mm["MAE"], "WMAPE": mm["WMAPE"], "R2": mm["R2"]})
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})
        lo, hi = float(min(hold["y_true"].min(), hold["y_pred"].min())), float(max(hold["y_true"].max(), hold["y_pred"].max()))
        a1.scatter(hold["y_true"], hold["y_pred"], s=7 if len(hold) > 1000 else 14, color=EMERALD, alpha=0.4, linewidths=0)
        a1.plot([lo, hi], [lo, hi], color=CHARCOAL, linestyle="--", linewidth=1.2)
        a1.set_xlabel("Actual Guests"); a1.set_ylabel("Predicted Guests"); a1.set_aspect("equal", adjustable="box")
        a1.set_title(f"R² {mm['R2']:.3f} · WMAPE {mm['WMAPE']:.1%} · n = {mm['n']:,}", loc="left", fontsize=10.5)
        d = daily_aggregate(hold) if "market_key" in hold else hold
        a2.plot(d["Date"], d["y_true"], color=CHARCOAL, linewidth=1.8, label="Actual")
        a2.plot(d["Date"], d["y_pred"], color=EMERALD, linewidth=1.8, label="Predicted")
        a2.set_ylabel("Guests per day" + (" (markets summed)" if "market_key" in hold else "")); a2.legend(frameon=False, loc="upper left")
        a2.set_title(f"{MI.name(m).replace(' · reference', '')} — holdout {m['periods']['holdout'][0]} → {m['periods']['holdout'][1]}", loc="left", fontsize=10.5)
        fig.autofmt_xdate(); fig.savefig(FIGURES / f"actual_vs_predicted__{mid}.png"); plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "residual_diagnostics.csv", index=False, float_format="%.6g")
    print(f"{len(df)} models diagnosed (reference What-if models and the active model) · wrote residual_diagnostics.csv and figures")
