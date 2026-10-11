"""Recompute the holdout metrics of every saved model from its own saved predictions.

    python evaluation/evaluate_models.py

Writes results/model_benchmarks/all_models.csv: target, grain, mode, holdout window, rows, MAE, RMSE, R², WMAPE,
Pearson, Spearman, whether the recomputed metrics equal the saved metadata, the artifact integrity checks, the
comparison group (same target, grain and evaluation rows) and simulator readiness.
"""
import numpy as np
import pandas as pd

from common import RESULTS
from quanara import registry
from quanara import model_identity as MI
from quanara.evaluation import comparison_key, metrics_from_predictions, verify_artifacts

if __name__ == "__main__":
    models = registry.list_models()
    keys = {m["model_id"]: comparison_key(m, registry.load_table(m["model_id"], "predictions_holdout")) for m in models}
    gid = {k: i + 1 for i, k in enumerate(dict.fromkeys(keys.values()))}
    labels = MI.labels(models)
    rows = []
    for m in models:
        mid = m["model_id"]
        hold = registry.load_table(mid, "predictions_holdout")
        mm = metrics_from_predictions(hold)
        saved = m["metrics_holdout"]
        checks = verify_artifacts(m, hold, registry.load_table(mid, "predictions_test"), None)
        rows.append({"model_id": mid, "model": labels[mid], "algorithm": MI.algorithm(m), "target": MI.target(m), "grain": MI.scope(m), "mode": MI.mode(m),
                     "holdout_start": m["periods"]["holdout"][0], "holdout_end": m["periods"]["holdout"][1], "holdout_rows": mm["n"],
                     "MAE": mm["MAE"], "RMSE": mm["RMSE"], "R2": mm["R2"], "WMAPE": mm["WMAPE"], "Pearson": mm["Pearson"], "Spearman": mm["Spearman"],
                     "matches_saved_metadata": all(np.isclose(mm[k], saved[k], rtol=1e-6, atol=1e-9) for k in ("MAE", "RMSE", "R2", "WMAPE")),  # CSV precision
                     "integrity_checks": f"{sum(c['ok'] for c in checks)}/{len(checks)}", "comparison_group": gid[keys[mid]],
                     "reference_model": bool(m.get("reference")), "simulator_ready": MI.scenario_ready(m), "not_ready_reason": MI.not_ready_reason(m) or ""})
    df = pd.DataFrame(rows).sort_values(["comparison_group", "mode", "WMAPE"])
    p = RESULTS / "all_models.csv"
    df.to_csv(p, index=False, float_format="%.6g")
    print(f"{len(df)} saved models · {df['matches_saved_metadata'].sum()} recomputed metrics equal their metadata · "
          f"{(df['integrity_checks'].str.split('/').str[0] == df['integrity_checks'].str.split('/').str[1]).sum()} pass every integrity check · "
          f"{df['comparison_group'].nunique()} comparison groups\nwrote {p}")
