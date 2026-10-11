"""Train the verified reference models used by the Judges' Demo (reproducible, seed 42).

Usage:  python scripts/train_reference_models.py [--fast]
Writes artifacts/models/*, artifacts/ablations/*, reports/reference_models.csv, activates the
demo model and saves the demo scenario.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from quanara import config, registry  # noqa: E402
from quanara import scenario as S  # noqa: E402
from quanara.features import default_features  # noqa: E402
from quanara.integration import get_analytic  # noqa: E402
from quanara.training import TrainConfig, run_ablation, train  # noqa: E402

# (scope, mode, algorithm, log_target, cv_folds, demo_active)
PLAN = [
    ("international_market", "whatif", "catboost", True, 3, True),
    ("international_market", "whatif", "seasonal_naive", False, 3, False),
    ("international_market", "whatif", "ridge", False, 3, False),
    ("international_market", "whatif", "random_forest", True, 3, False),
    ("international_market", "whatif", "xgboost", True, 3, False),
    ("international_market", "whatif", "svr", True, 2, False),
    ("international_total", "whatif", "seasonal_naive", False, 4, False),
    ("international_total", "whatif", "ridge", False, 4, False),
    ("international_total", "whatif", "random_forest", False, 4, False),
    ("international_total", "whatif", "catboost", False, 4, False),
    ("international_total", "whatif", "xgboost", False, 4, False),
    ("international_total", "whatif", "svr", False, 4, False),
    ("overall", "whatif", "seasonal_naive", False, 4, False),
    ("overall", "whatif", "ridge", False, 4, False),
    ("overall", "whatif", "random_forest", False, 4, False),
    ("overall", "whatif", "catboost", False, 4, False),
    ("domestic", "whatif", "seasonal_naive", False, 4, False),
    ("domestic", "whatif", "ridge", False, 4, False),
    ("domestic", "whatif", "catboost", False, 4, False),
    ("international_market", "competition", "catboost", True, 3, False),
    ("international_market", "competition", "xgboost", True, 3, False),
    ("international_total", "competition", "ridge", False, 4, False),
    ("international_total", "competition", "catboost", False, 4, False),
    ("domestic", "competition", "ridge", False, 4, False),
    ("domestic", "competition", "catboost", False, 4, False),
    ("overall", "competition", "ridge", False, 4, False),
    ("overall", "competition", "catboost", False, 4, False),
]

# Selection rule (documented): a top-3 guest market whose scenario inputs stay inside the training
# range for every model feature (no extrapolation), in the forecast period (no observed Guests).
DEMO_SCENARIO = dict(market="RUSSIAN FEDERATION", start="2025-09-01", end="2025-09-30", seats_change_pct=20.0)


def run_demo(a, demo_id: str, market_ids: list[str]) -> None:
    registry.activate(demo_id)
    print("activated demo model:", demo_id)
    res = S.run(S.ScenarioConfig(model_id=demo_id, note="Judges' Demo reference scenario", **DEMO_SCENARIO), a,
                compare_with=[m for m in market_ids if m != demo_id])
    demo = {"model_id": demo_id, "scenario_id": res.scenario_id, "scenario": DEMO_SCENARIO, "summary": res.summary,
            "comparison": res.comparison.to_dict(orient="records") if res.comparison is not None else None, "warnings": res.warnings}
    (config.ARTIFACTS_DIR / "judges_demo.json").write_text(json.dumps(demo, indent=2, default=str))
    print(S.judges_summary(res))
    if res.comparison is not None:
        print(res.comparison[["algorithm", "holdout_WMAPE", "baseline_avg_daily", "scenario_avg_daily", "pct_change"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip SVR on the market panel and the ablations")
    ap.add_argument("--demo-only", action="store_true", help="re-run only the demo scenario with the existing reference model")
    args = ap.parse_args()
    config.ensure_dirs()
    if args.demo_only:
        a = get_analytic()
        refs = [m for m in registry.list_models() if m.get("reference") and m["mode"] == "whatif" and m["scope"] == "international_market"]
        demo_id = next(m["model_id"] for m in refs if m["algorithm"] == "catboost")
        run_demo(a, demo_id, [m["model_id"] for m in refs])
        return
    a = get_analytic(rebuild=True)
    # remove previous reference models so the demo set is exactly reproducible
    for m in registry.list_models():
        if m.get("reference"):
            registry.delete(m["model_id"])
    out, demo_id, market_ids = [], None, []
    for scope, mode, algo, lt, folds, active in PLAN:
        if args.fast and algo == "svr" and scope == "international_market":
            continue
        t = time.perf_counter()
        cfg = TrainConfig(scope, mode, algo, default_features(scope, mode), log_target=lt, cv_folds=folds, reference=True)
        r = train(cfg, a)
        m = r.metadata["metrics_holdout"]
        cv = r.metadata["metrics_cv_mean"] or {}
        out.append({"model_id": r.model_id, "scope": scope, "mode": mode, "algorithm": algo, "log_target": lt, "monotone": r.metadata["monotone"],
                    "n_features": len(cfg.features), "holdout_MAE": m["MAE"], "holdout_RMSE": m["RMSE"], "holdout_R2": m["R2"],
                    "holdout_WMAPE": m["WMAPE"], "holdout_Pearson": m["Pearson"], "holdout_Spearman": m["Spearman"],
                    "cv_MAE": cv.get("MAE"), "cv_WMAPE": cv.get("WMAPE"), "seconds": round(time.perf_counter() - t, 1)})
        print(f"{r.model_id:58s} MAE={m['MAE']:9.1f} WMAPE={m['WMAPE']:.3f} R2={m['R2']:.3f} ({out[-1]['seconds']}s)", flush=True)
        if active:
            demo_id = r.model_id
        if scope == "international_market" and mode == "whatif":
            market_ids.append(r.model_id)
    summary = pd.DataFrame(out)
    summary.to_csv(config.REPORTS_DIR / "reference_models.csv", index=False)

    run_demo(a, demo_id, market_ids)

    if not args.fast:
        for scope, algo, lt in [("international_market", "catboost", True), ("international_total", "random_forest", False),
                                ("international_total", "ridge", False)]:
            cfg = TrainConfig(scope, "whatif", algo, default_features(scope, "whatif"), log_target=lt, reference=True)
            abl_id, s = run_ablation(cfg, a, tag=f"REF-ABL-{scope}-{algo}")
            print(abl_id)
            print(s[["feature_set", "n_features", "MAE", "RMSE", "R2", "WMAPE"]].to_string(index=False))


if __name__ == "__main__":
    main()
