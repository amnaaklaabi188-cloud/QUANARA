"""Train, predict and evaluate one algorithm — the shared engine behind models/<algorithm>/{train,predict,evaluate}.py.

Every step calls the same functions as the Streamlit application (quanara.training, quanara.registry,
quanara.scenario, quanara.evaluation), so a model produced here is identical in kind to one trained in
the ML Training Studio.

  train     re-run the configuration recorded in a saved model's metadata (--from MODEL_ID), or a
            configuration from models/<algorithm>/config.json (--config-index N). --dry-run prints the exact
            TrainConfig without fitting; --no-save fits in memory and compares with the recorded metrics.
  predict   load a saved deployment pipeline and predict Guests for any supported date range.
  evaluate  recompute every holdout metric from the saved predictions and run the artifact integrity checks.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quanara import config, registry  # noqa: E402
from quanara import model_identity as MI  # noqa: E402

FOLDER = {"catboost": "catboost", "random_forest": "random_forest", "xgboost": "xgboost", "ridge": "ridge",
          "svr": "support_vector_regression", "seasonal_naive": "seasonal_naive", "extra_trees": "extra_trees"}


def _models(algorithm: str) -> list[dict]:
    return [m for m in registry.list_models() if m["algorithm"] == algorithm]


def _train_config(md_or_cfg: dict):
    from quanara.training import TrainConfig
    c = md_or_cfg.get("config", md_or_cfg)
    return TrainConfig(scope=c["scope"], mode=c["mode"], algorithm=c["algorithm"], features=list(c["features"]), params=dict(c.get("params", {})),
                       log_target=bool(c.get("log_target", False)), monotone=bool(c.get("monotone", True)),
                       holdout_start=c.get("holdout_start", str(config.DEFAULT_HOLDOUT_START.date())), cv_folds=int(c.get("cv_folds", 4)),
                       cv_horizon_days=int(c.get("cv_horizon_days", 91)), label=c.get("label", ""), reference=False,
                       seed=int(c.get("seed", config.RANDOM_STATE)), negative_policy=c.get("negative_policy", "none"))


def cmd_train(algorithm: str, args) -> int:
    from quanara.integration import get_analytic
    from quanara.training import train
    if args.source:
        md = registry.load_metadata(args.source)
        if md["algorithm"] != algorithm:
            print(f"{args.source} is a {md['algorithm']} model, not {algorithm}.")
            return 2
        cfg, recorded = _train_config(md), md["metrics_holdout"]
    else:
        cfgs = json.loads((ROOT / "models" / FOLDER[algorithm] / "config.json").read_text())["models"]
        entry = cfgs[args.config_index]
        cfg, recorded = _train_config(entry["training_config"]), entry["holdout_metrics"]
    print("TrainConfig:\n" + json.dumps(cfg.__dict__, indent=2, default=str))
    if args.dry_run:
        print("\n--dry-run: nothing was fitted.")
        return 0
    res = train(cfg, get_analytic(), lambda f, m: print(f"  [{f:4.0%}] {m}"), save=not args.no_save)
    new = res.metadata["metrics_holdout"]
    print(f"\n{'metric':8s} {'recorded':>14s} {'this run':>14s}")
    for k in ("MAE", "RMSE", "R2", "WMAPE"):
        print(f"{k:8s} {recorded[k]:14.6f} {new[k]:14.6f}")
    same = all(np.isclose(recorded[k], new[k], rtol=1e-6, atol=1e-9) for k in ("MAE", "RMSE", "R2", "WMAPE"))
    print("\nReproduced the recorded holdout metrics." if same else
          "\nThe metrics differ from the recorded run (library versions, hardware threads or data copies can cause this).")
    if not args.no_save:
        print(f"Saved as {res.model_id} in artifacts/models/.")
    return 0


def cmd_predict(algorithm: str, args) -> int:
    from quanara import scenario as S
    from quanara.integration import get_analytic
    if args.model_id:
        mid = args.model_id
    else:  # the active model if it is this algorithm, else this algorithm's reference What-if model (nationality-level when --market is given)
        cands = [m for m in _models(algorithm) if m["mode"] == "whatif"]
        act = registry.active_model_id()
        pref = [m for m in cands if m["model_id"] == act] + [m for m in cands if m.get("reference") and (m["scope"] == "international_market") == bool(args.market)] \
            + [m for m in cands if m.get("reference")]
        mid = pref[0]["model_id"] if pref else None
    if mid is None:
        print("No saved model for this algorithm; train one first.")
        return 2
    md = registry.load_metadata(mid)
    if md["mode"] != "whatif":
        print("Competition models need same-day hotel observations; predict with a What-if model.")
        return 2
    a = get_analytic()
    dates = pd.date_range(args.start, args.end, freq="D")
    markets = md.get("markets") if md["scope"] == "international_market" else None
    frame = S._frames(a, md["scope"], a.flight_daily_rows, dates, markets)
    pred = frame[["Date"] + (["market_key"] if markets else [])].copy()
    pred["y_pred"] = S._predict(mid, frame)
    if args.market and markets:
        pred = pred[pred["market_key"] == args.market]
    elif args.market:
        print(f"Note: {mid} predicts {MI.target(md).lower()} (not by nationality); --market is ignored.")
    out = Path(args.out or ROOT / "results" / "predictions" / f"{mid}_{args.start}_{args.end}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out, index=False)
    daily = pred.groupby("Date")["y_pred"].sum()
    print(f"{MI.name(md)} ({mid})\n{len(pred):,} predictions · {MI.target(md)} · mean {daily.mean():,.1f} guests per day over {len(daily)} days\nwrote {out}")
    return 0


def cmd_evaluate(algorithm: str, args) -> int:
    from quanara.evaluation import diagnostics, metrics_from_predictions, verify_artifacts
    ids = [args.model_id] if args.model_id else [m["model_id"] for m in _models(algorithm)]
    if not ids:
        print("No saved model for this algorithm.")
        return 2
    rows = []
    for mid in ids:
        md = registry.load_metadata(mid)
        hold = registry.load_table(mid, "predictions_holdout")
        m, dg = metrics_from_predictions(hold), diagnostics(hold)
        checks = verify_artifacts(md, hold, registry.load_table(mid, "predictions_test"), None)
        rows.append({"model_id": mid, "target": MI.target(md, short=True), "mode": MI.mode(md), "holdout": " → ".join(md["periods"]["holdout"]),
                     "n": m["n"], "MAE": round(m["MAE"], 2), "RMSE": round(m["RMSE"], 2), "R2": round(m["R2"], 4), "WMAPE": round(m["WMAPE"], 4),
                     "mean_residual": round(dg["mean_residual"], 2), "checks_passed": f"{sum(c['ok'] for c in checks)}/{len(checks)}"})
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.to_string(index=False))
    print("\nMetrics are recomputed from predictions_holdout.csv; compare only rows with the same target and holdout.")
    return 0


def main(algorithm: str, action: str) -> int:
    p = argparse.ArgumentParser(description=f"{MI.ALGORITHMS.get(algorithm, algorithm)} — {action}")
    if action == "train":
        g = p.add_mutually_exclusive_group()
        g.add_argument("--from", dest="source", help="re-run the configuration of this saved model ID")
        g.add_argument("--config-index", type=int, default=0, help="entry of models/<algorithm>/config.json (default 0)")
        p.add_argument("--dry-run", action="store_true", help="print the configuration only")
        p.add_argument("--no-save", action="store_true", help="fit in memory and compare with the recorded metrics; save nothing")
        return cmd_train(algorithm, p.parse_args())
    if action == "predict":
        p.add_argument("--model-id", help="saved What-if model (default: this algorithm's reference model)")
        p.add_argument("--start", default="2025-08-01")
        p.add_argument("--end", default="2026-02-28")
        p.add_argument("--market", help="keep one source market (market-level models only), e.g. 'RUSSIAN FEDERATION'")
        p.add_argument("--out", help="output CSV (default results/predictions/…)")
        return cmd_predict(algorithm, p.parse_args())
    p.add_argument("--model-id", help="one saved model (default: every saved model of this algorithm)")
    return cmd_evaluate(algorithm, p.parse_args())
