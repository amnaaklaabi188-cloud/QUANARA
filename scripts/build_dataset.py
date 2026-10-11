"""Milestone 2: build and validate the analytical datasets (daily grain).

Usage:  python scripts/build_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from quanara import config  # noqa: E402
from quanara.data_loading import load_all  # noqa: E402
from quanara.integration import build_analytic, model_frame, save_analytic, training_rows  # noqa: E402


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 120)
    a = build_analytic(load_all())
    save_analytic(a)
    print(a.log.frame().to_string(index=False))
    print()
    rows = []
    for scope in config.SCOPES:
        for mode in config.MODES:
            f = model_frame(a, scope, mode)
            tr = training_rows(f)
            te = f[f["split"] == "test"]
            feats = [c for c in f.columns if c not in ("Date", "market_key", "matched", "split", "y")]
            rows.append({"scope": scope, "mode": mode, "frame_rows": len(f), "train_rows": len(tr),
                         "train_period": f"{tr['Date'].min().date()} → {tr['Date'].max().date()}",
                         "test_rows": len(te), "n_features": len(feats),
                         "max_feature_missing_%_train": round(float(tr[feats].isna().mean().max() * 100), 2),
                         "target_mean": round(float(tr["y"].mean()), 1)})
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(config.REPORTS_DIR / "model_frames_summary.csv", index=False)
    a.log.frame().to_csv(config.REPORTS_DIR / "dataset_build_log.csv", index=False)


if __name__ == "__main__":
    main()
