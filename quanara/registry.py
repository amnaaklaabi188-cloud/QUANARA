"""Model registry: artefacts on disk under artifacts/models/<model_id>/

  metadata.json            configuration, periods, metrics, feature ranges, versions
  model_final.joblib       pipeline refit on all observed training dates (used for forecasting)
  model_eval.joblib        pipeline fit before the holdout (produced the holdout predictions)
  predictions_holdout.csv  out-of-sample predictions (source of every displayed metric)
  predictions_cv.csv       expanding-window CV predictions
  predictions_test.csv     forecasts for the official test dates (no observed target exists)
  importance.csv           native and permutation importance
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import pandas as pd

from . import config


def model_dir(model_id: str) -> Path:
    return config.MODELS_DIR / model_id


def save(model_id: str, metadata: dict, final_model, eval_model, tables: dict[str, pd.DataFrame]) -> Path:
    d = model_dir(model_id)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, d / "model_final.joblib")
    if eval_model is not None:
        joblib.dump(eval_model, d / "model_eval.joblib")
    for name, df in tables.items():
        if df is not None:
            df.to_csv(d / f"{name}.csv", index=False)
    (d / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    return d


def list_models() -> list[dict]:
    if not config.MODELS_DIR.exists():
        return []
    out = []
    for p in sorted(config.MODELS_DIR.iterdir()):
        m = p / "metadata.json"
        if p.is_dir() and m.exists() and (p / "model_final.joblib").exists():
            try:
                out.append(json.loads(m.read_text()))
            except json.JSONDecodeError:
                continue
    return sorted(out, key=lambda r: r.get("created_at", ""), reverse=True)


def models_frame() -> pd.DataFrame:
    rows = []
    for m in list_models():
        h = m.get("metrics_holdout", {})
        rows.append({"model_id": m["model_id"], "label": m.get("label", ""), "scope": m["scope"], "mode": m["mode"],
                     "algorithm": m["algorithm_label"], "n_features": len(m["features"]),
                     "MAE": h.get("MAE"), "RMSE": h.get("RMSE"), "R2": h.get("R2"), "WMAPE": h.get("WMAPE"),
                     "created_at": m["created_at"], "reference": m.get("reference", False)})
    return pd.DataFrame(rows)


def load_metadata(model_id: str) -> dict:
    return json.loads((model_dir(model_id) / "metadata.json").read_text())


def load_model(model_id: str, which: str = "final"):
    return joblib.load(model_dir(model_id) / f"model_{which}.joblib")


def load_table(model_id: str, name: str) -> pd.DataFrame | None:
    p = model_dir(model_id) / f"{name}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    for c in ("val_start", "val_end"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def activate(model_id: str) -> None:
    md = load_metadata(model_id)
    if md["mode"] != "whatif":
        raise ValueError("Only What-if-mode models can drive the simulator (competition models need future hotel observations).")
    config.ACTIVE_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.ACTIVE_MODEL_FILE.write_text(json.dumps({"model_id": model_id}))


def active_model_id() -> str | None:
    if not config.ACTIVE_MODEL_FILE.exists():
        return None
    mid = json.loads(config.ACTIVE_MODEL_FILE.read_text()).get("model_id")
    return mid if mid and (model_dir(mid) / "model_final.joblib").exists() else None


def delete(model_id: str) -> None:
    """Remove a user-trained model. Reference models are protected so the Judges' Demo stays reproducible."""
    if (model_dir(model_id) / "metadata.json").exists() and load_metadata(model_id).get("reference"):
        raise PermissionError("Reference models cannot be deleted.")
    d = model_dir(model_id)
    if d.exists() and d.parent == config.MODELS_DIR:
        shutil.rmtree(d)
    if active_model_id() is None and config.ACTIVE_MODEL_FILE.exists():
        config.ACTIVE_MODEL_FILE.unlink()
