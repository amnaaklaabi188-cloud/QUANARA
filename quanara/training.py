"""Training with chronological validation.

Protocol (no random splits anywhere):
  1. Modelling rows: observed Guests, 2023-03-01 … 2025-07-31.
  2. Chronological holdout: dates >= holdout_start (default 2025-02-01, 181 days) are never
     seen while fitting the evaluation model. Holdout predictions are saved and are the only
     source of the reported out-of-sample metrics.
  3. Expanding-window CV inside the pre-holdout period: k consecutive validation blocks of
     h days; each fold trains on every date before its block.
  4. The deployment model is refit with identical settings on all modelling rows (through
     2025-07-31) and is used for forecasting and the What-if simulator.
Preprocessing (imputation, scaling, encoding) lives inside the pipeline, so it is fitted on
training rows only in every fold.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

from . import config, registry
from .evaluation import cv_summary, daily_aggregate, metrics
from .features import CATALOG_BY_NAME, available_features
from .feature_facts import matrix_columns, matrix_fingerprint
from .integration import Analytic, model_frame, training_rows
from .models import ALGORITHMS, NonNegative, default_params, fresh, make_model, monotone_features, native_importance

ProgressFn = Callable[[float, str], None]


@dataclass
class TrainConfig:
    scope: str
    mode: str
    algorithm: str
    features: list[str]
    params: dict = field(default_factory=dict)
    log_target: bool = False
    monotone: bool = True  # shape constraint for tree models (see features.MONOTONE_INCREASING)
    holdout_start: str = str(config.DEFAULT_HOLDOUT_START.date())
    cv_folds: int = 4
    cv_horizon_days: int = 91
    permutation_importance: bool = True
    label: str = ""
    reference: bool = False
    seed: int = config.RANDOM_STATE
    negative_policy: str = "none"   # "none" (raw predictions) | "floor_zero" (documented floor at 0)

    def validate(self) -> None:
        if self.scope not in config.SCOPES:
            raise ValueError(f"unknown scope {self.scope}")
        if self.mode not in config.MODES:
            raise ValueError(f"unknown mode {self.mode}")
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"unknown algorithm {self.algorithm}")
        allowed = set(available_features(self.scope, self.mode))
        bad = [f for f in self.features if f not in allowed]
        if bad:
            raise ValueError(f"features not allowed for scope={self.scope}, mode={self.mode}: {bad}")
        if self.algorithm != "seasonal_naive" and not self.features:
            raise ValueError("select at least one feature")
        hs = pd.Timestamp(self.holdout_start)
        if not (config.MODEL_START + pd.Timedelta(days=270) <= hs <= config.MODEL_END - pd.Timedelta(days=28)):
            raise ValueError("holdout start must leave ≥ 270 training days and ≥ 28 holdout days")


@dataclass
class TrainResult:
    model_id: str
    metadata: dict
    holdout: pd.DataFrame
    cv: pd.DataFrame
    test: pd.DataFrame
    importance: pd.DataFrame


def _model_id(cfg: TrainConfig) -> str:
    h = hashlib.sha1(json.dumps(asdict(cfg), sort_keys=True, default=str).encode()).hexdigest()[:6]
    short = {"international_total": "INTL", "international_market": "MKT", "domestic": "DOM", "overall": "ALL"}[cfg.scope]
    return f"{short}-{'WI' if cfg.mode == 'whatif' else 'COMP'}-{cfg.algorithm}-{datetime.now():%Y%m%d-%H%M%S}-{h}"


def _keys(scope: str) -> list[str]:
    return ["Date", "market_key"] if scope == "international_market" else ["Date"]


def _needed_columns(cfg: TrainConfig) -> list[str]:
    cols = list(cfg.features)
    if cfg.algorithm == "seasonal_naive":
        cols += ["guests_lag364", "guests_lag364_7d"] + (["market"] if cfg.scope == "international_market" else [])
    return list(dict.fromkeys(cols))


def data_fingerprint() -> dict:
    """SHA-256 of the six source files (from data/raw/SHA256SUMS.txt) and a digest of the set."""
    f = config.RAW_DIR / "SHA256SUMS.txt"
    if not f.exists():
        return {}
    pairs = dict(reversed(ln.split("  ", 1)) for ln in f.read_text().splitlines() if ln.strip())
    return {"files": pairs, "digest": hashlib.sha256(f.read_bytes()).hexdigest()[:16]}


def cv_folds(dates: pd.Series, holdout_start: pd.Timestamp, k: int, h: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    folds = []
    for j in range(k):
        vs = holdout_start - pd.Timedelta(days=(k - j) * h)
        ve = vs + pd.Timedelta(days=h - 1)
        if (vs - dates.min()).days >= 180:
            folds.append((vs, ve))
    return folds


def permutation_importance(model, X: pd.DataFrame, y: np.ndarray, features: list[str], seed: int, n_repeats: int = 3,
                           max_rows: int = 4000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    base = np.sqrt(np.mean((y - model.predict(X)) ** 2))
    out = {}
    for f in features:
        incs = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            incs.append(np.sqrt(np.mean((y - model.predict(Xp)) ** 2)) - base)
        out[f] = float(np.mean(incs))
    return out


def feature_ranges(X: pd.DataFrame, features: list[str], scope: str) -> dict:
    num = [f for f in features if CATALOG_BY_NAME[f].kind == "numeric"]
    rng = {"global": {f: [float(np.nanmin(X[f])), float(np.nanmax(X[f]))] for f in num if X[f].notna().any()}}
    if scope == "international_market" and "market" in X:
        rng["by_market"] = {m: {f: [float(np.nanmin(g[f])), float(np.nanmax(g[f]))] for f in num if g[f].notna().any()}
                            for m, g in X.groupby("market")}
    return rng


def train(cfg: TrainConfig, a: Analytic, progress: ProgressFn | None = None, save: bool = True) -> TrainResult:
    cfg.validate()
    say = progress or (lambda f, m: None)
    t0 = time.perf_counter()
    by_market = cfg.scope == "international_market"
    params = {**default_params(cfg.algorithm), **cfg.params}
    keys = _keys(cfg.scope)

    say(0.02, "Building model frame from the analytical dataset")
    frame = model_frame(a, cfg.scope, cfg.mode)
    rows = training_rows(frame)
    cols = _needed_columns(cfg)
    hs = pd.Timestamp(cfg.holdout_start)
    fit_rows, hold_rows = rows[rows["Date"] < hs].reset_index(drop=True), rows[rows["Date"] >= hs].reset_index(drop=True)

    def build():
        return make_model(cfg.algorithm, cfg.features, params, cfg.log_target, by_market, cfg.seed, monotone=cfg.monotone,
                          negative_policy=cfg.negative_policy)

    # ---- expanding-window CV
    folds = cv_folds(fit_rows["Date"], hs, cfg.cv_folds, cfg.cv_horizon_days) if cfg.cv_folds > 0 else []
    cv_parts, fold_details = [], []
    for i, (vs, ve) in enumerate(folds):
        say(0.05 + 0.55 * i / max(len(folds), 1), f"CV fold {i + 1}/{len(folds)}: train < {vs.date()}, validate {vs.date()} → {ve.date()}")
        tr = fit_rows[fit_rows["Date"] < vs]
        va = fit_rows[(fit_rows["Date"] >= vs) & (fit_rows["Date"] <= ve)]
        m = fresh(build()).fit(tr[cols], tr["y"].to_numpy())
        cv_parts.append(va[keys].assign(fold=i + 1, y_true=va["y"].to_numpy(), y_pred=m.predict(va[cols])))
        fold_details.append({"fold": i + 1, "train_start": str(tr["Date"].min().date()), "train_end": str(tr["Date"].max().date()),
                             "val_start": str(vs.date()), "val_end": str(ve.date()), "train_rows": int(len(tr)), "val_rows": int(len(va)),
                             "train_dates": int(tr["Date"].nunique()), "val_dates": int(va["Date"].nunique())})
    cv_pred = pd.concat(cv_parts, ignore_index=True) if cv_parts else pd.DataFrame(columns=keys + ["fold", "y_true", "y_pred"])

    # ---- chronological holdout
    say(0.62, f"Fitting evaluation model on {fit_rows['Date'].min().date()} → {fit_rows['Date'].max().date()} ({len(fit_rows):,} rows)")
    eval_model = fresh(build()).fit(fit_rows[cols], fit_rows["y"].to_numpy())
    say(0.72, f"Predicting chronological holdout {hold_rows['Date'].min().date()} → {hold_rows['Date'].max().date()} ({len(hold_rows):,} rows)")
    hold_pred = hold_rows[keys].assign(y_true=hold_rows["y"].to_numpy(), y_pred=eval_model.predict(hold_rows[cols]))
    m_hold = metrics(hold_pred["y_true"], hold_pred["y_pred"])
    m_hold_daily = metrics(*daily_aggregate(hold_pred)[["y_true", "y_pred"]].to_numpy().T) if by_market else None
    rel = (hold_pred["y_true"] - hold_pred["y_pred"]).abs() / hold_pred["y_pred"].abs().clip(lower=1)
    abs_res = (hold_pred["y_true"] - hold_pred["y_pred"]).abs()

    # ---- importance
    say(0.78, "Computing feature importance")
    nat = native_importance(eval_model, cfg.features) if cfg.algorithm != "seasonal_naive" else None
    perm = permutation_importance(eval_model, hold_rows[cols].reset_index(drop=True), hold_rows["y"].to_numpy(), cfg.features, cfg.seed) \
        if (cfg.permutation_importance and cfg.algorithm != "seasonal_naive") else None
    imp = pd.DataFrame({"feature": cfg.features})
    imp["label"] = imp["feature"].map(lambda f: CATALOG_BY_NAME[f].label)
    imp["group"] = imp["feature"].map(lambda f: CATALOG_BY_NAME[f].group)
    imp["native"] = imp["feature"].map(nat) if nat else np.nan
    imp["permutation_rmse_increase"] = imp["feature"].map(perm) if perm else np.nan

    # ---- deployment model
    say(0.86, f"Refitting deployment model on all {len(rows):,} modelling rows (through {rows['Date'].max().date()})")
    final_model = fresh(build()).fit(rows[cols], rows["y"].to_numpy())

    test_rows = frame[frame["split"] == "test"].reset_index(drop=True)
    test_pred = test_rows[keys].assign(y_pred=final_model.predict(test_rows[cols])) if len(test_rows) else pd.DataFrame()

    def raw(model, X):
        return model.predict_raw(X) if isinstance(model, NonNegative) else model.predict(X)
    neg = {"holdout": int((raw(eval_model, hold_rows[cols]) < 0).sum()),
           "test": int((raw(final_model, test_rows[cols]) < 0).sum()) if len(test_rows) else 0}

    runtime = time.perf_counter() - t0
    model_id = _model_id(cfg)
    import sklearn
    md = {
        "model_id": model_id,
        "label": cfg.label or f"{ALGORITHMS[cfg.algorithm]} · {config.SCOPES[cfg.scope].split(' (')[0]}",
        "reference": cfg.reference,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": cfg.scope, "scope_label": config.SCOPES[cfg.scope],
        "mode": cfg.mode, "mode_label": config.MODES[cfg.mode],
        "target": "Guests", "grain": "daily × nationality" if by_market else "daily",
        "algorithm": cfg.algorithm, "algorithm_label": ALGORITHMS[cfg.algorithm],
        "features": cfg.features, "params": params, "log_target": cfg.log_target,
        "monotone": bool(cfg.monotone and cfg.algorithm in ("random_forest", "extra_trees", "catboost", "xgboost")),
        "markets": sorted(rows["market_key"].unique()) if by_market else None,
        "periods": {
            "modelling": [str(rows["Date"].min().date()), str(rows["Date"].max().date())],
            "eval_fit": [str(fit_rows["Date"].min().date()), str(fit_rows["Date"].max().date())],
            "holdout": [str(hold_rows["Date"].min().date()), str(hold_rows["Date"].max().date())],
            "deployment_fit": [str(rows["Date"].min().date()), str(rows["Date"].max().date())],
            "cv_folds": [[str(a_.date()), str(b_.date())] for a_, b_ in folds],
        },
        "n_rows": {"modelling": len(rows), "eval_fit": len(fit_rows), "holdout": len(hold_rows), "test": len(test_rows)},
        "metrics_holdout": m_hold,
        "metrics_holdout_daily_total": m_hold_daily,
        "metrics_cv_mean": cv_summary(cv_pred).drop(columns=["fold", "val_start", "val_end"]).mean().to_dict() if len(cv_pred) else None,
        "interval": {"method": "empirical holdout residual quantiles (90%)", "rel_q90": float(np.quantile(rel, 0.9)),
                     "abs_q90": float(np.quantile(abs_res, 0.9))},
        "feature_ranges": feature_ranges(rows, cfg.features, cfg.scope),
        "matrix_fingerprint": matrix_fingerprint(rows[matrix_columns(cfg.scope, cfg.features)]),
        "data_fingerprint": data_fingerprint(),
        "target_transform": "log1p on fit, expm1 on predict (metrics in guests)" if cfg.log_target else "none",
        "negative_policy": cfg.negative_policy, "negative_predictions_raw": neg,
        "monotone_features": monotone_features(cfg.algorithm, cfg.features, cfg.monotone),
        "scenario_compatible": cfg.mode == "whatif" and cfg.algorithm != "seasonal_naive",
        "cv_fold_details": fold_details,
        "holdout_dates": int(hold_rows["Date"].nunique()), "seed": cfg.seed,
        "runtime_seconds": round(runtime, 2),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
        "config": asdict(cfg),
    }
    try:
        import catboost, xgboost  # noqa: E401
        md["versions"].update({"catboost": catboost.__version__, "xgboost": xgboost.__version__})
    except Exception:
        pass
    if save:
        say(0.95, "Saving pipeline, predictions and metadata")
        registry.save(model_id, md, final_model, eval_model,
                      {"predictions_holdout": hold_pred, "predictions_cv": cv_pred, "predictions_test": test_pred, "importance": imp})
    say(1.0, f"Done in {runtime:.1f}s — holdout MAE {m_hold['MAE']:,.1f}, R² {m_hold['R2']:.3f}")
    return TrainResult(model_id, md, hold_pred, cv_pred, test_pred, imp)


ABLATIONS_DIR = config.ARTIFACTS_DIR / "ablations"


ABLATION_EXPERIMENTS = {
    "All selected features": None,
    "Aviation only": ("aviation",),
    "Calendar only": ("calendar",),
    "Aviation + calendar": ("aviation", "calendar"),
    "Guest history only": ("history",),
    "All minus aviation": ("-", "aviation"),
    "All minus calendar": ("-", "calendar"),
    "All minus guest history": ("-", "history"),
    "All minus hotel observations": ("-", "hotel"),
}


def feature_family(f: str) -> str:
    """Ablation families: aviation (incl. derived and prior-week aviation), calendar, guest history,
    same-day hotel observations, and the market identifier (kept in every experiment)."""
    g = CATALOG_BY_NAME[f].group
    if f.startswith("guests_lag364"):
        return "history"
    if g in ("aviation", "derived") or f.endswith("_prev7d"):
        return "aviation"
    return {"time": "calendar", "competition": "hotel", "market": "market"}.get(g, g)


def ablation_sets(base: list[str], names: list[str]) -> dict[str, list[str]]:
    out = {}
    fam = {f: feature_family(f) for f in base}
    keep = [f for f in base if fam[f] == "market"]
    for n in names:
        rule = ABLATION_EXPERIMENTS[n]
        if rule is None:
            feats = list(base)
        elif rule[0] == "-":
            feats = [f for f in base if fam[f] not in rule[1:]]
        else:
            feats = [f for f in base if fam[f] in rule] + keep
        feats = list(dict.fromkeys(feats))
        duplicate = any(set(feats) == set(v) for v in out.values())
        if feats and [f for f in feats if fam[f] != "market"] and (rule is None or set(feats) != set(base)) and not duplicate:
            out[n] = feats
    return out


def run_ablation(base: TrainConfig, a: Analytic, progress: ProgressFn | None = None, tag: str | None = None,
                 experiments: list[str] | None = None, source_model_id: str | None = None) -> tuple[str, pd.DataFrame]:
    """Same algorithm, hyperparameters and chronological holdout; feature subsets of the base model.

    Holdout predictions of every variant are saved to artifacts/ablations/<id>/ and the summary
    metrics are recomputed from that file, so every number is traceable. Differences are relative to
    "All selected features" (the base model's own feature set)."""
    from .evaluation import metrics_from_predictions
    say = progress or (lambda f, m: None)
    names = experiments or list(ABLATION_EXPERIMENTS)
    if "All selected features" not in names:
        names = ["All selected features"] + list(names)
    sets = ablation_sets(base.features, names)
    preds, info = [], []
    for i, (name, feats) in enumerate(sets.items()):
        say(i / max(len(sets), 1), f"Ablation {i + 1}/{len(sets)}: {name} ({len(feats)} features)")
        cfg = TrainConfig(**{**asdict(base), "features": feats, "cv_folds": 0, "permutation_importance": False})
        r = train(cfg, a, save=False)
        preds.append(r.holdout.assign(feature_set=name))
        info.append({"feature_set": name, "n_features": len(feats), "features": ", ".join(feats), "note": ""})
    for n in names:
        if n not in sets:
            info.append({"feature_set": n, "n_features": 0, "features": "", "note": "skipped — no selected features in this family, or identical to the full set"})
    pred = pd.concat(preds, ignore_index=True)
    abl_id = tag or f"ABL-{base.scope}-{base.mode}-{base.algorithm}-{datetime.now():%Y%m%d-%H%M%S}"
    d = ABLATIONS_DIR / abl_id
    d.mkdir(parents=True, exist_ok=True)
    pred.to_csv(d / "predictions_holdout.csv", index=False)
    summary = pd.DataFrame(info)
    m = {fs: metrics_from_predictions(g) for fs, g in pred.groupby("feature_set")}
    summary = summary.join(pd.DataFrame(m).T, on="feature_set")
    summary["holdout"] = f"{pred['Date'].min():%Y-%m-%d} → {pred['Date'].max():%Y-%m-%d}"
    summary.to_csv(d / "summary.csv", index=False)
    (d / "config.json").write_text(json.dumps({**asdict(base), "source_model_id": source_model_id, "experiments": list(sets)}, indent=2, default=str))
    say(1.0, "Ablation complete")
    return abl_id, summary
