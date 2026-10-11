"""Out-of-sample metrics. Always computed from saved prediction tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def metrics(y_true, y_pred) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 3:
        return {k: float("nan") for k in ("MAE", "RMSE", "R2", "WMAPE", "Pearson", "Spearman")} | {"n": int(len(y))}
    e = y - p
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "R2": float(1 - np.sum(e ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "WMAPE": float(np.sum(np.abs(e)) / np.sum(np.abs(y))) if np.sum(np.abs(y)) > 0 else float("nan"),
        "Pearson": float(stats.pearsonr(y, p)[0]) if np.std(p) > 0 else float("nan"),
        "Spearman": float(stats.spearmanr(y, p)[0]) if np.std(p) > 0 else float("nan"),
        "n": int(len(y)),
    }


def metrics_from_predictions(pred: pd.DataFrame) -> dict[str, float]:
    return metrics(pred["y_true"], pred["y_pred"])


def daily_aggregate(pred: pd.DataFrame) -> pd.DataFrame:
    """Sum market-level predictions to daily totals (for the source-market model)."""
    return pred.groupby("Date", as_index=False)[["y_true", "y_pred"]].sum()


def cv_summary(cv_pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, g in cv_pred.groupby("fold"):
        m = metrics_from_predictions(g)
        rows.append({"fold": int(fold), "val_start": g["Date"].min(), "val_end": g["Date"].max(), **m})
    return pd.DataFrame(rows)


METRIC_HELP = {
    "MAE": "Mean absolute error, in guests.",
    "RMSE": "Root mean squared error, in guests (penalises large misses).",
    "R2": "Share of variance explained on the holdout (1 = perfect, 0 = predicting the mean).",
    "WMAPE": "Σ|error| / Σ actual — weighted percentage error, robust for small markets.",
    "Pearson": "Linear correlation between actual and predicted.",
    "Spearman": "Rank correlation between actual and predicted.",
}


# ------------------------------------------------------------------ diagnostics (Model Evaluation / Training Studio)

def diagnostics(pred: pd.DataFrame) -> dict[str, float]:
    """Bias and error-distribution statistics. Residual = actual − predicted, so a positive residual
    means the model UNDER-predicted."""
    y = pred["y_true"].astype(float).to_numpy()
    p = pred["y_pred"].astype(float).to_numpy()
    ok = np.isfinite(y) & np.isfinite(p)
    r = y[ok] - p[ok]
    a = np.abs(r)
    if len(r) == 0:
        return {}
    return {"mean_residual": float(r.mean()), "median_residual": float(np.median(r)), "median_abs_error": float(np.median(a)),
            "p90_abs_error": float(np.quantile(a, 0.9)), "max_abs_error": float(a.max()),
            "under_share": float((r > 0).mean()), "over_share": float((r < 0).mean()), "n": int(len(r)),
            "negative_predictions": int((p[ok] < 0).sum())}


def comparison_key(md: dict, hold: pd.DataFrame) -> tuple:
    """Two models are directly comparable only if they predict the same target at the same grain for
    exactly the same evaluation rows (same keys, same dates, same actual values)."""
    keys = ["Date", "market_key"] if "market_key" in hold else ["Date"]
    h = hold.sort_values(keys)
    import hashlib
    sig = hashlib.sha256(pd.util.hash_pandas_object(h[keys + ["y_true"]], index=False).to_numpy().tobytes()).hexdigest()[:12]
    return (md["scope"], md.get("target", "Guests"), md.get("grain", ""), sig)


def fold_table(cv_pred: pd.DataFrame, md: dict) -> pd.DataFrame:
    """Per-fold metrics recomputed from predictions_cv.csv, with the saved fold layout where available."""
    if cv_pred is None or not len(cv_pred):
        return pd.DataFrame()
    det = {d["fold"]: d for d in (md.get("cv_fold_details") or [])}
    eval_start = md["periods"]["eval_fit"][0]
    rows = []
    for fold, g in cv_pred.groupby("fold"):
        m = metrics_from_predictions(g)
        d = det.get(int(fold), {})
        rows.append({"fold": int(fold), "train_start": d.get("train_start", eval_start),
                     "train_end": d.get("train_end", str((g["Date"].min() - pd.Timedelta(days=1)).date())),
                     "val_start": str(g["Date"].min().date()), "val_end": str(g["Date"].max().date()),
                     "train_rows": d.get("train_rows"), "val_rows": int(len(g)), "val_dates": int(g["Date"].nunique()), **m})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ artifact integrity

def verify_artifacts(md: dict, hold: pd.DataFrame, test: pd.DataFrame | None, observed: pd.DataFrame | None) -> list[dict]:
    """Checks that the saved predictions belong to this model and this holdout, and that every saved
    actual is the observed Guests value for that date (and nationality). `observed` has the same keys
    plus a column `Guests` taken from the training files."""
    out = []
    m = metrics_from_predictions(hold)
    saved = md.get("metrics_holdout", {})
    ok = all(np.isclose(m[k], saved.get(k, np.nan), equal_nan=True) for k in ("MAE", "RMSE", "R2", "WMAPE"))
    out.append({"check": "Metrics recomputed from predictions_holdout.csv equal the saved metadata", "ok": bool(ok),
                "detail": f"MAE {m['MAE']:,.2f} vs saved {saved.get('MAE', float('nan')):,.2f}"})
    n_ok = len(hold) == md["n_rows"]["holdout"]
    out.append({"check": "Holdout row count matches the metadata", "ok": bool(n_ok), "detail": f"{len(hold):,} rows vs {md['n_rows']['holdout']:,}"})
    h0, h1 = pd.Timestamp(md["periods"]["holdout"][0]), pd.Timestamp(md["periods"]["holdout"][1])
    fit_end = pd.Timestamp(md["periods"]["eval_fit"][1])
    in_period = bool(hold["Date"].between(h0, h1).all()) and fit_end < hold["Date"].min()
    out.append({"check": "Every holdout date lies after the evaluation model's last training date", "ok": in_period,
                "detail": f"fit ends {fit_end:%Y-%m-%d}; holdout {hold['Date'].min():%Y-%m-%d} → {hold['Date'].max():%Y-%m-%d}"})
    if observed is not None:
        keys = ["Date", "market_key"] if "market_key" in hold else ["Date"]
        j = hold[keys + ["y_true"]].merge(observed[keys + ["Guests"]], on=keys, how="left")
        match = bool(np.allclose(j["y_true"], j["Guests"], equal_nan=False)) if j["Guests"].notna().all() else False
        out.append({"check": "Saved actuals equal the observed Guests in the training files", "ok": match,
                    "detail": f"{int(np.isclose(j['y_true'], j['Guests']).sum()):,} of {len(j):,} rows match"})
    if test is not None and len(test):
        out.append({"check": "Official-test forecasts carry no actual values", "ok": "y_true" not in test.columns,
                    "detail": f"{len(test):,} forecast rows, {test['Date'].min():%Y-%m-%d} → {test['Date'].max():%Y-%m-%d}"})
    return out


# ------------------------------------------------------------------ permutation importance (on demand, descriptive)

def permutation_table(model, X: pd.DataFrame, y: np.ndarray, groups: dict[str, list[str]], n_repeats: int = 5, seed: int = 42,
                      max_rows: int = 4000) -> pd.DataFrame:
    """RMSE increase when a feature (or a group of features, permuted jointly) is shuffled; mean and SD over
    repeats. Negative values are kept: they mean the model did better with the feature shuffled."""
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    base = float(np.sqrt(np.mean((y - model.predict(X)) ** 2)))
    rows = []
    for name, cols in groups.items():
        incs = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            for c in cols:
                Xp[c] = Xp[c].to_numpy()[perm]
            incs.append(float(np.sqrt(np.mean((y - model.predict(Xp)) ** 2))) - base)
        rows.append({"name": name, "columns": ", ".join(cols), "mean": float(np.mean(incs)), "sd": float(np.std(incs, ddof=1)) if n_repeats > 1 else 0.0,
                     "repeats": n_repeats, "rows_used": int(len(X)), "base_rmse": base})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ rule-based insights (no LLM)

def insights(md: dict, hold: pd.DataFrame, baseline: dict | None = None) -> list[dict]:
    """Evidence-based statements computed from the holdout predictions. Each item: title, text, tone."""
    daily = hold.groupby("Date", as_index=False)[["y_true", "y_pred"]].sum() if "market_key" in hold else hold
    m, d = metrics_from_predictions(hold), diagnostics(hold)
    mean_y = float(hold["y_true"].mean())
    out = []
    r2 = m["R2"]
    level = "strong" if r2 >= 0.8 else "moderate" if r2 >= 0.5 else "weak" if r2 >= 0 else "worse than predicting the holdout mean"
    out.append({"title": "Overall performance", "tone": "info" if r2 >= 0.5 else "warn",
                "text": f"On {m['n']:,} unseen rows ({md['periods']['holdout'][0]} → {md['periods']['holdout'][1]}) the model explains {r2:.0%} of the "
                        f"variance (R² = {r2:.3f}, {level}). Its weighted error is {m['WMAPE']:.1%} of actual Guests; the typical miss (median absolute "
                        f"error) is {d['median_abs_error']:,.0f} guests. R² is not a percentage accuracy."})
    rel = d["mean_residual"] / mean_y if mean_y else 0
    if abs(rel) < 0.01 and 0.4 <= d["under_share"] <= 0.6:
        btxt, tone = "no material systematic bias on average", "info"
    else:
        btxt = f"a tendency to {'under' if d['mean_residual'] > 0 else 'over'}-predict, by {abs(d['mean_residual']):,.0f} guests on average ({abs(rel):.1%} of the mean)"
        tone = "warn" if abs(rel) >= 0.02 else "info"
    out.append({"title": "Systematic bias", "tone": tone,
                "text": f"Mean residual (actual − predicted) is {d['mean_residual']:+,.0f}: the model shows {btxt}. It under-predicts on "
                        f"{d['under_share']:.0%} of rows and over-predicts on {d['over_share']:.0%}. A small average can still hide large opposite errors."})
    q = daily["y_true"].quantile(0.9)
    peak = daily[daily["y_true"] >= q]
    if len(peak) >= 3:
        pr = peak["y_true"] - peak["y_pred"]
        out.append({"title": "Demand spikes", "tone": "warn" if pr.mean() > 0 and (pr > 0).mean() >= 0.7 else "info",
                    "text": f"On the busiest 10% of days (≥ {q:,.0f} guests, {len(peak)} days) the mean residual is {pr.mean():+,.0f} and "
                            f"{(pr > 0).mean():.0%} of those days are under-predicted"
                            + (" — peaks are systematically missed." if pr.mean() > 0 and (pr > 0).mean() >= 0.7 else ".")})
    dd = daily.sort_values("Date")
    half = len(dd) // 2
    if half >= 14:
        e1 = (dd["y_true"] - dd["y_pred"]).abs().iloc[:half].mean()
        e2 = (dd["y_true"] - dd["y_pred"]).abs().iloc[half:].mean()
        ch = e2 / e1 - 1 if e1 else 0
        if ch > 0.25:
            tail = "; accuracy deteriorates as the forecast moves away from the training data."
        elif ch < -0.25:
            tail = "; the error is concentrated early in the holdout, so accuracy is not stable over time (a single window cannot show whether this persists)."
        else:
            tail = "; no strong change between the two halves."
        out.append({"title": "Errors over time", "tone": "warn" if abs(ch) > 0.25 else "info",
                    "text": f"Mean absolute error is {e1:,.0f} in the first half of the holdout and {e2:,.0f} in the second ({ch:+.0%}){tail}"})
    if baseline:
        imp = 1 - m["MAE"] / baseline["MAE"] if baseline["MAE"] else float("nan")
        out.append({"title": "Versus the seasonal baseline", "tone": "info" if imp > 0 else "warn",
                    "text": f"The same-weekday-last-year baseline scores MAE {baseline['MAE']:,.0f} on exactly the same rows; this model's MAE "
                            f"{m['MAE']:,.0f} is {abs(imp):.0%} {'lower' if imp > 0 else 'higher'}."})
    else:
        out.append({"title": "Versus a baseline", "tone": "muted", "text": "No seasonal-naive baseline with identical evaluation rows is saved for this scope and mode, "
                                                                          "so no baseline comparison is shown."})
    if md["mode"] == "whatif" and md["algorithm"] != "seasonal_naive":
        stxt = ("Eligible for the What-if Simulator: it uses only scenario-time inputs (aviation, calendar, guests ≥ 364 days earlier). Scenario outputs remain "
                "model-based conditional estimates, not causal effects.")
        tone = "info"
    elif md["algorithm"] == "seasonal_naive":
        stxt, tone = "Not suitable for scenarios: the baseline ignores aviation inputs, so a scenario cannot change its prediction.", "warn"
    else:
        stxt, tone = ("Not eligible for the What-if Simulator: it relies on same-day hotel observations (New Arrivals) that exist only "
                      "retrospectively. Its accuracy describes nowcasting, not forecasting."), "warn"
    out.append({"title": "Scenario suitability", "tone": tone, "text": stxt})
    out.append({"title": "What would raise confidence", "tone": "muted",
                "text": "More holdout seasons (only one February–July window is evaluated), rolling-origin evaluation across several years, prediction "
                        "intervals checked for coverage, and a holiday calendar (Ramadan / Eid move between years) — none of which the supplied files allow fully."})
    return out
