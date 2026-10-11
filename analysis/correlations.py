"""Shared correlation analysis used by the scripts in this folder.

One observation = one calendar day. Frames come from quanara.correlation.eda_frame (complete daily calendar from
2023-01-01, every numeric feature, observed Guests on training dates only). The analysis window is the modelling
window (2023-03-01 → 2025-07-31) restricted to days with an observed target, exactly as in the Correlation Lab.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quanara import config  # noqa: E402
from quanara import correlation as C  # noqa: E402
from quanara.features import CATALOG_BY_NAME, available_features  # noqa: E402

LEVELS = {"international_total": "International", "domestic": "Domestic", "overall": "Overall"}
RESULTS = ROOT / "results" / "correlations"
FIGURES = ROOT / "figures"
_A = None


def analytic():
    global _A
    if _A is None:
        from quanara.integration import get_analytic
        _A = get_analytic()
    return _A


def frame(scope: str, market: str | None = None) -> pd.DataFrame:
    """Observed days of one scope (or one source market) inside the modelling window."""
    level = "International — one source market" if market else LEVELS[scope]
    df = C.eda_frame(analytic(), level, market)
    df = df[(df["Date"] >= config.MODEL_START) & (df["Date"] <= config.HOTEL_TRAIN_END)]
    return df.reset_index(drop=True)


def features_of(df: pd.DataFrame) -> list[str]:
    return [c for c in C.numeric_columns(df) if c != "Guests"]


def feature_to_target(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Every eligible feature against Guests: Pearson, Spearman and Kendall with p-values, valid n, the lag-1
    autocorrelations and the effective sample size, plus the same Pearson on day-over-day differences."""
    obs = df[df["Guests"].notna()]
    whatif = set(available_features(scope, "whatif"))
    rows = []
    for f in features_of(df):
        s = C.pair_stats(obs[f], obs["Guests"])
        d = C.pair_stats(C.transform(df[f].astype(float), "diff1")[obs.index], C.transform(df["Guests"].astype(float), "diff1")[obs.index])
        rows.append({"feature": f, "label": C.label(f), "group": CATALOG_BY_NAME[f].group, "whatif_eligible": f in whatif,
                     "n": s["n"], "missing_pct": round(float(obs[f].isna().mean() * 100), 3),
                     "pearson_r": s["pearson"], "pearson_p": s["pearson_p"], "spearman_rho": s["spearman"], "spearman_p": s["spearman_p"],
                     "kendall_tau": s["kendall"], "lag1_autocorr_feature": s["rho_x"], "lag1_autocorr_guests": s["rho_y"],
                     "effective_n": s["n_eff"], "pearson_r_day_over_day": d["pearson"], "n_day_over_day": d["n"]})
    out = pd.DataFrame(rows)
    return out.reindex(out["pearson_r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def feature_to_feature(df: pd.DataFrame, method: str, threshold: float = 0.95) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlation matrix between predictors (pairwise complete days) and the long list of pairs with valid n."""
    obs = df[df["Guests"].notna()]
    feats = features_of(df)
    m = obs[feats].astype(float).corr(method=method)
    valid = obs[feats].notna().astype(int)
    n = valid.T @ valid
    pairs = []
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            r = m.loc[a, b]
            if np.isfinite(r):
                pairs.append({"feature_a": a, "feature_b": b, f"{method}_r": r, "abs_r": abs(r), "n": int(n.loc[a, b]),
                              "redundant_at_threshold": abs(r) >= threshold})
    return m, pd.DataFrame(pairs).sort_values("abs_r", ascending=False).reset_index(drop=True)


def write(df: pd.DataFrame, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    df.to_csv(p, index=False, float_format="%.6g")
    return p


def run_method(method: str) -> None:
    for scope, lv in LEVELS.items():
        df = frame(scope)
        ft = feature_to_target(df, scope)
        cols = ["feature", "label", "group", "whatif_eligible", "n", "missing_pct"] + (
            ["pearson_r", "pearson_p", "effective_n", "pearson_r_day_over_day", "n_day_over_day"] if method == "pearson" else
            ["spearman_rho", "spearman_p", "effective_n"])
        key = "pearson_r" if method == "pearson" else "spearman_rho"
        ft = ft[cols].reindex(ft[key].abs().sort_values(ascending=False).index)
        p1 = write(ft, f"{scope}_feature_to_target_{method}.csv")
        m, pairs = feature_to_feature(df, method)
        p2 = write(m.reset_index(names="feature"), f"{scope}_feature_matrix_{method}.csv")
        p3 = write(pairs, f"{scope}_feature_pairs_{method}.csv")
        top = ft.head(5)
        print(f"\n{lv} — {len(ft)} features, {int(ft['n'].max())} observed days ({config.MODEL_START.date()} → {config.HOTEL_TRAIN_END.date()})")
        for r in top.itertuples():
            print(f"  {getattr(r, key):+.3f}  {r.label}")
        print(f"  {int(pairs['redundant_at_threshold'].sum())} predictor pairs with |r| ≥ 0.95 · wrote {p1.name}, {p2.name}, {p3.name}")
