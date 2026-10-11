"""Correlation analysis for the Correlation & Visualization Lab.

Every observation is one calendar day at one compatible grain (daily total, or daily × one
source market). Frames are built on a COMPLETE daily calendar from 2023-01-01 (the first day
of daily flight data), so trailing windows, differences and lags always span the intended
number of calendar days — a day without a hotel row is a missing value, never a skipped row.

Statistics are descriptive. Days are autocorrelated, so the classical p-values (which assume
independent observations) are reported only with that caveat, next to an effective sample size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .features import CATALOG_BY_NAME, available_features, time_features
from .integration import Analytic, competition_extra, history_frame, model_frame

LEVELS = {"International": "international_total", "Domestic": "domestic", "Overall": "overall",
          "International — one source market": "international_market"}
HOTEL_COLS = {"Guests": "Guests (target)", "new_arrivals": "New Arrivals (day)", "same_day_guests": "Same-Day Guests (day)"}
UNITS = {"Guests": "guests", "new_arrivals": "guests", "same_day_guests": "guests"}
AVIATION_BASE = {"seats": "Total Seats", "p2p": "Total P2P", "pax": "Total PAX", "transfer": "Total Transfer", "flights": "Flights"}


# ------------------------------------------------------------------ frames

def eda_frame(a: Analytic, level: str, market: str | None = None) -> pd.DataFrame:
    """Daily rows 2023-01-01 → 2026-02-28 on a complete calendar with every numeric feature,
    the observed hotel fields and `Guests` (NaN on test dates and on unobserved days)."""
    scope = LEVELS[level]
    if scope != "international_market":
        f = model_frame(a, scope, "competition")
        f = f[f["Date"] >= config.FLIGHT_DAILY_START].rename(columns={"y": "Guests"})
        f.loc[f["split"] == "test", "Guests"] = np.nan
        return f.reset_index(drop=True)
    if market not in a.markets:
        raise ValueError("choose one of the matched source markets")
    av = a.av_market[a.av_market["market_key"] == market].copy()
    hm = a.hotel_market[(a.hotel_market["market_key"] == market)][["Date", "split", "Guests"]]
    hist = history_frame(a, "international_market")
    extra = competition_extra(a, "international_market")
    df = av.merge(hm, on="Date", how="left", validate="1:1")
    df = df.merge(hist[hist["market_key"] == market].drop(columns="market_key"), on="Date", how="left", validate="1:1")
    df = df.merge(extra[extra["market_key"] == market].drop(columns="market_key"), on="Date", how="left", validate="1:1")
    df = pd.concat([df.reset_index(drop=True), time_features(df["Date"]).reset_index(drop=True)], axis=1)
    df.loc[df["split"] != "train", "Guests"] = np.nan
    return df.sort_values("Date").reset_index(drop=True)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c in CATALOG_BY_NAME and CATALOG_BY_NAME[c].kind == "numeric"]
    return ["Guests"] + [c for c in cols if df[c].notna().any() and df[c].nunique() > 1]


def label(col: str) -> str:
    if col == "Guests":
        return "Guests (target)"
    return CATALOG_BY_NAME[col].label if col in CATALOG_BY_NAME else col


SHORT = {"Guests": "Guests", "seats": "Seats", "p2p": "P2P", "pax": "PAX", "transfer": "Transfer", "flights": "Flights", "load_factor": "LF",
         "seats_7d": "Seats 7d", "p2p_7d": "P2P 7d", "pax_7d": "PAX 7d", "transfer_7d": "Transfer 7d", "flights_7d": "Flights 7d",
         "load_factor_7d": "LF 7d", "seats_28d": "Seats 28d", "p2p_28d": "P2P 28d", "p2p_share_7d": "P2P share", "transfer_share_7d": "Transfer share",
         "seats_per_flight_7d": "Seats/flight", "business_seat_share_7d": "Business share", "etihad_seat_share_7d": "Etihad share",
         "n_routes": "Routes", "n_cities": "Cities", "n_airlines": "Airlines", "seats_growth_28d": "Seats growth", "p2p_growth_28d": "P2P growth",
         "seats_prev7d": "Seats prev wk", "p2p_prev7d": "P2P prev wk", "guests_lag364": "Guests LY", "guests_lag364_7d": "Guests LY 7d",
         "dow": "Weekday", "is_weekend": "Weekend", "month": "Month", "quarter": "Quarter", "year": "Year", "doy_sin": "Season sin",
         "doy_cos": "Season cos", "new_arrivals": "New arrivals", "new_arrivals_7d": "New arr. 7d", "same_day_guests": "Same-day"}


def short_label(col: str) -> str:
    return SHORT.get(col, label(col))


def unit(col: str) -> str:
    if col in UNITS:
        return UNITS[col]
    from .feature_facts import _QTY, _UNIT, _base
    if col in _UNIT:
        return _UNIT[col]
    q = _base(col)
    return _QTY[q][0] if q else ""


def is_hotel(col: str) -> bool:
    return col in HOTEL_COLS


def availability_end(cols: list[str]) -> pd.Timestamp:
    """Last date on which every column can be observed: hotel targets stop at the end of training."""
    return config.HOTEL_TRAIN_END if any(c == "Guests" for c in cols) else config.FLIGHT_END


def transform(s: pd.Series, how: str) -> pd.Series:
    """'levels', 'diff1' (day-over-day change) or 'diff7' (change versus the same weekday a week earlier).
    The frame is calendar-complete, so a shift of k rows is a shift of k days."""
    if how == "diff1":
        return s.diff(1)
    if how == "diff7":
        return s.diff(7)
    return s


# ------------------------------------------------------------------ statistics

def _lag1(v: np.ndarray) -> float:
    if len(v) < 4 or np.std(v) == 0:
        return float("nan")
    return float(np.corrcoef(v[:-1], v[1:])[0, 1])


def pair_stats(x: pd.Series, y: pd.Series) -> dict:
    """Correlations, OLS fit (y = a + b·x) and an effective sample size for autocorrelated days.

    n_eff = n · (1 − ρx·ρy) / (1 + ρx·ρy), with ρ the lag-1 autocorrelations (Bartlett's large-sample
    approximation for the variance of a correlation between two AR(1) series)."""
    xf, yf = pd.to_numeric(x, errors="coerce").astype(float), pd.to_numeric(y, errors="coerce").astype(float)
    ok = xf.notna() & yf.notna() & np.isfinite(xf) & np.isfinite(yf)
    xv, yv = xf[ok].to_numpy(), yf[ok].to_numpy()
    n = len(xv)
    nan = float("nan")
    out = {"n": n, "pearson": nan, "pearson_p": nan, "spearman": nan, "spearman_p": nan, "kendall": nan, "kendall_p": nan,
           "slope": nan, "intercept": nan, "r2": nan, "slope_se": nan, "slope_ci": (nan, nan), "rho_x": nan, "rho_y": nan,
           "n_eff": nan, "x_mean": nan, "sxx": nan, "resid_se": nan}
    if n < 3 or np.std(xv) == 0 or np.std(yv) == 0:
        return out
    pr, sr, kt = stats.pearsonr(xv, yv), stats.spearmanr(xv, yv), stats.kendalltau(xv, yv)
    lr = stats.linregress(xv, yv)
    resid = yv - (lr.intercept + lr.slope * xv)
    sxx = float(np.sum((xv - xv.mean()) ** 2))
    resid_se = float(np.sqrt(np.sum(resid ** 2) / (n - 2)))
    t = stats.t.ppf(0.975, n - 2)
    rx, ry = _lag1(xv), _lag1(yv)
    neff = n * (1 - rx * ry) / (1 + rx * ry) if np.isfinite(rx) and np.isfinite(ry) else nan
    out.update({"pearson": float(pr[0]), "pearson_p": float(pr[1]), "spearman": float(sr[0]), "spearman_p": float(sr[1]),
                "kendall": float(kt[0]), "kendall_p": float(kt[1]), "slope": float(lr.slope), "intercept": float(lr.intercept),
                "r2": float(lr.rvalue ** 2), "slope_se": float(lr.stderr), "slope_ci": (float(lr.slope - t * lr.stderr), float(lr.slope + t * lr.stderr)),
                "rho_x": rx, "rho_y": ry, "n_eff": float(max(min(neff, n), 3)) if np.isfinite(neff) else nan,
                "x_mean": float(xv.mean()), "sxx": sxx, "resid_se": resid_se})
    return out


def mean_ci_band(s: dict, xs: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise confidence band for the fitted MEAN line (classical OLS, independent errors)."""
    t = stats.t.ppf(0.5 + level / 2, s["n"] - 2)
    se = s["resid_se"] * np.sqrt(1 / s["n"] + (xs - s["x_mean"]) ** 2 / s["sxx"])
    fit = s["intercept"] + s["slope"] * xs
    return fit - t * se, fit + t * se


def outlier_mask(df: pd.DataFrame, cols: list[str], k: float = 3.0) -> pd.Series:
    """Extreme values: beyond Q1 − k·IQR or Q3 + k·IQR on any of `cols` (k = 3 flags only far tails)."""
    m = pd.Series(False, index=df.index)
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        if np.isfinite(iqr) and iqr > 0:
            m |= (v < q1 - k * iqr) | (v > q3 + k * iqr)
    return m


def corr_matrix(df: pd.DataFrame, cols: list[str], method: str = "pearson") -> pd.DataFrame:
    return df[cols].astype(float).corr(method=method.lower())


def ranking(df: pd.DataFrame, cols: list[str], target: str, method: str, scope: str, threshold: float) -> pd.DataFrame:
    """Every feature's correlation with the target, its missingness and its strongest partner."""
    feats = [c for c in cols if c != target and c in CATALOG_BY_NAME]
    whatif = set(available_features(scope, "whatif"))
    cm = df[feats].astype(float).corr(method=method.lower()) if len(feats) > 1 else pd.DataFrame()
    rows = []
    for c in feats:
        s = pair_stats(df[c], df[target])
        partner, pr = "", float("nan")
        if not cm.empty:
            others = cm[c].drop(c).abs().dropna()
            if len(others):
                partner = others.idxmax()
                pr = float(cm.loc[partner, c])
        rows.append({"feature": c, "label": label(c), "group": CATALOG_BY_NAME[c].group, "pearson": s["pearson"], "spearman": s["spearman"],
                     "kendall": s["kendall"], "n": s["n"], "missing_pct": float(df[c].isna().mean() * 100),
                     "whatif": c in whatif, "strongest_partner": label(partner) if partner else "", "partner_r": pr,
                     "redundant": bool(np.isfinite(pr) and abs(pr) >= threshold)})
    key = {"Pearson": "pearson", "Spearman": "spearman", "Kendall": "kendall"}[method]
    out = pd.DataFrame(rows)
    return out.sort_values(key, key=lambda v: v.abs(), ascending=False).reset_index(drop=True) if len(out) else out


def redundant_pairs(df: pd.DataFrame, cols: list[str], method: str, threshold: float) -> pd.DataFrame:
    feats = [c for c in cols if c in CATALOG_BY_NAME]
    cm = df[feats].astype(float).corr(method=method.lower())
    out = []
    for i, a_ in enumerate(feats):
        for b_ in feats[i + 1:]:
            r = cm.loc[a_, b_]
            if np.isfinite(r) and abs(r) >= threshold:
                out.append({"feature_a": label(a_), "feature_b": label(b_), "r": float(r)})
    return pd.DataFrame(out).sort_values("r", key=lambda v: v.abs(), ascending=False) if out else pd.DataFrame(columns=["feature_a", "feature_b", "r"])


def lag_correlation(df: pd.DataFrame, base: str, window: int, ycol: str, max_lag: int, method: str = "pearson",
                    start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """corr( Σ_{d=t−k−w+1…t−k} base(d) , y(t) ) for k = 0…max_lag on the complete daily calendar.

    The window sum needs w days and the lag k more, all on or after 2023-01-01; pairs whose window
    would reach before the first daily flight row are undefined and dropped (n is reported per lag)."""
    x = df[base].astype(float).rolling(window, min_periods=window).sum()
    y = df[ycol].astype(float)
    sel = pd.Series(True, index=df.index)
    if start is not None:
        sel &= df["Date"] >= start
    if end is not None:
        sel &= df["Date"] <= end
    rows = []
    for k in range(max_lag + 1):
        xs = x.shift(k)
        ok = sel & xs.notna() & y.notna()
        n = int(ok.sum())
        if n < 30:
            rows.append({"lag": k, "r": np.nan, "n": n})
            continue
        r = xs[ok].corr(y[ok], method=method.lower())
        rows.append({"lag": k, "r": float(r), "n": n})
    return pd.DataFrame(rows)


def aviation_series(df: pd.DataFrame, cols: list[str], window: int) -> pd.DataFrame:
    """Daily aviation counts or trailing w-day sums on the complete calendar (no partial windows)."""
    out = df[["Date"]].copy()
    for c in cols:
        out[c] = df[c].astype(float).rolling(window, min_periods=window).sum() if window > 1 else df[c].astype(float)
    return out


def guests_series(a: Analytic, level: str, market: str | None = None) -> pd.DataFrame:
    """Observed Guests from the training files only (2022-01-01 → 2025-07-31) on a complete calendar."""
    scope = LEVELS[level]
    dates = pd.date_range(config.HOTEL_TRAIN_START, config.HOTEL_TRAIN_END, name="Date")
    if scope == "international_market":
        hm = a.hotel_market[(a.hotel_market["market_key"] == market) & (a.hotel_market["split"] == "train")]
        s = hm.set_index("Date")["Guests"].reindex(dates)
    else:
        col = {"international_total": "intl_guests", "domestic": "dom_guests", "overall": "overall_guests"}[scope]
        ht = a.hotel_total[a.hotel_total["split"] == "train"]
        s = ht.set_index("Date")[col].reindex(dates)
    out = s.rename("Guests").reset_index()
    out["mean_7d"] = out["Guests"].rolling(7, min_periods=7).mean()
    out["mean_28d"] = out["Guests"].rolling(28, min_periods=28).mean()
    return out


def flights_vs_seats(df: pd.DataFrame) -> dict:
    """Why flights and seats are almost perfectly correlated: seats = flights × average gauge."""
    d = df[(df["flights"] > 0) & df["seats"].notna()]
    gauge = d["seats"] / d["flights"]
    return {"r": float(d["flights"].corr(d["seats"])), "r_7d": float(df[["flights_7d", "seats_7d"]].dropna().corr().iloc[0, 1]),
            "gauge_mean": float(gauge.mean()), "gauge_cv": float(gauge.std() / gauge.mean()), "n": int(len(d))}


def weekend_effect(df: pd.DataFrame, ycol: str = "Guests") -> dict:
    d = df[df[ycol].notna()]
    we, wd = d.loc[d["is_weekend"] == 1, ycol], d.loc[d["is_weekend"] == 0, ycol]
    by_dow = d.groupby("dow")[ycol].mean()
    return {"r": float(d["is_weekend"].corr(d[ycol])) if d["is_weekend"].nunique() > 1 else float("nan"),
            "mean_weekend": float(we.mean()), "mean_weekday": float(wd.mean()), "diff_pct": float(we.mean() / wd.mean() - 1),
            "sd_all": float(d[ycol].std()), "by_dow": {int(k): float(v) for k, v in by_dow.items()}, "n": int(len(d))}


DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def link_findings(df: pd.DataFrame) -> list[dict]:
    """The three relationships explained under the feature ranking, each typed by what kind of relationship it is
    (feature-to-feature or feature-to-target; observed association or mathematical construction). Every number is
    computed here from `df` (observed days of one Lab level); the wording never claims causation, and high correlation
    is never presented as proof that a feature adds no information."""
    n = int(len(df))
    fs = flights_vs_seats(df)
    ok = (df["flights"] > 0) & df["seats"].notna()
    rho_fs = float(stats.spearmanr(df.loc[ok, "flights"], df.loc[ok, "seats"])[0])
    sh = df[["p2p_share_7d", "transfer_share_7d"]].dropna()
    r_sh = float(sh.corr().iloc[0, 1]) if len(sh) > 3 else float("nan")
    dev = (sh["p2p_share_7d"] + sh["transfer_share_7d"] - 1).abs()
    n_dev, max_dev = int((dev > 1e-12).sum()), float(dev.max()) if len(dev) else float("nan")
    transit_days = int(((df["pax"] - df["p2p"] - df["transfer"]) > 0.5).sum())
    we = weekend_effect(df)
    by = we["by_dow"]
    busiest = DOW[max(by, key=by.get)] if by else "—"
    quietest = DOW[min(by, key=by.get)] if by else "—"
    return [
        {"badge": "Feature ↔ feature · strong observed association", "tone": "green", "title": "Flights vs seats",
         "stats": [("Pearson r", f"{fs['r']:.3f}"), ("Spearman ρ", f"{rho_fs:.3f}"), ("Days", f"{fs['n']:,}")],
         "summary": (f"Seats move almost one-for-one with flights because the average aircraft size barely changes "
                     f"({fs['gauge_mean']:,.0f} seats per flight, varying by {fs['gauge_cv']:.1%})."),
         "details": [
             "Flights are decoded from the Average Weekly Frequency column and seats come from Total Seats, so neither is computed "
             "from the other; the correlation is an observed regularity of the airline schedule.",
             f"Their trailing 7-day sums are correlated even more strongly (r = {fs['r_7d']:.3f}).",
             "The two inputs carry largely overlapping information and share credit in importance measures. Correlation alone does "
             "not show that one adds nothing: whether both help a model is measured with the ablation study and permutation importance.",
             "Linear models such as Ridge can give strongly collinear inputs offsetting coefficients, so read those coefficients together."]},
        {"badge": "Feature ↔ feature · near-exact accounting identity", "tone": "amber", "title": "P2P share vs transfer share",
         "stats": [("Pearson r", f"{r_sh:.3f}"), ("Days off the identity", f"{n_dev} of {len(sh):,}")],
         "summary": "The two shares mirror each other by construction: every arriving passenger is counted as point-to-point, transfer or transit.",
         "details": [
             "P2P share = ΣP2P ÷ ΣPAX and transfer share = ΣTransfer ÷ ΣPAX over the same 7 days, and PAX = P2P + Transfer + Transit holds "
             "on every flight row, so transfer share = 1 − P2P share − transit share.",
             f"Total Transit is non-zero on {transit_days} day(s) of this period, so the identity is not exact on {n_dev} of {len(sh):,} days "
             f"(largest gap {max_dev:.4f}).",
             "The correlation therefore reflects how the shares are defined, not an empirical finding about travellers. Using both shares in "
             "one model repeats the same quantity; the recommended feature sets include P2P share only."]},
        {"badge": "Feature ↔ target · weak observed association", "tone": "grey", "title": "Weekend indicator vs guests",
         "stats": [("Pearson r", f"{we['r']:+.3f}"), ("Weekend vs weekday", f"{we['diff_pct']:+.1%}"), ("Days", f"{we['n']:,}")],
         "summary": (f"Weekend days average {we['mean_weekend']:,.0f} guests versus {we['mean_weekday']:,.0f} on weekdays — "
                     f"small next to the day-to-day spread (SD {we['sd_all']:,.0f}), which is driven mostly by season."),
         "details": [
             "The indicator is 1 on Saturday and Sunday, the UAE weekend since 1 January 2022.",
             f"By weekday, the busiest day on average is {busiest} and the quietest is {quietest}.",
             "Neighbouring days are strongly autocorrelated, so a nominal p-value would overstate certainty; the calendar effect is small "
             "whichever way it points.",
             "This describes co-movement in historical data, not an effect caused by weekends."]},
    ]


def monthly_history(a: Analytic, scope: str) -> pd.DataFrame:
    """Monthly totals 2022-01 → 2026-02: monthly 2022 flight rows + daily rows summed to month.
    Monthly is the only grain at which 2022 flights are comparable with later years."""
    from .data_loading import load_dataset
    fl = load_dataset("flight")
    fl["month"] = fl["Date"].dt.to_period("M").dt.to_timestamp()
    av = fl.groupby("month")[["Total Seats", "Total P2P", "Total PAX", "Total Transfer"]].sum()
    av["Load Factor"] = av["Total PAX"] / av["Total Seats"]
    col = {"international_total": "intl_guests", "international_market": "intl_guests", "domestic": "dom_guests", "overall": "overall_guests"}[scope]
    h = a.hotel_total.copy()
    h["month"] = h["Date"].dt.to_period("M").dt.to_timestamp()
    g = h.groupby("month")[col].sum(min_count=28).rename("Guest-days")
    out = av.join(g, how="outer").reset_index()
    out["flight_grain"] = np.where(out["month"] < config.FLIGHT_DAILY_START, "monthly rows", "daily rows summed")
    return out
