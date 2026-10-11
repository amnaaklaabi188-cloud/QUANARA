"""What-if simulation engine.

Workflow (each step is recorded in `ScenarioResult.steps`):
  1. retrieve the baseline route-day flight rows for the selected market and period;
  2. build baseline model inputs with the SAME feature code used in training;
  3. apply the scenario to the route-day rows;
  4. recompute dependent quantities jointly:
        seats'   = seats × (1 + Δseats)                 (cabin capacities scale with seats)
        flights' = flights × (1 + Δfreq)                (Δfreq = Δseats unless set independently,
                                                         in which case seats/flight absorbs the rest
                                                         — total seats are never double-counted)
        PAX'     = LF' × seats'                         (LF' = baseline LF, a set value, or LF + Δpp)
        P2P'     = share' × PAX'                        (share' = baseline P2P share, set, or + Δpp)
        Transfer'= PAX' − P2P' − Transit'               (Transit ≈ 0: non-zero on 3 daily rows; scaled with PAX)
  5. validate capacity / passenger / load-factor consistency;
  6. recompute trailing and derived features from the modified rows;
  7. load the activated model; 8. predict baseline; 9. predict scenario with the same model;
 10. compute absolute and percentage change; 11. save configuration and outputs.

Outputs are model predictions under changed inputs. They are associations learned from
historical data — not causal effects.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from . import config, registry
from .features import CATALOG_BY_NAME, aggregate_aviation, aviation_features, build_feature_frame
from .integration import Analytic, history_frame

LOOKBACK_DAYS = config.FEATURE_WARMUP_DAYS  # longest trailing dependency of any feature
LF_HARD_MAX = 1.5


@dataclass
class ScenarioConfig:
    model_id: str
    market: str = "ALL"                 # market_key (e.g. "INDIA") or "ALL" for every departure country
    start: str = "2026-01-01"
    end: str = "2026-01-31"
    seats_change_pct: float = 0.0
    frequency_mode: str = "follow_seats"  # follow_seats | independent
    frequency_change_pct: float = 0.0
    load_factor_mode: str = "baseline"    # baseline | set | delta
    load_factor_value: float = 0.0        # fraction for "set", percentage points for "delta"
    p2p_share_mode: str = "baseline"      # baseline | set | delta
    p2p_share_value: float = 0.0
    sustained: bool = True                # change already in effect for the 56-day feature lookback
    note: str = ""

    def effective_frequency_pct(self) -> float:
        return self.seats_change_pct if self.frequency_mode == "follow_seats" else self.frequency_change_pct

    def seats_per_flight_change_pct(self) -> float:
        return ((1 + self.seats_change_pct / 100) / (1 + self.effective_frequency_pct() / 100) - 1) * 100


@dataclass
class ScenarioResult:
    scenario_id: str
    config: dict
    model: dict
    daily: pd.DataFrame
    summary: dict
    aviation_changes: pd.DataFrame
    feature_changes: pd.DataFrame
    market_contrib: pd.DataFrame | None
    checks: list[dict]
    extrapolation: pd.DataFrame
    warnings: list[str]
    steps: list[str] = field(default_factory=list)
    comparison: pd.DataFrame | None = None


class ScenarioError(ValueError):
    pass


# ---------------------------------------------------------------------------------- helpers

def _history(a: Analytic, scope: str) -> pd.DataFrame:
    cache = a.__dict__.setdefault("_hist_cache", {})
    if scope not in cache:
        cache[scope] = history_frame(a, scope)
    return cache[scope]


def supported_period() -> tuple[pd.Timestamp, pd.Timestamp]:
    return config.FLIGHT_DAILY_START + pd.Timedelta(days=LOOKBACK_DAYS), config.FLIGHT_END


def apply_scenario(rows: pd.DataFrame, cfg: ScenarioConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Return modified copy of route-day rows and the boolean mask of changed rows."""
    start = pd.Timestamp(cfg.start)
    apply_from = start - pd.Timedelta(days=LOOKBACK_DAYS) if cfg.sustained else start
    m = rows["Date"].between(apply_from, pd.Timestamp(cfg.end))
    if cfg.market != "ALL":
        m &= rows["market_key"].eq(cfg.market)
    r = rows.copy()
    for c in ("Total Seats", "Total PAX", "Total P2P", "Total Transfer", "Total Transit", "flights", "Load Factor",
              "Business Class Seat Capacity", "First Class Seat Capacity"):
        r[c] = r[c].astype(float)
    s = 1 + cfg.seats_change_pct / 100
    f = 1 + cfg.effective_frequency_pct() / 100
    seats0, pax0, p2p0, tr0 = (r.loc[m, c].astype(float) for c in ("Total Seats", "Total PAX", "Total P2P", "Total Transit"))
    lf0 = pax0 / seats0.where(seats0 > 0)
    share0 = (p2p0 / pax0.where(pax0 > 0)).fillna(0)
    lf = {"baseline": lf0, "set": pd.Series(cfg.load_factor_value, index=lf0.index), "delta": lf0 + cfg.load_factor_value / 100}[cfg.load_factor_mode].clip(lower=0)
    share = {"baseline": share0, "set": pd.Series(cfg.p2p_share_value, index=share0.index), "delta": share0 + cfg.p2p_share_value / 100}[cfg.p2p_share_mode].clip(0, 1)
    seats1 = seats0 * s
    # PAX = load factor × seats (unchanged load factor → PAX scales with seats); P2P = P2P share × PAX; transit scales with PAX.
    # Unchanged inputs are carried through as recorded values times a ratio, so a zero change reproduces every row exactly.
    pax1 = pax0 * s if cfg.load_factor_mode == "baseline" else (lf * seats1).fillna(0)
    ratio = (pax1 / pax0.where(pax0 > 0)).fillna(0)
    transit1 = tr0 * ratio
    p2p1 = p2p0 * ratio if cfg.p2p_share_mode == "baseline" else share * pax1
    r.loc[m, "Total Seats"] = seats1
    for c in ("Business Class Seat Capacity", "First Class Seat Capacity"):
        r.loc[m, c] = r.loc[m, c].astype(float) * s
    r.loc[m, "flights"] = r.loc[m, "flights"].astype(float) * f
    r.loc[m, "Total PAX"] = pax1
    r.loc[m, "Total Transit"] = transit1
    r.loc[m, "Total P2P"] = p2p1
    r.loc[m, "Total Transfer"] = pax1 - p2p1 - transit1
    if cfg.load_factor_mode != "baseline":  # otherwise PAX and seats scale together and the recorded load factor stands
        r.loc[m, "Load Factor"] = lf
    return r, m


def consistency_checks(rows: pd.DataFrame, mask: pd.Series, observed_lf_max: float) -> list[dict]:
    x = rows[mask]
    if x.empty:
        return [{"check": "rows changed", "passed": False, "detail": "no flight rows match the market and period"}]
    pax, seats, p2p, tr, trs = (x[c].astype(float) for c in ("Total PAX", "Total Seats", "Total P2P", "Total Transfer", "Total Transit"))
    lf = pax / seats.where(seats > 0)
    return [
        {"check": "Seats ≥ 0", "passed": bool((seats >= 0).all()), "detail": f"min {seats.min():,.1f}"},
        {"check": "PAX = P2P + Transfer + Transit", "passed": bool(((pax - p2p - tr - trs).abs() < 1e-6).all()), "detail": "identity recomputed on every changed route-day"},
        {"check": "0 ≤ P2P ≤ PAX", "passed": bool(((p2p >= -1e-9) & (p2p <= pax + 1e-9)).all()), "detail": f"P2P share {p2p.sum() / max(pax.sum(), 1):.1%}"},
        {"check": "Transfer ≥ 0", "passed": bool((tr >= -1e-9).all()), "detail": f"min {tr.min():,.1f}"},
        {"check": f"Load factor ≤ {LF_HARD_MAX:.0%} (hard limit)", "passed": bool((lf.fillna(0) <= LF_HARD_MAX).all()), "detail": f"max {lf.max():.1%}"},
        {"check": f"Load factor within observed daily maximum ({observed_lf_max:.1%})", "passed": bool((lf.fillna(0) <= observed_lf_max + 1e-9).all()),
         "detail": f"max {lf.max():.1%} (advisory: above it the scenario is outside observed operations)"},
    ]


def _frames(a: Analytic, scope: str, rows: pd.DataFrame, dates: pd.DatetimeIndex, markets: list[str] | None) -> pd.DataFrame:
    lo = dates.min() - pd.Timedelta(days=LOOKBACK_DAYS + 7)
    r = rows[rows["Date"].between(lo, dates.max())]
    by_market = scope == "international_market"
    agg = aggregate_aviation(r, by_market=by_market, markets=markets, start=lo, end=dates.max())
    av = aviation_features(agg, by_market=by_market)
    return build_feature_frame(av, _history(a, scope), by_market=by_market, dates=dates)


def _aviation_table(rows: pd.DataFrame, cfg: ScenarioConfig, start, end) -> pd.DataFrame:
    m = rows["Date"].between(start, end)
    if cfg.market != "ALL":
        m &= rows["market_key"].eq(cfg.market)
    x = rows[m]
    seats, pax, p2p, tr, fl = (float(x[c].sum()) for c in ("Total Seats", "Total PAX", "Total P2P", "Total Transfer", "flights"))
    return pd.Series({"Seats": seats, "Flights": fl, "Seats per flight": seats / fl if fl else np.nan, "PAX": pax, "P2P passengers": p2p,
                      "Transfer passengers": tr, "Load factor": pax / seats if seats else np.nan, "P2P share": p2p / pax if pax else np.nan})


def _extrapolation(md: dict, frame: pd.DataFrame, scope: str, market: str) -> pd.DataFrame:
    rng = md["feature_ranges"]
    out = []
    for f in md["features"]:
        if CATALOG_BY_NAME[f].kind != "numeric":
            continue
        sub = frame
        bounds = rng["global"].get(f)
        if scope == "international_market" and market != "ALL":
            sub = frame[frame["market_key"] == market]
            bounds = rng.get("by_market", {}).get(market, {}).get(f, bounds)
        if not bounds or sub.empty:
            continue
        v = sub[f].astype(float)
        outside = ((v < bounds[0]) | (v > bounds[1])) & v.notna()
        out.append({"feature": f, "label": CATALOG_BY_NAME[f].label, "train_min": bounds[0], "train_max": bounds[1],
                    "scenario_min": float(v.min()), "scenario_max": float(v.max()), "pct_days_outside": float(outside.mean() * 100)})
    return pd.DataFrame(out)


def _predict(model_id: str, frame: pd.DataFrame) -> np.ndarray:
    md = registry.load_metadata(model_id)
    model = registry.load_model(model_id, "final")
    cols = list(dict.fromkeys(md["features"] + (["guests_lag364", "guests_lag364_7d"] + (["market"] if md["scope"] == "international_market" else [])
                                                 if md["algorithm"] == "seasonal_naive" else [])))
    return np.asarray(model.predict(frame[cols]), dtype=float)


def selection_error_band(model_id: str, md: dict, market: str) -> tuple[float, str]:
    """90th percentile of |actual − predicted| / predicted over holdout days, for the same
    selection the scenario reports on (one market, or the daily total), from saved predictions."""
    hold = registry.load_table(model_id, "predictions_holdout")
    basis = "holdout days, daily total"
    if md["scope"] == "international_market":
        if market != "ALL":
            hold = hold[hold["market_key"] == market]
            basis = f"holdout days, {market.title()}"
        hold = hold.groupby("Date", as_index=False)[["y_true", "y_pred"]].sum()
    if hold.empty:
        return float(md["interval"]["rel_q90"]), "holdout, all rows"
    rel = (hold["y_true"] - hold["y_pred"]).abs() / hold["y_pred"].abs().clip(lower=1)
    return float(np.quantile(rel, 0.9)), basis + f" (n={len(hold)})"


def compatible_models(model_id: str) -> list[dict]:
    md = registry.load_metadata(model_id)
    return [m for m in registry.list_models() if m["scope"] == md["scope"] and m["mode"] == "whatif" and m["grain"] == md["grain"]]


# ---------------------------------------------------------------------------------- main entry

def validate_config(cfg: ScenarioConfig, md: dict, a: Analytic) -> list[str]:
    if md["mode"] != "whatif":
        raise ScenarioError("The selected model was trained in Competition mode (it needs observed hotel arrivals). Activate a What-if-mode model.")
    lo, hi = supported_period()
    s, e = pd.Timestamp(cfg.start), pd.Timestamp(cfg.end)
    if s > e:
        raise ScenarioError("Period start is after period end.")
    if s < lo or e > hi:
        raise ScenarioError(f"Supported scenario dates are {lo.date()} → {hi.date()} (daily flight data plus a {LOOKBACK_DAYS}-day feature lookback).")
    if not -100 < cfg.seats_change_pct <= 300 or not -100 < cfg.effective_frequency_pct() <= 300:
        raise ScenarioError("Seat and frequency changes must lie in (−100%, +300%].")
    if cfg.load_factor_mode == "set" and not 0 < cfg.load_factor_value <= LF_HARD_MAX:
        raise ScenarioError(f"Load factor must be in (0, {LF_HARD_MAX:.0%}].")
    if cfg.p2p_share_mode == "set" and not 0 <= cfg.p2p_share_value <= 1:
        raise ScenarioError("P2P share must be between 0% and 100%.")
    if md["scope"] == "international_market" and cfg.market != "ALL" and cfg.market not in (md.get("markets") or []):
        raise ScenarioError(f"{cfg.market} is not a market of this model.")
    if cfg.market != "ALL" and cfg.market not in a.markets:
        raise ScenarioError(f"{cfg.market} has no departure-country match in the flight data.")
    warns = []
    if md["scope"] == "domestic":
        warns.append("Domestic guests are UAE residents; inbound-flight features have no direct mechanism on domestic stays. The domestic model's response to flight changes reflects historical co-movement (largely seasonal), not a plausible channel — interpret with great caution.")
    if e <= config.HOTEL_TRAIN_END:
        warns.append("The period lies inside the observed training history; the scenario is a retrospective counterfactual and the model has already seen these dates (the deployment model is refit on all observed dates).")
    if cfg.frequency_mode == "independent":
        warns.append(f"Frequency is set independently: seats per flight change by {cfg.seats_per_flight_change_pct():+.1f}% so that total seats change by exactly {cfg.seats_change_pct:+.1f}%.")
    return warns


def run(cfg: ScenarioConfig, a: Analytic, compare_with: list[str] | None = None, save: bool = True) -> ScenarioResult:
    md = registry.load_metadata(cfg.model_id)
    warns = validate_config(cfg, md, a)
    steps = []
    scope = md["scope"]
    by_market = scope == "international_market"
    dates = pd.date_range(cfg.start, cfg.end, freq="D")
    markets = md.get("markets") if by_market else None

    base_rows = a.flight_daily_rows
    steps.append(f"1. Retrieved {int(base_rows['Date'].between(dates.min(), dates.max()).sum()):,} baseline route-day rows in the period "
                 f"({'all markets' if cfg.market == 'ALL' else cfg.market}).")
    base_frame = _frames(a, scope, base_rows, dates, markets)
    steps.append(f"2. Built baseline model inputs: {len(base_frame):,} rows × {len(md['features'])} features with the training feature code.")
    scn_rows, mask = apply_scenario(base_rows, cfg)
    steps.append(f"3. Applied the scenario to {int(mask.sum()):,} route-day rows "
                 f"({'sustained: from ' + str((pd.Timestamp(cfg.start) - pd.Timedelta(days=LOOKBACK_DAYS)).date()) if cfg.sustained else 'from ' + cfg.start} to {cfg.end}).")
    steps.append("4. Recomputed seats, flights, PAX, P2P, transfer and load factor jointly on every changed row.")
    m_lf = base_rows["Load Factor"] if cfg.market == "ALL" else base_rows.loc[base_rows["market_key"] == cfg.market, "Load Factor"]
    checks = consistency_checks(scn_rows, mask, float(m_lf.max()))
    hard_fail = [c for c in checks[:5] if not c["passed"]]
    if hard_fail:
        raise ScenarioError("Scenario failed consistency checks: " + "; ".join(f"{c['check']} ({c['detail']})" for c in hard_fail))
    if not checks[5]["passed"]:
        warns.append(f"Scenario load factor exceeds the highest daily load factor ever observed for this selection ({float(m_lf.max()):.1%}).")
    steps.append(f"5. Validated consistency: {sum(c['passed'] for c in checks)}/{len(checks)} checks passed.")
    if mask.any() and float(base_rows.loc[mask, "Total Seats"].sum()) == 0:
        warns.append("The selected market has no operated flights in the period, so percentage changes have nothing to scale.")
    scn_frame = _frames(a, scope, scn_rows, dates, markets)
    steps.append("6. Recomputed trailing (7/28-day), growth, share and ratio features from the modified rows.")
    steps.append(f"7. Loaded model {cfg.model_id} ({md['algorithm_label']}, trained {md['periods']['deployment_fit'][0]} → {md['periods']['deployment_fit'][1]}).")
    yb = _predict(cfg.model_id, base_frame)
    steps.append("8. Predicted baseline Guests.")
    ys = _predict(cfg.model_id, scn_frame)
    steps.append("9. Predicted scenario Guests with the same model and identical non-aviation inputs.")

    keys = ["Date", "market_key"] if by_market else ["Date"]
    daily = base_frame[keys].copy()
    daily["baseline"] = yb
    daily["scenario"] = ys
    # observed Guests for reference where the period overlaps the training data
    if by_market:
        obs = a.hotel_market[a.hotel_market["split"] == "train"][["Date", "market_key", "Guests"]]
        daily = daily.merge(obs, on=keys, how="left", validate="1:1").rename(columns={"Guests": "observed"})
    else:
        tcol = {"international_total": "intl_guests", "domestic": "dom_guests", "overall": "overall_guests"}[scope]
        obs = a.hotel_total[["Date", tcol]].rename(columns={tcol: "observed"})
        daily = daily.merge(obs, on="Date", how="left", validate="1:1")

    focus = daily if (not by_market or cfg.market == "ALL") else daily[daily["market_key"] == cfg.market]
    if by_market:
        focus_daily = focus.groupby("Date", as_index=False)[["baseline", "scenario", "observed"]].sum(min_count=1)
    else:
        focus_daily = focus
    b_tot, s_tot = float(focus_daily["baseline"].sum()), float(focus_daily["scenario"].sum())
    n_days = len(dates)
    rel_q, band_basis = selection_error_band(cfg.model_id, md, cfg.market)
    summary = {
        "days": n_days,
        "baseline_avg_daily_guests": b_tot / n_days, "scenario_avg_daily_guests": s_tot / n_days,
        "baseline_guest_days": b_tot, "scenario_guest_days": s_tot, "abs_change_guest_days": s_tot - b_tot,
        "baseline_guest_nights": b_tot, "scenario_guest_nights": s_tot,   # legacy aliases (= guest-days); kept for saved results
        "abs_change_avg_daily": (s_tot - b_tot) / n_days, "abs_change_guest_nights": s_tot - b_tot,
        "pct_change": (s_tot / b_tot - 1) * 100 if b_tot else float("nan"),
        "observed_avg_daily_guests": float(focus_daily["observed"].sum() / n_days) if focus_daily["observed"].notna().all() else None,
        "daily_error_band_rel_q90": rel_q, "daily_error_band_basis": band_basis,
        "seats_change_pct": cfg.seats_change_pct, "frequency_change_pct": cfg.effective_frequency_pct(),
        "seats_per_flight_change_pct": cfg.seats_per_flight_change_pct(),
    }
    steps.append(f"10. Change: {summary['abs_change_avg_daily']:+,.0f} guests/day ({summary['pct_change']:+.2f}%).")

    av = pd.DataFrame({"baseline": _aviation_table(base_rows, cfg, dates.min(), dates.max()),
                       "scenario": _aviation_table(scn_rows, cfg, dates.min(), dates.max())})
    av["change"] = av["scenario"] - av["baseline"]
    av["change_%"] = np.where(av["baseline"].abs() > 0, av["change"] / av["baseline"] * 100, np.nan)
    av = av.reset_index(names="quantity")

    fsub_b = base_frame if (not by_market or cfg.market == "ALL") else base_frame[base_frame["market_key"] == cfg.market]
    fsub_s = scn_frame if (not by_market or cfg.market == "ALL") else scn_frame[scn_frame["market_key"] == cfg.market]
    fc = []
    for f in md["features"]:
        if CATALOG_BY_NAME[f].kind != "numeric":
            continue
        b, s_ = float(fsub_b[f].mean()), float(fsub_s[f].mean())
        fc.append({"feature": f, "label": CATALOG_BY_NAME[f].label, "group": CATALOG_BY_NAME[f].group, "baseline_mean": b, "scenario_mean": s_,
                   "change_%": (s_ / b - 1) * 100 if b else np.nan, "changed": bool(not np.isclose(b, s_, equal_nan=True))})
    feature_changes = pd.DataFrame(fc)

    market_contrib = None
    if by_market:
        market_contrib = daily.groupby("market_key", as_index=False)[["baseline", "scenario"]].sum()
        market_contrib["change"] = market_contrib["scenario"] - market_contrib["baseline"]
        market_contrib["change_%"] = market_contrib["change"] / market_contrib["baseline"] * 100
        market_contrib = market_contrib.sort_values("baseline", ascending=False)
        tot_b, tot_s = market_contrib["baseline"].sum(), market_contrib["scenario"].sum()
        summary["all_markets_baseline_guest_days"] = summary["all_markets_baseline_guest_nights"] = float(tot_b)
        summary["all_markets_scenario_guest_days"] = summary["all_markets_scenario_guest_nights"] = float(tot_s)
        summary["all_markets_pct_change"] = float((tot_s / tot_b - 1) * 100)

    extrap = _extrapolation(md, scn_frame, scope, cfg.market)
    if len(extrap) and (extrap["pct_days_outside"] > 0).any():
        n = int((extrap["pct_days_outside"] > 0).sum())
        msg = f"{n} model feature(s) leave the training range on some days — predictions there are extrapolations."
        if md["algorithm"] in ("random_forest", "catboost", "xgboost"):
            msg += " Tree models predict flat beyond the training range, so they understate responses to large changes."
        warns.append(msg)
    changed = feature_changes[feature_changes["changed"]] if len(feature_changes) else feature_changes
    if cfg.seats_change_pct != 0 and changed.empty:
        warns.append("No model feature changed — the selected period/market is unaffected or the model uses no scenario-affected features.")
    if abs(summary["pct_change"]) < 1e-9 and not changed.empty:
        warns.append("Features changed but the prediction did not: the model is locally insensitive to these features (e.g. a tree split not crossed).")

    if (daily[["baseline", "scenario"]] < 0).any().any():
        warns.append("Some predicted daily values are below zero — impossible for guest counts. This model extrapolates below 0 for this "
                     "selection; treat the result with caution or use a model trained with a zero floor.")
    sid_src = json.dumps(asdict(cfg), sort_keys=True)
    scenario_id = f"SCN-{datetime.now():%Y%m%d-%H%M%S}-{hashlib.sha1(sid_src.encode()).hexdigest()[:6]}"
    res = ScenarioResult(scenario_id, asdict(cfg), {k: md[k] for k in ("model_id", "label", "scope", "algorithm_label", "metrics_holdout", "periods")},
                         daily, summary, av, feature_changes, market_contrib, checks, extrap, warns, steps)

    if compare_with:
        rows = []
        for mid in dict.fromkeys([cfg.model_id] + list(compare_with)):
            mmd = registry.load_metadata(mid)
            if mmd["scope"] != scope or mmd["mode"] != "whatif":
                continue
            pb, ps = _predict(mid, base_frame), _predict(mid, scn_frame)
            sel = np.ones(len(base_frame), bool) if (not by_market or cfg.market == "ALL") else (base_frame["market_key"] == cfg.market).to_numpy()
            b, s_ = float(pb[sel].sum()), float(ps[sel].sum())
            rows.append({"model_id": mid, "algorithm": mmd["algorithm_label"], "holdout_WMAPE": mmd["metrics_holdout"]["WMAPE"],
                         "holdout_R2": mmd["metrics_holdout"]["R2"], "baseline_avg_daily": b / n_days, "scenario_avg_daily": s_ / n_days,
                         "change_avg_daily": (s_ - b) / n_days, "pct_change": (s_ / b - 1) * 100 if b else np.nan})
        res.comparison = pd.DataFrame(rows)

    if save:
        d = config.SCENARIOS_DIR / scenario_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "assumptions.json").write_text(json.dumps(assumptions(res), indent=2, default=str))
        daily.to_csv(d / "daily_predictions.csv", index=False)
        results_table(res).to_csv(d / "results.csv", index=False)
        res.steps.append(f"11. Saved configuration and outputs to artifacts/scenarios/{scenario_id}/.")
    return res


def assumptions(res: ScenarioResult) -> dict:
    c = res.config
    return {
        "scenario_id": res.scenario_id, "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": res.model, "config": c,
        "derived": {"effective_frequency_change_pct": res.summary["frequency_change_pct"],
                    "seats_per_flight_change_pct": res.summary["seats_per_flight_change_pct"]},
        "assumptions": [
            "Hotel-side inputs (seasonality, last-year guests) are held at their baseline values.",
            "Cabin mix, route network and airline mix are unchanged; capacity scales proportionally on every affected route-day.",
            f"Load factor: {c['load_factor_mode']}{'' if c['load_factor_mode'] == 'baseline' else ' ' + str(c['load_factor_value'])}; "
            f"P2P share: {c['p2p_share_mode']}{'' if c['p2p_share_mode'] == 'baseline' else ' ' + str(c['p2p_share_value'])}.",
            "Sustained change: in effect for the 56-day feature lookback before the period." if c["sustained"] else "Change starts on the first day of the period (trailing features ramp in).",
            "Departure country is used as a proxy for the guest source market; departing passengers are not all nationals of that country.",
            "Guests are reported per day and as guest-days (Σ daily Guests); guest nights are not estimated because length of stay is not in the files.",
            "Results are predictive associations, not causal effects.",
        ],
        "checks": res.checks, "warnings": res.warnings, "summary": res.summary, "steps": res.steps,
    }


def results_table(res: ScenarioResult) -> pd.DataFrame:
    s = res.summary
    rows = [
        ("Baseline predicted guests (avg per day)", s["baseline_avg_daily_guests"]),
        ("Scenario predicted guests (avg per day)", s["scenario_avg_daily_guests"]),
        ("Absolute difference (guests per day)", s["abs_change_avg_daily"]),
        ("Percentage change (%)", s["pct_change"]),
        ("Baseline guest-days (Σ daily Guests)", s["baseline_guest_days"]),
        ("Scenario guest-days (Σ daily Guests)", s["scenario_guest_days"]),
        ("Difference in guest-days", s["abs_change_guest_days"]),
    ]
    df = pd.DataFrame(rows, columns=["metric", "value"])
    df.insert(0, "scenario_id", res.scenario_id)
    df["model_id"] = res.config["model_id"]
    df["market"] = res.config["market"]
    df["period"] = f"{res.config['start']} → {res.config['end']}"
    return df


def judges_summary(res: ScenarioResult) -> str:
    c, s = res.config, res.summary
    mk = "all departure countries" if c["market"] == "ALL" else c["market"].title()
    lines = [
        f"QuanAra — scenario {res.scenario_id}",
        f"Question: what happens to {res.model['scope'].replace('_', ' ')} hotel guests in Abu Dhabi if seat capacity from {mk} changes by {c['seats_change_pct']:+.0f}% "
        f"({c['start']} → {c['end']})?",
        f"Model: {res.model['algorithm_label']} ({res.model['model_id']}); holdout WMAPE {res.model['metrics_holdout']['WMAPE']:.1%}, R² {res.model['metrics_holdout']['R2']:.2f}.",
        f"Baseline: {s['baseline_avg_daily_guests']:,.0f} guests/day · Scenario: {s['scenario_avg_daily_guests']:,.0f} guests/day · "
        f"Change: {s['abs_change_avg_daily']:+,.0f}/day ({s['pct_change']:+.2f}%), {s['abs_change_guest_days']:+,.0f} guest-days over {s['days']} days.",
        f"Typical daily model error for this selection (90% of {s['daily_error_band_basis']} within): ±{s['daily_error_band_rel_q90']:.0%} of the prediction.",
        "This is a model-based predictive scenario, not a causal estimate.",
    ]
    if res.warnings:
        lines.append("Warnings: " + " | ".join(res.warnings))
    return "\n".join(lines)
