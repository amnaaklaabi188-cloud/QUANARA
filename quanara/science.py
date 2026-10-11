"""Shared scientific layer used by the Simulator, AI Copilot, Methodology and Judges' Demo.

  conversion_chain   the aviation-to-hospitality pathway for a scenario, stage by stage, each
                     labelled observed / derived / predicted / unavailable — no invented rates
  sensitivity        one-input-at-a-time scenario responses computed by MODEL INFERENCE with the
                     same fitted pipeline, baseline rows, dates and feature code as the scenario
  market_season      observed seasonal and market context for the selected market and period
  scenario_record    one reproducible record per scenario (model, inputs, features, predictions,
                     metrics, warnings, provenance) shared across pages and saved with the outputs

Quantities are typed so pages never blur them: observed (in the files), derived (computed from
observed fields), predicted (model output), assumption (a scenario input) and unavailable (not
estimable from the supplied data).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, registry
from . import scenario as S
from .evaluation import metrics_from_predictions
from .integration import Analytic

FEATURE_CODE = Path(__file__).with_name("features.py")


def pipeline_version() -> str:
    """Short hash of the feature-construction code, recorded with every scenario."""
    return hashlib.sha256(FEATURE_CODE.read_bytes()).hexdigest()[:10]


def population_label(md: dict, market: str) -> str:
    """What the model's output counts — never relabelled by the market selection."""
    if md["scope"] == "international_market":
        return (f"{market.title()} nationality guests" if market != "ALL" else f"guests of the {len(md.get('markets') or [])} matched nationalities")
    return {"international_total": "international guests (all nationalities)", "domestic": "domestic guests",
            "overall": "international + domestic guests"}[md["scope"]]


# ------------------------------------------------------------------ conversion chain

def conversion_chain(res: S.ScenarioResult, md: dict) -> list[dict]:
    av = res.aviation_changes.set_index("quantity")
    b, s = av["baseline"], av["scenario"]
    transit_b = b["PAX"] - b["P2P passengers"] - b["Transfer passengers"]
    transit_s = s["PAX"] - s["P2P passengers"] - s["Transfer passengers"]
    sm = res.summary
    obs = sm.get("observed_avg_daily_guests")
    days = sm["days"]
    return [
        {"stage": "Seat capacity", "kind": "observed", "field": "Total Seats", "baseline": b["Seats"], "scenario": s["Seats"], "unit": "seats",
         "note": f"{b['Flights']:,.0f} → {s['Flights']:,.0f} flights; seats per flight {b['Seats per flight']:,.0f} → {s['Seats per flight']:,.0f}."},
        {"stage": "Passenger volume", "kind": "observed", "field": "Total PAX", "baseline": b["PAX"], "scenario": s["PAX"], "unit": "passengers",
         "note": f"Load factor (derived = PAX ÷ seats) {b['Load factor']:.1%} → {s['Load factor']:.1%}."},
        {"stage": "Destination P2P passengers", "kind": "observed", "field": "Total P2P", "baseline": b["P2P passengers"], "scenario": s["P2P passengers"],
         "unit": "passengers", "note": f"P2P share {b['P2P share']:.1%} → {s['P2P share']:.1%}; transfer {b['Transfer passengers']:,.0f} → {s['Transfer passengers']:,.0f}; "
                                       f"transit {transit_b:,.0f} → {transit_s:,.0f}."},
        {"stage": "Potential visitors", "kind": "unavailable", "field": "—", "baseline": None, "scenario": None, "unit": "",
         "note": "Not estimable: the files do not say what share of P2P passengers are visitors rather than residents or day travellers."},
        {"stage": "Hotel guests", "kind": "predicted", "field": "Guests (model output)", "baseline": sm["baseline_avg_daily_guests"],
         "scenario": sm["scenario_avg_daily_guests"], "unit": "guests per day",
         "note": f"{(lambda t: t[:1].upper() + t[1:])(population_label(md, res.config['market']))}, average per day over {days} days"
                 + (f"; observed in this period: {obs:,.0f} per day." if obs else "; no observed Guests for these dates.")},
        {"stage": "Guest nights", "kind": "unavailable", "field": "—", "baseline": None, "scenario": None, "unit": "",
         "note": f"Not estimated: length of stay is not in the files. The sum of daily Guests ({sm['baseline_guest_days']:,.0f} baseline guest-days) is reported "
                 "as guest-days, not guest nights."},
    ]


# ------------------------------------------------------------------ sensitivity (model inference)

SENSITIVITY_INPUTS = {
    "seats": ("Seat capacity (same aircraft size)", "%", [-20, -10, 10, 20], lambda v: dict(seats_change_pct=float(v))),
    "frequency": ("Flight frequency (seats fixed)", "%", [-20, -10, 10, 20],
                  lambda v: dict(frequency_mode="independent", frequency_change_pct=float(v))),
    "load_factor": ("Load factor", "pp", [-10, -5, 5, 10], lambda v: dict(load_factor_mode="delta", load_factor_value=float(v))),
    "p2p_share": ("P2P passenger share", "pp", [-10, -5, 5, 10], lambda v: dict(p2p_share_mode="delta", p2p_share_value=float(v))),
}


def sensitivity(model_id: str, market: str, start: str, end: str, a: Analytic, sustained: bool = True) -> pd.DataFrame:
    """Change ONE scenario input at a time and re-predict with the saved deployment pipeline.

    The baseline frame is built once; every variant rebuilds features from the modified route-day rows
    with the training feature code (exactly what S.run does). Invalid variants (e.g. a load factor above
    the hard limit) are reported as not valid rather than silently clipped."""
    md = registry.load_metadata(model_id)
    by_market = md["scope"] == "international_market"
    dates = pd.date_range(start, end, freq="D")
    markets = md.get("markets") if by_market else None
    rows = a.flight_daily_rows
    base_frame = S._frames(a, md["scope"], rows, dates, markets)
    sel = np.ones(len(base_frame), bool) if (not by_market or market == "ALL") else (base_frame["market_key"] == market).to_numpy()
    yb = S._predict(model_id, base_frame)[sel].sum() / len(dates)
    out = []
    for key, (label, unit, levels, kw) in SENSITIVITY_INPUTS.items():
        for v in levels:
            cfg = S.ScenarioConfig(model_id=model_id, market=market, start=start, end=end, sustained=sustained, **kw(v))
            rec = {"input": key, "label": label, "unit": unit, "change": v, "baseline": yb}
            try:
                S.validate_config(cfg, md, a)
                mod, mask = S.apply_scenario(rows, cfg)
                checks = S.consistency_checks(mod, mask, float(rows["Load Factor"].max()))
                if not all(c["passed"] for c in checks[:5]):
                    raise S.ScenarioError("; ".join(c["check"] for c in checks[:5] if not c["passed"]))
                ys = S._predict(model_id, S._frames(a, md["scope"], mod, dates, markets))[sel].sum() / len(dates)
                rec.update(scenario=ys, diff=ys - yb, pct=(ys / yb - 1) * 100 if yb else np.nan, valid=True, reason="")
            except S.ScenarioError as e:
                rec.update(scenario=np.nan, diff=np.nan, pct=np.nan, valid=False, reason=str(e))
            out.append(rec)
    return pd.DataFrame(out)


def sensitivity_ranking(sens: pd.DataFrame) -> pd.DataFrame:
    v = sens[sens["valid"]]
    g = v.groupby(["input", "label", "unit"]).agg(low=("pct", lambda x: x.iloc[0]), high=("pct", lambda x: x.iloc[-1]),
                                                   span=("pct", lambda x: float(x.max() - x.min())), max_abs=("pct", lambda x: float(x.abs().max())))
    return g.reset_index().sort_values("span", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ market & seasonal context

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def observed_guest_series(a: Analytic, md: dict, market: str) -> pd.Series:
    """Observed daily Guests for the population the model predicts (training files only)."""
    if md["scope"] == "international_market":
        hm = a.hotel_market[(a.hotel_market["split"] == "train")]
        if market != "ALL":
            hm = hm[hm["market_key"] == market]
        else:
            hm = hm[hm["market_key"].isin(md.get("markets") or [])]
        return hm.groupby("Date")["Guests"].sum()
    col = {"international_total": "intl_guests", "domestic": "dom_guests", "overall": "overall_guests"}[md["scope"]]
    ht = a.hotel_total[a.hotel_total["split"] == "train"]
    return ht.set_index("Date")[col]


def market_season(a: Analytic, md: dict, market: str, start: str, end: str) -> dict:
    g = observed_guest_series(a, md, market)
    by_month = g.groupby(g.index.month).mean().reindex(range(1, 13))
    years = g.groupby(g.index.month).apply(lambda x: x.index.year.nunique()).reindex(range(1, 13))
    by_dow = g.groupby(g.index.dayofweek).mean().reindex(range(7))
    rows = a.flight_daily_rows
    sel = rows if market == "ALL" else rows[rows["market_key"] == market]
    seats_m = sel.groupby(sel["Date"].dt.month)["Total Seats"].sum() / sel.groupby(sel["Date"].dt.month)["Date"].nunique()
    seats_m = seats_m.reindex(range(1, 13))
    dates = pd.date_range(start, end, freq="D")
    months = sorted(set(dates.month))
    period_mean = float(by_month.loc[months].mean()) if len(months) else float("nan")
    annual = float(by_month.mean())
    rank = int(by_month.rank(ascending=False).loc[months].mean()) if len(months) == 1 else None
    win = rows[rows["Date"].between(dates.min(), dates.max())]
    share = float(win.loc[win["market_key"] == market, "Total Seats"].sum() / win["Total Seats"].sum()) if market != "ALL" and win["Total Seats"].sum() else None
    acc = None
    if md["scope"] == "international_market" and market != "ALL":
        h = registry.load_table(md["model_id"], "predictions_holdout")
        h = h[h["market_key"] == market]
        if len(h) >= 30:
            acc = metrics_from_predictions(h)
    return {"population": population_label(md, market), "by_month": by_month, "years_per_month": years, "by_weekday": by_dow,
            "seats_by_month": seats_m, "months": months, "period_mean": period_mean, "annual_mean": annual,
            "period_vs_annual": period_mean / annual - 1 if annual else float("nan"), "month_rank": rank, "market_seat_share": share,
            "market_holdout": acc, "obs_start": g.index.min(), "obs_end": g.index.max()}


# ------------------------------------------------------------------ reproducible scenario record

def scenario_record(res: S.ScenarioResult, md: dict, sens: pd.DataFrame | None = None) -> dict:
    hold = registry.load_table(md["model_id"], "predictions_holdout")
    m = metrics_from_predictions(hold)
    f = config.RAW_DIR / "SHA256SUMS.txt"
    return {
        "scenario_id": res.scenario_id,
        "model": {"model_id": md["model_id"], "version": md["created_at"], "algorithm": md["algorithm_label"], "scope": md["scope"],
                  "grain": md.get("grain"), "mode": md["mode"], "reference": bool(md.get("reference")),
                  "holdout": md["periods"]["holdout"], "holdout_metrics": {k: m[k] for k in ("MAE", "RMSE", "R2", "WMAPE")}},
        "feature_pipeline_version": pipeline_version(),
        "population": population_label(md, res.config["market"]),
        "config": res.config,
        "baseline_aviation": res.aviation_changes.set_index("quantity")["baseline"].to_dict(),
        "scenario_aviation": res.aviation_changes.set_index("quantity")["scenario"].to_dict(),
        "changed_features": res.feature_changes[res.feature_changes["changed"]].to_dict("records") if len(res.feature_changes) else [],
        "summary": res.summary,
        "daily": res.daily.assign(Date=res.daily["Date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
        "sensitivity": sens.to_dict("records") if sens is not None else None,
        "checks": res.checks, "warnings": res.warnings,
        "provenance": {"data_sha256sums": hashlib.sha256(f.read_bytes()).hexdigest()[:16] if f.exists() else None,
                       "source_files": "six supplied competition files (read-only copies)", "steps": res.steps},
    }


def save_record(record: dict) -> Path:
    d = config.SCENARIOS_DIR / record["scenario_id"]
    d.mkdir(parents=True, exist_ok=True)
    p = d / "scenario_record.json"
    p.write_text(json.dumps(record, indent=2, default=str))
    return p
