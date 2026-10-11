"""Section 7 — What-if Aviation Simulator.

Layout (reference design): model · market & period · controls → four result cards → chart panel with
four tabs (demand over time, sensitivity, conversion chain, market & season) → key insights and an
expandable scenario audit. Every number comes from the saved pipelines via quanara.scenario and
quanara.science; nothing is hard-coded.
"""
from __future__ import annotations

import json
from collections import Counter
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import config, registry
from .. import model_identity as MI
from .. import scenario as S
from .. import science as SC
from ..evaluation import metrics_from_predictions
from . import design, heroes
from .common import active_id, analytic, csv_button, next_step, plot, png_button
from .design import ICON, style_fig

BASE_C, SCEN_C, MUTED, SOFT = "#4b5563", "#059669", "#a0a4a2", "#c9dbd2"  # baseline, scenario, observed, context
SCOPE_TAG = {"international_market": "Intl · by market", "international_total": "Intl · total", "domestic": "Domestic", "overall": "Overall"}
SCOPE_SHORT = {"international_market": "International · by source market (daily × nationality)", "international_total": "International · all nationalities (daily total)",
               "domestic": "Domestic (daily total)", "overall": "Overall · international + domestic (daily total)"}
KIND = {"observed": ("Observed", "blue"), "derived": ("Derived", "teal"), "predicted": ("Predicted", "violet"), "unavailable": ("Not estimable", "grey")}

CSS = """
<style>
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-sim_c_"]) { height:100%; }
[data-testid="stLayoutWrapper"]:has(> .st-key-sim_top), .st-key-sim_top, .st-key-sim_top > [data-testid="stLayoutWrapper"],
.st-key-sim_top > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"] { height:100%; }
.st-key-sim_top [data-testid="stHorizontalBlock"] { align-items:stretch; }
.st-key-sim_top [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { height:100%; }
[class*="st-key-sim_c_"], .st-key-sim_results, .st-key-sim_tabs, .st-key-sim_insights {
  background:#fff; border:1px solid #e7e8e7; border-radius:18px; padding:15px 17px 13px; gap:.55rem;
  box-shadow:0 1px 2px rgba(15,23,20,.04),0 10px 26px rgba(15,23,20,.06); animation:dsFadeUp .5s ease both; }
[class*="st-key-sim_c_"] { height:100%; }
.st-key-sim_c_market { animation-delay:.05s; } .st-key-sim_c_controls { animation-delay:.1s; } .st-key-sim_results { animation-delay:.12s; }
[class*="st-key-sim_"] [data-testid="stMarkdownContainer"] { margin-bottom:0 !important; }
.sm-h { display:flex; gap:11px; align-items:center; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.sm-h b { font-size:15px; color:#101614; font-weight:780; } .sm-h .ds-ico { width:36px; height:36px; border-radius:11px; background:#059669; color:#fff; }
.sm-h .ds-ico svg { width:19px; height:19px; }
[class*="st-key-sim_c_"] [data-baseweb="select"] div { font-size:13.2px; }
.sm-sub { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.4px; color:#6f7572; line-height:1.4; }
.st-key-sim_c_model { container-type:inline-size; }
.sm-tiles { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:7px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.sm-tiles div { border:1px solid #eceeed; border-radius:11px; padding:8px 9px; background:#f8fdfb; min-width:0; }
.sm-tiles span { display:block; font-size:11.2px; color:#515553; font-weight:650; }
.sm-tiles b { display:block; font-size:17px; color:#101614; font-weight:800; white-space:nowrap; letter-spacing:-.01em; }
@container (max-width: 330px) { .sm-tiles { grid-template-columns:repeat(2,minmax(0,1fr)); } .sm-tiles b { font-size:18px; } }
.sm-base { display:flex; gap:12px; align-items:center; background:#ecfdf5; border:1px solid #bbf0d6; border-radius:12px; padding:11px 13px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.sm-base svg { width:30px; height:30px; color:#059669; flex:none; } .sm-base b { font-size:13px; color:#101614; } .sm-base span { font-size:12.2px; color:#444846; line-height:1.5; display:block; }
.sm-res { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.sm-r { border:1px solid #eceeed; border-radius:14px; padding:12px 14px; display:flex; gap:12px; align-items:center; background:#fff; min-width:0; }
.sm-r .ds-ico { width:40px; height:40px; flex:none; } .sm-r span { display:block; font-size:12.2px; color:#515553; font-weight:650; }
.sm-r b { display:block; font-size:23px; font-weight:800; color:#101614; line-height:1.2; letter-spacing:-.01em; animation:fePop .45s cubic-bezier(.3,1.4,.6,1); }
.sm-r small { display:block; font-size:11.6px; color:#6f7572; }
.sm-r.empty b { color:#a0a4a2; font-size:16px; }
@keyframes fePop { from { transform:scale(.85); opacity:.25; } to { transform:none; opacity:1; } }
.sm-ins { display:flex; gap:11px; border-radius:13px; padding:11px 13px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; margin-bottom:9px; }
.sm-ins svg { width:20px; height:20px; flex:none; margin-top:1px; } .sm-ins b { display:block; font-size:13.4px; margin-bottom:2px; }
.sm-ins p { margin:0; font-size:12.5px; color:#444846; line-height:1.48; }
.sm-ins.green { background:#effcf5; border:1px solid #d3f0e1; } .sm-ins.green b, .sm-ins.green svg { color:#047857; }
.sm-ins.blue { background:#ecfdf5; border:1px solid #bbf0d6; } .sm-ins.blue b, .sm-ins.blue svg { color:#047857; }
.sm-ins.violet { background:#f0fdf6; border:1px solid #c6f0dc; } .sm-ins.violet b, .sm-ins.violet svg { color:#065f46; }
.sm-ins.grey { background:#f0f9f6; border:1px solid #eceeed; } .sm-ins.grey b, .sm-ins.grey svg { color:#515553; }
.st-key-sim_run button { min-height:48px; font-size:16px; } .st-key-sim_run button p { font-size:16px; font-weight:750; }
.st-key-sim_tabs [role="tablist"] { gap:4px; background:#ecf9f4; border-radius:12px; padding:4px; flex-wrap:wrap; overflow:visible !important; }
.st-key-sim_tabs [role="tablist"]::after, .st-key-sim_tabs [role="tablist"]::before { display:none !important; }
.st-key-sim_tabs [data-testid="stTab"] { height:38px; padding:0 16px; border-radius:9px; border:none !important; }
.st-key-sim_tabs [data-testid="stTab"] p { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-weight:650; font-size:13.6px; color:#313432; }
.st-key-sim_tabs [data-testid="stTab"][aria-selected="true"] { background:#059669; box-shadow:0 4px 12px rgba(5,150,105,.25); }
.st-key-sim_tabs [data-testid="stTab"][aria-selected="true"] p { color:#fff; }
.st-key-sim_tabs [role="tabpanel"] { animation:dsFadeUp .4s ease both; }
.sm-ct { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:15.5px; font-weight:780; color:#101614; margin:4px 0 0; }
.sm-note { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.3px; color:#6f7572; line-height:1.5; } .sm-note b { color:#313432; }
.sm-warn { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.4px; line-height:1.45; border-radius:10px; padding:8px 11px; display:flex; gap:8px;
           background:#fff7e8; color:#8a5a00; border:1px solid #fbe3b4; margin:4px 0; } .sm-warn svg { width:15px; height:15px; flex:none; margin-top:1px; }
.sm-chain { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; margin-top:6px; }
.sm-st { border:1px solid #eceeed; border-radius:12px; padding:10px 11px; background:#f8fdfb; min-width:0; display:flex; flex-direction:column; gap:4px; }
.sm-st.unavailable { background:repeating-linear-gradient(135deg,#f4f9f7,#f4f9f7 6px,#f5f7f6 6px,#f5f7f6 12px); }
.sm-st .n { font-size:11px; color:#6f7572; font-weight:700; } .sm-st .t { font-size:13px; color:#101614; font-weight:750; line-height:1.25; }
.sm-st .v { font-size:15px; font-weight:800; color:#101614; } .sm-st .v small { font-size:11px; color:#6f7572; font-weight:600; display:block; }
.sm-st .d { font-size:11.4px; color:#515553; line-height:1.4; }
.sm-k { display:inline-block; font-size:10.5px; font-weight:800; letter-spacing:.04em; border-radius:6px; padding:1px 6px; align-self:flex-start; }
.sm-k.blue { color:#047857; background:#ecfdf5; } .sm-k.teal { color:#0f7f74; background:#e1f5f2; } .sm-k.violet { color:#065f46; background:#ecfdf5; } .sm-k.grey { color:#606562; background:#eaebea; }
.sm-facts { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; margin:6px 0; }
.sm-facts div { background:#f0f9f6; border-radius:10px; padding:7px 10px; } .sm-facts span { display:block; font-size:11px; color:#6f7572; } .sm-facts b { font-size:14px; color:#101614; }
.st-key-sim_results { container-type:inline-size; }
@container (max-width: 760px) { .sm-res { grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width: 1100px) {
  .st-key-sim_main > div > [data-testid="stHorizontalBlock"], .st-key-sim_top [data-testid="stHorizontalBlock"] { flex-wrap:wrap; row-gap:12px; }
  .st-key-sim_main > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"], .st-key-sim_top [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex:1 1 100% !important; min-width:100% !important; }
  .sm-facts { grid-template-columns:repeat(2,minmax(0,1fr)); }
}
@media (max-width: 640px) { .sm-chain { grid-template-columns:repeat(2,minmax(0,1fr)); } }
</style>
"""


# ------------------------------------------------------------------ helpers

def _head(icon: str, title: str) -> None:
    design.html(f'<div class="sm-h"><div class="ds-ico">{ICON[icon]}</div><b>{escape(title)}</b></div>')


def _months() -> list[tuple[str, str, str]]:
    lo, hi = S.supported_period()
    ms = pd.period_range(lo.to_period("M") + (1 if lo.day > 1 else 0), hi.to_period("M"), freq="M")
    return [(m.strftime("%b %Y") + ("  · forecast" if m.start_time > config.HOTEL_TRAIN_END else ""), str(m.start_time.date()), str(m.end_time.date())) for m in ms]


@st.cache_data(show_spinner=False)
def _baseline(market: str, start: str, end: str) -> dict:
    rows = analytic().flight_daily_rows
    m = rows["Date"].between(pd.Timestamp(start), pd.Timestamp(end))
    if market != "ALL":
        m &= rows["market_key"].eq(market)
    x = rows[m]
    seats, pax, p2p, fl = (float(x[c].sum()) for c in ("Total Seats", "Total PAX", "Total P2P", "flights"))
    return {"seats": seats, "flights": fl, "lf": pax / seats if seats else np.nan, "p2p_share": p2p / pax if pax else np.nan,
            "routes": int(x.groupby(["Departure City", "Airline Name"]).ngroups), "days": int(x["Date"].nunique())}


@st.cache_data(show_spinner="Re-predicting one input at a time with the saved model …")
def _sensitivity(model_id: str, market: str, start: str, end: str, sustained: bool, stamp: float) -> pd.DataFrame:
    return SC.sensitivity(model_id, market, start, end, analytic(), sustained)


@st.cache_data(show_spinner=False)
def _market_season(model_id: str, market: str, start: str, end: str) -> dict:
    return SC.market_season(analytic(), registry.load_metadata(model_id), market, start, end)


def _switch() -> None:
    registry.activate(st.session_state["sim_model"])
    st.session_state.pop("last_scenario", None)
    st.session_state.pop("scenario_record", None)


def _num(v, d=0) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:,.{d}f}"


# ------------------------------------------------------------------ page

def render() -> None:
    design.inject()
    st.markdown(heroes.CSS + CSS, unsafe_allow_html=True)
    design.html(heroes.banner("simulator", "Section 7", "What-if Aviation Simulator",
                              "Explore how changes in flight operations affect predicted hotel demand in Abu Dhabi."))
    st.write("")
    a = analytic()
    wi = [m for m in registry.list_models() if MI.scenario_ready(m)]
    if not wi:
        design.html(f'<div class="sm-warn">{ICON["warning"]}<span>No What-if model is saved yet. Train one in the ML Training Studio (mode: What-if).</span></div>')
        return
    ids = [m["model_id"] for m in wi]
    act = active_id()
    if act not in ids:
        registry.activate(ids[0])
        act = ids[0]
    if st.session_state.get("sim_model") not in ids:
        st.session_state["sim_model"] = act

    with st.container(key="sim_main"):
        left, right = st.columns([2.05, 1], gap="medium")
        with left:
            with st.container(key="sim_top"):
                c1, c2 = st.columns([1.3, 1], gap="medium")  # wider model card so the model name is not cut off
                # ---------------------------------------------------- A · model
                with c1, st.container(key="sim_c_model"):
                    _head("trend", "Active predictive model")
                    lab = MI.labels(wi)
                    short = {k: v.replace(" · reference", "") for k, v in lab.items()}
                    n_short = Counter(short.values())
                    lab = {k: short[k] if n_short[short[k]] == 1 else v for k, v in lab.items()}  # keep the suffix only where it disambiguates
                    mid = st.selectbox("Model", ids, key="sim_model", format_func=lambda x: lab[x], on_change=_switch, label_visibility="collapsed",
                                       help="Selecting a model makes it the active What-if model for the whole app.")
                    md = registry.load_metadata(mid)
                    hm = metrics_from_predictions(registry.load_table(mid, "predictions_holdout"))
                    design.html(f'<div class="sm-sub">{escape(MI.target(md))} · {escape(MI.scope(md).lower())} · {escape(MI.validation(md))}</div>'
                                '<div class="sm-tiles">'
                                f'<div><span>WMAPE</span><b>{hm["WMAPE"]:.1%}</b></div><div><span>R²</span><b>{hm["R2"]:.3f}</b></div>'
                                f'<div><span>MAE</span><b>{hm["MAE"]:,.0f}</b></div><div><span>RMSE</span><b>{hm["RMSE"]:,.0f}</b></div></div>'
                                f'<div class="sm-sub">Scored on {hm["n"]:,} unseen {MI.row_unit(md)} · MAE and RMSE in {MI.unit(md)}.</div>')
                    with st.expander("View model details", icon=":material/description:"):
                        st.markdown("\n".join(f"- **{k}:** {v}" for k, v in MI.facts(md).items()) +
                                    f"\n- **Trained:** {md['periods']['eval_fit'][0]} – {md['periods']['eval_fit'][1]}; refit to "
                                    f"{md['periods']['deployment_fit'][1]} for scenarios\n- **Features:** {len(md['features'])}"
                                    f"{' · monotone constraint on volume features' if md.get('monotone') else ''}\n- **Metrics:** recomputed from the saved holdout "
                                    f"predictions ({hm['n']:,} {MI.row_unit(md)}).\n- **Model ID:** `{mid}`")
                # ---------------------------------------------------- B · market & period
                with c2, st.container(key="sim_c_market"):
                    _head("globe", "Scenario market & period")
                    by_market = md["scope"] == "international_market"
                    markets = (md.get("markets") or []) if by_market else a.markets
                    opts = ["ALL"] + list(markets)
                    if st.session_state.get("sim_market") not in opts:
                        st.session_state["sim_market"] = "RUSSIAN FEDERATION" if "RUSSIAN FEDERATION" in opts else "ALL"
                    market = st.selectbox("Source market (departure country)", opts, key="sim_market",
                                          format_func=lambda m: "All departure countries" if m == "ALL" else m.title())
                    months = _months()
                    labels = [x[0] for x in months] + ["Custom date range…"]
                    if st.session_state.get("sim_period") not in labels:
                        st.session_state["sim_period"] = next((x[0] for x in months if x[1] == "2025-09-01"), labels[-2])
                    per = st.selectbox("Scenario period", labels, key="sim_period")
                    if per.startswith("Custom"):
                        lo, hi = S.supported_period()
                        dr = st.date_input("Dates", (pd.Timestamp("2025-12-01").date(), pd.Timestamp("2025-12-31").date()), min_value=lo.date(), max_value=hi.date(),
                                           key="sim_custom", label_visibility="collapsed")
                        start, end = (str(dr[0]), str(dr[1])) if isinstance(dr, (list, tuple)) and len(dr) == 2 else (str(lo.date()), str(lo.date()))
                    else:
                        _, start, end = months[labels.index(per)]
                    bp = _baseline(market, start, end)
                    design.html(f'<div class="sm-base">{ICON["bars"]}<div><b>Baseline aviation</b> <span style="display:inline;color:#6f7572">'
                                f'({pd.Timestamp(start):%d %b} – {pd.Timestamp(end):%d %b %Y})</span>'
                                f'<span>{bp["seats"]:,.0f} seats · {bp["flights"]:,.0f} flights · {bp["routes"]} routes</span>'
                                f'<span>Load factor {bp["lf"]:.1%} · P2P share {bp["p2p_share"]:.1%}</span></div></div>')
                    if not by_market and market != "ALL":
                        design.html(f'<div class="sm-sub">This model predicts {escape(SC.population_label(md, market))}; {escape(market.title())} flights change '
                                    'its aviation inputs, but the output is not a country-level prediction.</div>')

        # ---------------------------------------------------------------- C · controls
        with right, st.container(key="sim_c_controls"):
            _head("sliders", "Aviation scenario controls")
            seats = st.slider("Seat capacity change", -50, 50, 20, 5, format="%+d%%", key="sim_seats")
            same = st.checkbox("Same aircraft size (frequency follows seats)", value=True, key="sim_same")
            freq = st.slider("Flight frequency change", -50, 50, int(seats) if same else 0, 5, format="%+d%%", key=f"sim_freq_{int(same)}_{seats if same else ''}",
                             disabled=same, help="When unlinked, seats per flight absorb the difference so total seats change exactly as set (never double-counted).")
            freq = seats if same else freq
            spf = ((1 + seats / 100) / (1 + freq / 100) - 1) * 100
            lf0 = round(bp["lf"] * 100, 1) if np.isfinite(bp["lf"]) else 80.0
            sh0 = round(bp["p2p_share"] * 100, 1) if np.isfinite(bp["p2p_share"]) else 50.0
            lf = st.slider("Load factor", min(40.0, lf0), max(105.0, lf0), lf0, 0.5, format="%.1f%%", key=f"sim_lf_{market}_{start}_{end}",
                           help="Average across affected route-days. A change is applied as a shift in percentage points to every route-day.")
            sh = st.slider("P2P passenger share", min(5.0, sh0), max(95.0, sh0), sh0, 0.5, format="%.1f%%", key=f"sim_sh_{market}_{start}_{end}",
                           help="Share of arriving passengers whose journey ends in Abu Dhabi; transfer = PAX − P2P − transit adjusts automatically.")
            sustained = st.checkbox("Change already in effect before the period (sustained)", value=True, key="sim_sust",
                                    help="Applies the change over the 56-day feature look-back so trailing 7/28-day features reflect the new level.")
            design.html(f'<div class="sm-sub">Derived: flights {freq:+.0f}% · seats per flight {spf:+.1f}% · total seats {seats:+.0f}%, applied once.</div>')
            lf_mode, lf_val = ("baseline", 0.0) if abs(lf - lf0) < 1e-9 else ("delta", lf - bp["lf"] * 100)
            sh_mode, sh_val = ("baseline", 0.0) if abs(sh - sh0) < 1e-9 else ("delta", sh - bp["p2p_share"] * 100)
            cfg = S.ScenarioConfig(model_id=mid, market=market, start=start, end=end, seats_change_pct=float(seats),
                                   frequency_mode="follow_seats" if same else "independent", frequency_change_pct=float(freq),
                                   load_factor_mode=lf_mode, load_factor_value=float(lf_val), p2p_share_mode=sh_mode, p2p_share_value=float(sh_val), sustained=sustained)
            run = st.button("Run Simulation", type="primary", icon=":material/play_arrow:", key="sim_run", width="stretch")

    if run:
        with st.spinner("Rebuilding features and predicting with the saved model …"):
            try:
                res = S.run(cfg, a)
                st.session_state["last_scenario"] = res
                sens = _sensitivity(mid, market, start, end, sustained, registry.model_dir(mid).stat().st_mtime)
                st.session_state["sim_sens_for"] = (res.scenario_id, sens)
                rec = SC.scenario_record(res, md, sens)
                SC.save_record(rec)
                st.session_state["scenario_record"] = rec
            except S.ScenarioError as e:
                st.session_state["sim_error"] = str(e)
            except Exception as e:  # surface real errors
                st.session_state["sim_error"] = f"{type(e).__name__}: {e}"
    err = st.session_state.pop("sim_error", None)
    res = st.session_state.get("last_scenario")
    if res is not None and res.config.get("model_id") != mid:
        res = None
    stale = res is not None and res.config != {k: v for k, v in cfg.__dict__.items()}

    # ---------------------------------------------------------------- results + tabs (left) / insights (right)
    with st.container(key="sim_lower"):
        left2, right2 = st.columns([2.05, 1], gap="medium")
        with left2:
            with st.container(key="sim_results"):
                pop = SC.population_label(md, res.config["market"] if res else market)
                _head("target", "Predicted hotel demand")
                design.html(f'<div class="sm-sub">{escape(pop[:1].upper() + pop[1:])} · average per day over the scenario period · model output</div>')
                if err:
                    design.html(f'<div class="sm-warn">{ICON["warning"]}<span>Scenario rejected: {escape(err)}</span></div>')
                if res is None:
                    design.html('<div class="sm-res">' + "".join(
                        f'<div class="sm-r empty"><div class="ds-ico {t}">{ICON[i]}</div><div><span>{n}</span><b>Run the simulation</b></div></div>'
                        for n, i, t in (("Baseline", "home", "blue"), ("Scenario", "chart", "teal"), ("Difference", "trend", "orange"), ("Change", "scale", "violet"))) + "</div>")
                else:
                    s = res.summary
                    design.html('<div class="sm-res">'
                                f'<div class="sm-r"><div class="ds-ico blue">{ICON["home"]}</div><div><span>Baseline</span><b>{_num(s["baseline_avg_daily_guests"])}</b><small>guests/day</small></div></div>'
                                f'<div class="sm-r"><div class="ds-ico teal">{ICON["chart"]}</div><div><span>Scenario</span><b>{_num(s["scenario_avg_daily_guests"])}</b><small>guests/day</small></div></div>'
                                f'<div class="sm-r"><div class="ds-ico orange">{ICON["trend"]}</div><div><span>Difference</span><b>{s["abs_change_avg_daily"]:+,.0f}</b><small>guests/day</small></div></div>'
                                f'<div class="sm-r"><div class="ds-ico violet">{ICON["scale"]}</div><div><span>Change</span><b>{s["pct_change"]:+.1f}%</b><small>vs baseline</small></div></div></div>')
                    if stale:
                        design.html(f'<div class="sm-warn">{ICON["info"]}<span>The controls changed after this result — press Run Simulation to update it.</span></div>')
            st.write("")
            with st.container(key="sim_tabs"):
                tabs = st.tabs(["Demand over time", "Sensitivity analysis", "Conversion chain", "Market & seasonal view"], key="sim_tabnav", on_change="rerun")
                if res is None:
                    with tabs[0]:
                        design.html('<div class="sm-note" style="padding:18px 4px">Set the scenario and press <b>Run Simulation</b>. Results appear here.</div>')
                else:
                    if tabs[0].open:
                        with tabs[0]:
                            _tab_demand(res, md)
                    if tabs[1].open:
                        with tabs[1]:
                            _tab_sensitivity(res, md)
                    if tabs[2].open:
                        with tabs[2]:
                            _tab_chain(res, md)
                    if tabs[3].open:
                        with tabs[3]:
                            _tab_market(res, md)
        with right2:
            with st.container(key="sim_insights"):
                _head("bulb", "Key insights")
                if res is None:
                    design.html(f'<div class="sm-ins grey">{ICON["info"]}<div><b>Waiting for a scenario</b><p>Insights are generated from the simulation results.</p></div></div>')
                else:
                    for title, text, tone, icon in _insights(res, md):
                        design.html(f'<div class="sm-ins {tone}">{ICON[icon]}<div><b>{escape(title)}</b><p>{escape(text)}</p></div></div>')
                    if st.button("Scenario audit & technical details", icon=":material/fact_check:", key="sim_audit_btn", width="stretch"):
                        _audit(res, md, a)
    next_step("sim")


# ------------------------------------------------------------------ tabs

def _focus(res, md) -> pd.DataFrame:
    d = res.daily
    if md["scope"] == "international_market":
        if res.config["market"] != "ALL":
            d = d[d["market_key"] == res.config["market"]]
        d = d.groupby("Date", as_index=False)[["baseline", "scenario", "observed"]].sum(min_count=1)
    return d.sort_values("Date")


def _demand_fig(res, md) -> go.Figure:
    d = _focus(res, md)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=d["baseline"], mode="lines", name="Baseline prediction", line=dict(color=BASE_C, width=2, dash="dash"),
                             fill="tozeroy", fillcolor="rgba(75,85,99,0.05)", hovertemplate="%{x|%d %b %Y}: %{y:,.0f}<extra>baseline</extra>"))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["scenario"], mode="lines", name="Scenario prediction", line=dict(color=SCEN_C, width=2.4),
                             fill="tonexty", fillcolor="rgba(5,150,105,0.12)", hovertemplate="%{x|%d %b %Y}: %{y:,.0f}<extra>scenario</extra>"))
    if d["observed"].notna().any():
        fig.add_trace(go.Scatter(x=d["Date"], y=d["observed"], mode="lines", name="Observed Guests", line=dict(color=MUTED, width=1.5, dash="dot"),
                                 hovertemplate="%{x|%d %b %Y}: %{y:,.0f}<extra>observed</extra>"))
    style_fig(fig, height=330)
    fig.update_yaxes(title_text="Guests per day", rangemode="tozero", separatethousands=True)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.06, x=1, xanchor="right"))
    return fig


def _tab_demand(res, md) -> None:
    design.html(f'<div class="sm-ct">Predicted hotel guests — baseline vs scenario</div>'
                f'<div class="sm-note">{escape((lambda t: t[:1].upper() + t[1:])(SC.population_label(md, res.config["market"])))}, daily model predictions.</div>')
    plot(_demand_fig(res, md), key="sim_demand")
    d = _focus(res, md)
    diff = d["scenario"] - d["baseline"]
    f = go.Figure(go.Bar(x=d["Date"], y=diff, marker=dict(color=[SCEN_C if v >= 0 else BASE_C for v in diff], cornerradius=3),
                         hovertemplate="%{x|%d %b %Y}: %{y:+,.1f} guests<extra></extra>"))
    style_fig(f, height=150, legend=False)
    f.update_yaxes(title_text="Δ guests", separatethousands=True, zeroline=True, zerolinecolor="#cfd1d0")
    f.update_layout(margin=dict(t=8, b=8))
    design.html('<div class="sm-note"><b>Daily difference</b> (scenario − baseline), same units — shown separately so small changes stay visible.</div>')
    plot(f, key="sim_demand_diff")


def _sens_for(res) -> pd.DataFrame:
    sid, sens = st.session_state.get("sim_sens_for", (None, None))
    if sid != res.scenario_id or sens is None:
        c = res.config
        sens = _sensitivity(c["model_id"], c["market"], c["start"], c["end"], c["sustained"], registry.model_dir(c["model_id"]).stat().st_mtime)
        st.session_state["sim_sens_for"] = (res.scenario_id, sens)
    return sens


def _tab_sensitivity(res, md) -> None:
    sens = _sens_for(res)
    rk = SC.sensitivity_ranking(sens)
    design.html('<div class="sm-ct">Model-based sensitivity — one input at a time</div>'
                '<div class="sm-note">Each input is changed alone and the saved model re-predicts the same dates and population. Ranked by the spread of the '
                'predicted change. These are model responses, not causal effects.</div>')
    fig = go.Figure()
    lab = [f"{r.label} (±{20 if r.unit == '%' else 10}{'%' if r.unit == '%' else ' pp'})" for r in rk.itertuples()]
    fig.add_trace(go.Bar(y=lab, x=rk["low"], orientation="h", name="Decrease", marker=dict(color=BASE_C, cornerradius=4),
                         hovertemplate="%{y}<br>decrease: %{x:+.2f}%<extra></extra>"))
    fig.add_trace(go.Bar(y=lab, x=rk["high"], orientation="h", name="Increase", marker=dict(color=SCEN_C, cornerradius=4),
                         hovertemplate="%{y}<br>increase: %{x:+.2f}%<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#cfd1d0", width=1))
    style_fig(fig, height=max(220, 56 * len(rk) + 70))
    fig.update_layout(barmode="overlay", bargap=0.35, legend=dict(orientation="h", y=1.08, x=0))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Predicted change in guests per day (%)", ticksuffix="%")
    plot(fig, key="sim_sens_fig")
    t = sens.copy()
    t["input change"] = t.apply(lambda r: f"{r['change']:+.0f}{'%' if r['unit'] == '%' else ' pp'}", axis=1)
    design.html(f'<div class="sm-note">Baseline for every row: <b>{t["baseline"].iloc[0]:,.0f}</b> guests/day.</div>')
    st.dataframe(t[["label", "input change", "scenario", "diff", "pct", "valid"]], hide_index=True, width="stretch", height=260, column_config={
        "label": st.column_config.TextColumn("Input", width="medium"), "input change": st.column_config.TextColumn("Change"),
        "scenario": st.column_config.NumberColumn("Scenario / day", format="%,.0f"), "diff": st.column_config.NumberColumn("Difference", format="%+,.0f"),
        "pct": st.column_config.NumberColumn("Change %", format="%+.2f"), "valid": st.column_config.CheckboxColumn("Valid")})
    nonmono = [r.label for r in rk.itertuples() if (sens[(sens["input"] == r.input) & sens["valid"]].sort_values("change")["pct"].diff().dropna() < 0).any()]
    if nonmono:
        design.html(f'<div class="sm-warn">{ICON["info"]}<span>Non-monotonic response for {escape(", ".join(nonmono))}: tree models predict in steps, so small input changes '
                    'can cross split thresholds in either direction.</span></div>')
    design.html('<div class="sm-note">Seat capacity moves flights with it (same aircraft size); frequency changes keep total seats fixed (aircraft size absorbs the '
                'difference); load factor and P2P share shift every affected route-day in percentage points. Inputs that would breach a constraint are marked not valid.</div>')
    csv_button(t, f"{res.scenario_id}_sensitivity.csv", "Download sensitivity (CSV)", key="sim_dl_sens")


def _tab_chain(res, md) -> None:
    chain = SC.conversion_chain(res, md)
    design.html('<div class="sm-ct">Aviation-to-hospitality pathway</div>'
                '<div class="sm-note">Observed and derived stages come from the flight file; hotel guests are the model\'s prediction. The model links aviation '
                'features to Guests statistically — it does not estimate the unobserved stages, and no conversion rate is implied.</div>')
    cards = []
    for i, c in enumerate(chain):
        word, tone = KIND[c["kind"]]
        if c["baseline"] is None:
            val = '<div class="v">—<small>not estimated</small></div>'
        else:
            val = f'<div class="v">{c["baseline"]:,.0f} → {c["scenario"]:,.0f}<small>{escape(c["unit"])}</small></div>'
        cards.append(f'<div class="sm-st {c["kind"]}"><span class="n">{i + 1:02d}</span><span class="t">{escape(c["stage"])}</span><span class="sm-k {tone}">{word}</span>'
                     f'{val}<span class="d">{escape(c["note"])}</span></div>')
    design.html('<div class="sm-chain">' + "".join(cards) + "</div>")
    design.html('<div class="sm-note" style="margin-top:8px">Aviation values are totals for the selected departure market and period (baseline → scenario). '
                'Unavailable: visitor share of P2P passengers, hotel capture rate and length of stay.</div>')


def _tab_market(res, md) -> None:
    c = res.config
    ms = _market_season(c["model_id"], c["market"], c["start"], c["end"])
    mk = "all departure countries" if c["market"] == "ALL" else c["market"].title()
    design.html(f'<div class="sm-ct">Market & seasonal context</div><div class="sm-note">Guests: observed {escape(ms["population"])}, {ms["obs_start"]:%b %Y} – '
                f'{ms["obs_end"]:%b %Y}. Seats: flights from {escape(mk)} (departure country), daily rows from 2023.</div>')
    facts = [("Period vs annual mean", f"{ms['period_vs_annual']:+.0%}" if np.isfinite(ms["period_vs_annual"]) else "—"),
             ("Month rank (busiest = 1)", f"{ms['month_rank']} of 12" if ms["month_rank"] else "multi-month"),
             ("Market share of seats", f"{ms['market_seat_share']:.1%}" if ms["market_seat_share"] is not None else "all markets"),
             ("Market holdout WMAPE", f"{ms['market_holdout']['WMAPE']:.1%}" if ms["market_holdout"] else "model not by market")]
    design.html('<div class="sm-facts">' + "".join(f"<div><span>{escape(k)}</span><b>{escape(v)}</b></div>" for k, v in facts) + "</div>")
    a, b = st.columns(2, gap="small")
    bm = ms["by_month"]
    col = [SCEN_C if (i + 1) in ms["months"] else SOFT for i in range(12)]
    f1 = go.Figure(go.Bar(x=SC.MONTHS, y=bm.to_numpy(), marker=dict(color=col, cornerradius=4), customdata=ms["years_per_month"].to_numpy(),
                          hovertemplate="%{x}: %{y:,.0f} guests/day (mean of %{customdata} years)<extra></extra>"))
    style_fig(f1, height=250, legend=False)
    f1.update_yaxes(title_text="Guests / day", separatethousands=True)
    f1.update_xaxes(tickangle=0)
    f2 = go.Figure(go.Scatter(x=SC.MONTHS, y=ms["seats_by_month"].to_numpy(), mode="lines+markers", line=dict(color=BASE_C, width=2),
                              marker=dict(size=[11 if (i + 1) in ms["months"] else 7 for i in range(12)], color=[SCEN_C if (i + 1) in ms["months"] else BASE_C for i in range(12)]),
                              hovertemplate="%{x}: %{y:,.0f} seats/day<extra></extra>"))
    style_fig(f2, height=250, legend=False)
    f2.update_yaxes(title_text="Seats / day", separatethousands=True)
    f2.update_xaxes(tickangle=0)
    with a:
        design.html('<div class="sm-note"><b>Guests by calendar month</b> (green = scenario period)</div>')
        plot(f1, key="sim_ms_g")
    with b:
        design.html(f'<div class="sm-note"><b>Arriving seats by calendar month</b> · {escape(mk)}</div>')
        plot(f2, key="sim_ms_s")
    dow = ms["by_weekday"]
    design.html('<div class="sm-note">Weekday profile (observed guests/day): ' + " · ".join(f"{d} {v:,.0f}" for d, v in zip(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.to_numpy())) +
                '. Departure country is a proxy for nationality: flights from a country also carry other nationals, and nationals arrive via other hubs.</div>')


# ------------------------------------------------------------------ insights & audit

def _insights(res, md) -> list[tuple[str, str, str, str]]:
    c, s = res.config, res.summary
    ms = _market_season(c["model_id"], c["market"], c["start"], c["end"])
    pop = SC.population_label(md, c["market"])
    parts = [f"seats {c['seats_change_pct']:+.0f}%"]
    if c["frequency_mode"] == "independent":
        parts.append(f"frequency {c['frequency_change_pct']:+.0f}%")
    if c["load_factor_mode"] != "baseline":
        parts.append(f"load factor {c['load_factor_value']:+.1f} pp")
    if c["p2p_share_mode"] != "baseline":
        parts.append(f"P2P share {c['p2p_share_value']:+.1f} pp")
    up = s["abs_change_avg_daily"] >= 0
    none = not any(v for v in (c["seats_change_pct"], c["frequency_change_pct"] if c["frequency_mode"] == "independent" else 0)) and \
        c["load_factor_mode"] == "baseline" and c["p2p_share_mode"] == "baseline"
    if none:
        t1 = "No aviation input was changed, so the scenario reproduces the baseline prediction exactly — a check that the pipeline adds no drift."
    elif abs(s["pct_change"]) < 0.05:
        t1 = f"With {', '.join(parts)}, predicted {pop} are essentially unchanged ({s['abs_change_avg_daily']:+,.1f} per day)."
    else:
        t1 = f"With {', '.join(parts)}, predicted {pop} {'rise' if up else 'fall'} by {abs(s['abs_change_avg_daily']):,.0f} per day ({s['pct_change']:+.1f}%)."
    mon = ", ".join(SC.MONTHS[m - 1] for m in ms["months"][:3])
    t2 = (f"{mon} averages {ms['period_mean']:,.0f} observed guests/day for this population, {abs(ms['period_vs_annual']):.0%} "
          f"{'below' if ms['period_vs_annual'] < 0 else 'above'} the annual mean." if np.isfinite(ms["period_vs_annual"]) else "No observed history for this selection.")
    if ms["market_seat_share"] is not None:
        t2 += f" {c['market'].title()} supplies {ms['market_seat_share']:.1%} of arriving seats in the period."
    m = md["metrics_holdout"]
    t3 = (f"{MI.algorithm(md)} (holdout WMAPE {m['WMAPE']:.1%}). A conditional, model-based estimate, not a causal effect.")
    if any("extrapolation" in w for w in res.warnings):
        t3 += " Some inputs leave the training range."
    head = "Baseline reproduced" if none else ("No material change" if abs(s["pct_change"]) < 0.05 else ("Higher predicted demand" if up else "Lower predicted demand"))
    return [(head, t1, "green" if up and not none else "blue", "trend"),
            ("Market & seasonal context", t2, "blue", "calendar"), ("Model-based estimate", t3, "violet", "chart")]


@st.dialog("Scenario audit & technical details", width="large")
def _audit(res, md, a) -> None:
    rec = st.session_state.get("scenario_record")
    if not rec or rec.get("scenario_id") != res.scenario_id:
        rec = SC.scenario_record(res, md, _sens_for(res))
    st.markdown(f"**Scenario** `{res.scenario_id}` · model `{res.config['model_id']}` · feature pipeline `{rec['feature_pipeline_version']}`")
    st.markdown("**Consistency checks**")
    for ch in res.checks:
        st.markdown(f"{'✅' if ch['passed'] else '⚠️'} {ch['check']} — {ch['detail']}")
    for w in res.warnings:
        st.warning(w)
    st.markdown("**Aviation inputs (selected market and period)**")
    st.dataframe(res.aviation_changes, hide_index=True, width="stretch", column_config={"quantity": st.column_config.TextColumn("Quantity"),
        "baseline": st.column_config.NumberColumn("Baseline", format="%,.3f"), "scenario": st.column_config.NumberColumn("Scenario", format="%,.3f"),
        "change": st.column_config.NumberColumn("Change", format="%+,.3f"), "change_%": st.column_config.NumberColumn("Change %", format="%+.2f")})
    fc = res.feature_changes
    if len(fc):
        st.markdown("**Recomputed model features** (mean over the period)")
        st.dataframe(fc[fc["changed"]][["label", "baseline_mean", "scenario_mean", "change_%"]], hide_index=True, width="stretch",
                     column_config={"label": st.column_config.TextColumn("Feature"), "baseline_mean": st.column_config.NumberColumn("Baseline mean", format="%,.3f"),
                                    "scenario_mean": st.column_config.NumberColumn("Scenario mean", format="%,.3f"),
                                    "change_%": st.column_config.NumberColumn("Change %", format="%+.2f")})
    if len(res.extrapolation) and (res.extrapolation["pct_days_outside"] > 0).any():
        st.markdown("**Inputs outside the training range**")
        ex = res.extrapolation[res.extrapolation["pct_days_outside"] > 0]
        st.dataframe(ex[["label", "train_min", "train_max", "scenario_min", "scenario_max", "pct_days_outside"]], hide_index=True, width="stretch", column_config={
            "label": st.column_config.TextColumn("Feature"), "train_min": st.column_config.NumberColumn("Training min", format="%,.3f"),
            "train_max": st.column_config.NumberColumn("Training max", format="%,.3f"), "scenario_min": st.column_config.NumberColumn("Scenario min", format="%,.3f"),
            "scenario_max": st.column_config.NumberColumn("Scenario max", format="%,.3f"), "pct_days_outside": st.column_config.NumberColumn("Days outside (%)", format="%.1f")})
    st.markdown("**Assumptions**")
    for line in S.assumptions(res)["assumptions"]:
        st.markdown(f"- {line}")
    with st.container(horizontal=True, gap="small"):
        csv_button(S.results_table(res), f"{res.scenario_id}_results.csv", "Results (CSV)", key="sim_dl_res")
        csv_button(res.daily, f"{res.scenario_id}_daily_predictions.csv", "Daily predictions (CSV)", key="sim_dl_daily")
        st.download_button("Scenario record (JSON)", json.dumps(rec, indent=2, default=str).encode(), f"{res.scenario_id}_record.json", "application/json",
                           icon=":material/data_object:", key="sim_dl_rec", on_click="ignore")
        png_button(_demand_fig(res, md), f"{res.scenario_id}_demand.png", "Chart (PNG)", key="sim_dl_png")
