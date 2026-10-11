"""Judges' Demo — the whole solution in five short stages, built for a 1–2 minute presentation.

challenge → verified model → aviation scenario → hotel-demand prediction → practical insight.
The featured model and scenario are read from artifacts/judges_demo.json; every metric is recomputed
from that exact model's saved predictions, and the scenario (with its model comparison) is re-run
live and checked against the saved reference result.
"""
from __future__ import annotations

import json
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from .. import config, registry
from .. import model_identity as MI
from .. import scenario as S
from .. import science as SC
from ..copilot import answer_planning
from ..evaluation import metrics_from_predictions, verify_artifacts
from . import design, heroes
from .common import analytic, go_to, next_step, plot
from .design import ICON

CSS = """
<style>
[class*="st-key-dm_"] [data-testid="stMarkdownContainer"] { margin-bottom:0 !important; }
[class*="st-key-dm_stage_"] { background:#fff; border:1px solid #e5e8e6; border-radius:16px; padding:16px 20px 14px; gap:.6rem;
    box-shadow:0 1px 2px rgba(15,23,20,.04),0 8px 24px rgba(15,23,20,.06); animation:dsFadeUp .5s ease both; }
.dm-h { display:flex; gap:12px; align-items:center; font-family:var(--q-font); }
.dm-n { width:32px; height:32px; border-radius:50%; background:#047857; color:#fff; font-weight:800; font-size:14px; display:flex; align-items:center; justify-content:center; flex:none; }
.dm-h b { font-size:16px; color:#101614; font-weight:780; } .dm-h span { font-size:12.6px; color:#6f7572; display:block; font-weight:500; }
.dm-q { font-family:var(--q-font); font-size:16px; color:#101614; font-weight:600; line-height:1.5; border-left:3px solid #10b981; padding:6px 0 6px 14px; }
.dm-q em { font-style:normal; color:#047857; font-weight:750; }
.dm-tiles { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; font-family:var(--q-font); }
.dm-t { border:1px solid #e9eceb; border-radius:12px; padding:10px 13px; background:#fbfcfb; min-width:0; }
.dm-t span { display:block; font-size:11.8px; color:#4b4f4d; font-weight:650; } .dm-t b { display:block; font-size:22px; color:#101614; font-weight:800; line-height:1.2; }
.dm-t small { display:block; font-size:11.5px; color:#6f7572; line-height:1.35; }
.dm-t.hl { background:#ecfdf5; border-color:#c6f0dc; } .dm-t.hl b { color:#047857; }
.dm-chip { display:inline-flex; gap:6px; align-items:center; font-family:var(--q-font); font-size:12.4px; font-weight:700; border-radius:999px; padding:4px 11px; }
.dm-chip svg { width:14px; height:14px; } .dm-chip.ok { color:#065f46; background:#d1fae5; } .dm-chip.bad { color:#9b1c1c; background:#fdeeee; }
.dm-chip.grey { color:#3a3f3c; background:#eef0ef; }
.dm-kinds { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; font-family:var(--q-font); }
.dm-kinds div { border-radius:12px; padding:10px 12px; font-size:12.8px; line-height:1.45; color:#2b302d; }
.dm-kinds b { display:block; font-size:12px; letter-spacing:.08em; text-transform:uppercase; margin-bottom:3px; }
.dm-kinds .o { background:#f3f6f4; } .dm-kinds .o b { color:#3a3f3c; } .dm-kinds .p { background:#ecfdf5; } .dm-kinds .p b { color:#047857; }
.dm-kinds .u { background:#fff7e8; } .dm-kinds .u b { color:#8a5a00; }
.dm-note { font-family:var(--q-font); font-size:12.4px; color:#6f7572; line-height:1.5; }
@media (max-width: 1100px) { .dm-tiles { grid-template-columns:repeat(2,minmax(0,1fr)); } .dm-kinds { grid-template-columns:1fr; } }
</style>
"""


def _stage(n: int, title: str, sub: str) -> None:
    design.html(f'<div class="dm-h"><div class="dm-n">{n}</div><div><b>{escape(title)}</b><span>{escape(sub)}</span></div></div>')


def _tiles(items: list[tuple]) -> str:
    return '<div class="dm-tiles">' + "".join(f'<div class="dm-t{" hl" if len(t) > 3 and t[3] else ""}"><span>{escape(t[0])}</span><b>{escape(t[1])}</b>'
                                              f'<small>{escape(t[2])}</small></div>' for t in items) + "</div>"


def _observed(scope: str) -> pd.DataFrame:
    from .evaluation_page import _observed as obs
    return obs(scope)


def _demand_fig(res, md):
    import plotly.graph_objects as go
    d = res.daily
    if md["scope"] == "international_market" and res.config["market"] != "ALL":
        d = d[d["market_key"] == res.config["market"]]
    d = d.groupby("Date", as_index=False)[["baseline", "scenario"]].sum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=d["baseline"], mode="lines", name="Baseline prediction", line=dict(color="#4b5563", width=2, dash="dash"),
                             hovertemplate="%{x|%d %b}: %{y:,.0f}<extra>baseline</extra>"))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["scenario"], mode="lines", name="Scenario prediction", line=dict(color="#059669", width=2.6),
                             fill="tonexty", fillcolor="rgba(5,150,105,0.12)", hovertemplate="%{x|%d %b}: %{y:,.0f}<extra>scenario</extra>"))
    design.style_fig(fig, height=270)
    fig.update_yaxes(title_text="Predicted guests per day", separatethousands=True)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
    return fig


def render() -> None:
    design.inject()
    st.markdown(heroes.CSS + CSS, unsafe_allow_html=True)
    design.html(heroes.banner("demo", "Guided tour", "Judges’ <span>Demo</span>", "The whole solution in five short stages, with a verified model and a reproducible scenario."))
    st.write("")
    p = config.ARTIFACTS_DIR / "judges_demo.json"
    if not p.exists():
        st.warning("The reference scenario is not available yet. Train a What-if model in the ML Training Studio to enable the guided tour.")
        return
    demo = json.loads(p.read_text())
    mid = demo["model_id"]
    try:
        md = registry.load_metadata(mid)
    except FileNotFoundError:
        st.error("The reference model files are missing. Rebuild them with the project's reference-training script (see the README).")
        return
    sc = demo["scenario"]
    hold = registry.load_table(mid, "predictions_holdout")
    m = metrics_from_predictions(hold)
    checks = verify_artifacts(md, hold, registry.load_table(mid, "predictions_test"), _observed(md["scope"]))
    ok = all(c["ok"] for c in checks)
    mk = sc["market"].title()

    # ---------------------------------------------------------------- 1 · challenge
    with st.container(key="dm_stage_1"):
        _stage(1, "The challenge", "What the tool must do")
        design.html('<div class="dm-q">How might we build a tool that predicts how <em>flight changes</em> affect <em>hotel visits</em> in Abu Dhabi?</div>'
                    '<div class="dm-note">QuanAra learns how daily arrivals into Abu Dhabi relate to hotel guests, validates that on months the models never saw, '
                    'and re-predicts guests when aviation inputs change.</div>')
    st.write("")
    # ---------------------------------------------------------------- 2 · verified model
    with st.container(key="dm_stage_2"):
        _stage(2, "The verified model", f"{MI.algorithm(md)} · {MI.target(md)} · {MI.validation(md)}")
        daily = md.get("metrics_holdout_daily_total")
        design.html(_tiles([("Holdout WMAPE", f"{m['WMAPE']:.1%}", "weighted error on unseen days", True),
                            ("Holdout R²", f"{m['R2']:.3f}", "share of variance explained"),
                            ("MAE", f"{m['MAE']:,.0f}", MI.unit(md)),
                            ("Daily-total WMAPE", f"{daily['WMAPE']:.1%}" if daily else "—", "markets summed per day" if daily else "single daily series")]))
        design.html(f'<span class="dm-chip {"ok" if ok else "bad"}">{ICON["shield" if ok else "warning"]}'
                    f'{"Metrics recomputed from this model’s saved predictions — all integrity checks passed" if ok else "An integrity check failed — see Model Evaluation"}</span>')
    st.write("")
    # ---------------------------------------------------------------- 3 · scenario
    with st.container(key="dm_stage_3"):
        _stage(3, "The aviation scenario", "A reproducible question for the model")
        design.html(f'<div class="dm-q">What does the model predict for <em>{escape(mk)}</em> guests in <em>{pd.Timestamp(sc["start"]):%d %b} – '
                    f'{pd.Timestamp(sc["end"]):%d %b %Y}</em> if seat capacity from {escape(mk)} is <em>{sc["seats_change_pct"]:+.0f}%</em> higher '
                    '(same aircraft size, load factor and P2P share)?</div>')
        run = st.button("Run the demo scenario live", type="primary", icon=":material/play_arrow:", key="dm_run")
    if run:
        with st.spinner("Rebuilding features and predicting with the reference model and the comparable models …"):
            peers = [x["model_id"] for x in registry.list_models() if x["model_id"] != mid and MI.comparable(md, x)
                     and (MI.scenario_ready(x) or x["algorithm"] == "seasonal_naive")]
            st.session_state["demo_result"] = S.run(S.ScenarioConfig(model_id=mid, **sc), analytic(), compare_with=peers, save=False)
    res = st.session_state.get("demo_result")
    if res is not None and res.config.get("model_id") != mid:
        res = None
    saved = demo["summary"]
    st.write("")
    # ---------------------------------------------------------------- 4 · prediction
    with st.container(key="dm_stage_4"):
        _stage(4, "The hotel-demand prediction", "Average guests per day over the period, from the saved model")
        if res is None:
            design.html(f'<div class="dm-note">Press <b>Run the demo scenario live</b> above. The saved reference result is '
                        f'{saved["baseline_avg_daily_guests"]:,.0f} → {saved["scenario_avg_daily_guests"]:,.0f} guests/day ({saved["pct_change"]:+.2f}%).</div>')
        else:
            s = res.summary
            design.html(_tiles([("Baseline", f"{s['baseline_avg_daily_guests']:,.0f}", "guests/day"), ("Scenario", f"{s['scenario_avg_daily_guests']:,.0f}", "guests/day", True),
                                ("Difference", f"{s['abs_change_avg_daily']:+,.0f}", f"guests/day · {s['abs_change_guest_days']:+,.0f} guest-days"),
                                ("Change", f"{s['pct_change']:+.2f}%", "scenario vs baseline", True)]))
            plot(_demand_fig(res, md), key="dm_fig")
            same = (np.isclose(s["baseline_avg_daily_guests"], saved["baseline_avg_daily_guests"])
                    and np.isclose(s["scenario_avg_daily_guests"], saved["scenario_avg_daily_guests"]))
            design.html(f'<span class="dm-chip {"ok" if same else "bad"}">{ICON["check" if same else "warning"]}'
                        f'{"Reproduced exactly — identical to the saved reference result" if same else "Differs from the saved reference result"}</span>')
            for w in res.warnings:
                st.warning(w)
    st.write("")
    # ---------------------------------------------------------------- 5 · insight
    with st.container(key="dm_stage_5"):
        _stage(5, "What it means", "Observed, predicted and uncertain — kept apart")
        if res is None:
            design.html('<div class="dm-note">Run the scenario to see its practical reading.</div>')
        else:
            s, av = res.summary, res.aviation_changes.set_index("quantity")
            band = s.get("daily_error_band_rel_q90")
            cmp = res.comparison
            resp = cmp[cmp["model_id"].map(lambda x: registry.load_metadata(x)["algorithm"] != "seasonal_naive")] if cmp is not None else None
            spread = (f"comparable models predict {resp['pct_change'].min():+.1f}% to {resp['pct_change'].max():+.1f}%"
                      if resp is not None and len(resp) > 1 else "no comparable model to cross-check")
            design.html('<div class="dm-kinds">'
                        f'<div class="o"><b>Observed</b>Baseline flights from {escape(mk)}: {av.loc["Seats", "baseline"]:,.0f} seats and '
                        f'{av.loc["P2P passengers", "baseline"]:,.0f} point-to-point passengers in the period.</div>'
                        f'<div class="p"><b>Predicted</b>{s["abs_change_avg_daily"]:+,.0f} {escape(SC.population_label(md, sc["market"]))} per day '
                        f'({s["pct_change"]:+.1f}%) when seats rise {sc["seats_change_pct"]:+.0f}%.</div>'
                        f'<div class="u"><b>Uncertain</b>{"Daily error band ±" + format(band, ".0%") + " (holdout); " if band is not None else ""}{escape(spread)}. '
                        'A model-based estimate, not a causal effect.</div></div>')
            ms = SC.market_season(analytic(), md, sc["market"], sc["start"], sc["end"])
            st.markdown(answer_planning(res, mid, ms))
            if cmp is not None and len(cmp) > 1:
                labels = MI.labels(registry.list_models())
                t = cmp.assign(Model=cmp["model_id"].map(lambda x: labels.get(x, x) + (" (no aviation input)" if registry.load_metadata(x)["algorithm"] == "seasonal_naive" else "")))
                with st.expander("Same scenario, comparable models (run live)", icon=":material/compare_arrows:"):
                    st.dataframe(t[["Model", "holdout_WMAPE", "baseline_avg_daily", "scenario_avg_daily", "pct_change"]], hide_index=True, width="stretch", column_config={
                        "holdout_WMAPE": st.column_config.NumberColumn("Holdout WMAPE", format="percent"),
                        "baseline_avg_daily": st.column_config.NumberColumn("Baseline / day", format="%,.0f"),
                        "scenario_avg_daily": st.column_config.NumberColumn("Scenario / day", format="%,.0f"),
                        "pct_change": st.column_config.NumberColumn("Change %", format="%+.2f")})
                    design.html('<div class="dm-note">All rows share the same prediction target and holdout period. The seasonal baseline uses no aviation '
                                'input, so it cannot respond to the scenario. Disagreement on the size of the effect is part of the honest answer.</div>')
    with st.expander("Reproducibility details", icon=":material/fingerprint:"):
        st.markdown(f"- **Model ID:** `{mid}`\n- **Saved reference scenario:** `{demo['scenario_id']}`\n"
                    f"- **Trained:** {md['periods']['deployment_fit'][0]} → {md['periods']['deployment_fit'][1]} · seed {md['config']['seed']} · "
                    f"{'scenario-safe shape constraint on volume features' if md.get('monotone') else 'unconstrained'} · {'log' if md.get('log_target') else 'raw'} target")
    st.write("")
    with st.container(horizontal=True, gap="small", key="dm_links"):
        if st.button("Inspect the data", icon=":material/database:", key="dm_go_data"):
            go_to("data")
        if st.button("See the relationship", icon=":material/scatter_plot:", key="dm_go_eda"):
            go_to("eda")
        if st.button("Review model performance", icon=":material/monitoring:", key="dm_go_eval"):
            go_to("eval")
        if st.button("Try your own scenario", icon=":material/flight_takeoff:", type="primary", key="dm_go_sim"):
            go_to("sim")
    next_step("demo")
