"""Section 6 — Model Evaluation (Model Performance Observatory).

Loads saved artifacts only — nothing is retrained. Every metric is recomputed from the saved
holdout predictions, the artifacts are integrity-checked against the metadata and the training
files, and the official-test forecasts are never scored (they have no actual values).
"""
from __future__ import annotations

import json
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from .. import config, registry
from ..evaluation import (METRIC_HELP, comparison_key, daily_aggregate, diagnostics, fold_table, insights, metrics_from_predictions,
                          permutation_table, verify_artifacts)
from ..features import CATALOG_BY_NAME
from ..integration import model_frame, training_rows
from ..models import importance_kind, native_importance
from ..training import feature_family
from . import design, eval_charts as EC, heroes
from .common import active_id, analytic, csv_button, plot, png_button, next_step
from .design import ICON
from .. import model_identity as MI
from .training_page import SCOPE_SHORT, _ablation_result, _ablations_for, _label, _registry_frame, _registry_stamp

FAMILY = {"aviation": "Aviation (incl. derived, prior week)", "calendar": "Calendar", "history": "Guest history (≥ 364 days)",
          "hotel": "Same-day hotel observations", "market": "Market identifier"}

CSS = """
<style>
.st-key-ev_ctx { background:#fff; border:1px solid #e7e8e7; border-radius:18px; padding:14px 18px 12px; box-shadow:0 1px 2px rgba(15,23,20,.04),0 10px 28px rgba(15,23,20,.06);
   animation:dsFadeUp .5s ease both; }
[class*="st-key-ev_"] [data-testid="stMarkdownContainer"], [class*="st-key-ds_panel_ev"] [data-testid="stMarkdownContainer"] { margin-bottom:0 !important; }
.ev-ctx { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:8px 14px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; margin-top:6px; }
.ev-ctx div { min-width:0; } .ev-ctx span { display:block; font-size:11.3px; color:#6f7572; font-weight:600; }
.ev-ctx b { display:block; font-size:13px; color:#101614; font-weight:650; line-height:1.35; overflow-wrap:anywhere; }
.ev-badges { display:flex; flex-wrap:wrap; gap:6px; margin-top:4px; }
.ev-chip { display:inline-flex; gap:5px; align-items:center; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12px; font-weight:650; border-radius:999px; padding:3px 10px; }
.ev-chip svg { width:13px; height:13px; } .ev-chip.ok { color:#047857; background:#e3f8ee; } .ev-chip.warn { color:#8a5a00; background:#fff2d6; }
.ev-chip.bad { color:#a61b1b; background:#fde8e8; } .ev-chip.info { color:#047857; background:#ecfdf5; } .ev-chip.grey { color:#606562; background:#e9f3ef; }
.ev-kpis { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:12px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.ev-kpi { background:#fff; border:1px solid #e7e8e7; border-radius:16px; padding:12px 14px; display:flex; gap:11px; align-items:flex-start;
   box-shadow:0 1px 2px rgba(15,23,20,.04),0 8px 22px rgba(15,23,20,.05); animation:dsFadeUp .5s ease both; min-width:0; }
.ev-kpi:nth-child(2) { animation-delay:.04s; } .ev-kpi:nth-child(3) { animation-delay:.08s; } .ev-kpi:nth-child(4) { animation-delay:.12s; }
.ev-kpi:nth-child(5) { animation-delay:.16s; } .ev-kpi:nth-child(6) { animation-delay:.2s; }
.ev-kpi .ds-ico { width:38px; height:38px; } .ev-kpi .ds-ico svg { width:19px; height:19px; }
.ev-kpi span { display:block; font-size:12px; color:#515553; font-weight:700; } .ev-kpi b { display:block; font-size:24px; font-weight:800; color:#101614; line-height:1.15; letter-spacing:-.01em; }
.ev-kpi small { display:block; font-size:11.2px; color:#6f7572; line-height:1.35; margin-top:2px; } .ev-kpi > div { min-width:0; }
.ev-strip { display:flex; flex-wrap:wrap; gap:6px 22px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.4px; color:#6f7572; margin:10px 2px 0; }
.ev-strip b { color:#313432; font-weight:700; }
.ev-title { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:16.5px; font-weight:800; color:#101614; }
.ev-sub { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.8px; color:#6f7572; line-height:1.45; margin-top:2px; }
.ev-note { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.3px; color:#6f7572; line-height:1.5; } .ev-note b { color:#313432; }
.ev-box { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.4px; line-height:1.45; border-radius:10px; padding:8px 11px; display:flex; gap:8px; margin:6px 0; }
.ev-box svg { width:15px; height:15px; flex:none; margin-top:1px; }
.ev-box.amber { background:#fff7e8; color:#8a5a00; border:1px solid #fbe3b4; } .ev-box.blue { background:#ecfdf5; color:#047857; border:1px solid #c6f0dc; }
.ev-box.green { background:#eaf8f2; color:#065f46; border:1px solid #c9ecdd; } .ev-box.red { background:#fdeeee; color:#9b1c1c; border:1px solid #f6caca; }
.ev-bias { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:10px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; margin:6px 0 4px; }
.ev-bias div { background:#f3fbf8; border:1px solid #e8f2ee; border-radius:12px; padding:8px 11px; } .ev-bias span { display:block; font-size:11.3px; color:#6f7572; font-weight:600; }
.ev-bias b { font-size:17px; color:#101614; font-weight:800; }
.ev-ins { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; } .ev-ins4 { grid-template-columns:repeat(4,minmax(0,1fr)) !important; } .ev-ins { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.ev-ins > div { border:1px solid #eceeed; border-radius:14px; padding:12px 14px; background:#f8fdfb; border-left:4px solid #059669; animation:dsFadeUp .5s ease both; }
.ev-ins > div.warn { border-left-color:#f2a33a; } .ev-ins > div.muted { border-left-color:#b0b4b2; }
.ev-ins .t { font-size:13.4px; font-weight:750; color:#101614; margin-bottom:4px; } .ev-ins p { margin:0; font-size:12.5px; color:#444846; line-height:1.5; }
.ev-checks { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:6px 18px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.4px; }
.ev-checks div { display:flex; gap:7px; align-items:flex-start; color:#313432; } .ev-checks svg { width:15px; height:15px; flex:none; margin-top:2px; }
.ev-checks .ok svg { color:#047857; } .ev-checks .bad svg { color:#c62828; } .ev-checks small { color:#6f7572; }
.ev-empty { display:flex; gap:22px; align-items:center; padding:12px 6px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.ev-empty svg.art { width:150px; height:110px; flex:none; } .ev-empty h4 { margin:0 0 6px; font-size:16px; color:#101614; } .ev-empty p { margin:0 0 6px; font-size:12.8px; color:#515553; line-height:1.5; }
.st-key-ev_tabnav [role="tablist"], [class*="st-key-ds_panel_ev"] [role="tablist"] { flex-wrap:wrap; overflow:visible !important; }
[class*="st-key-ds_panel_ev"] .stPlotlyChart { animation:labReveal .6s ease both; }
@keyframes labReveal { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
/* metric cards and insights follow the width they actually get (the window also holds the sidebar) */
.ev-kq { container-type:inline-size; } .ev-kpi b { white-space:nowrap; }
@container (max-width: 1180px) { .ev-kpis { grid-template-columns:repeat(3,minmax(0,1fr)); } .ev-ins4 { grid-template-columns:repeat(2,minmax(0,1fr)) !important; } }
@container (max-width: 560px) { .ev-kpis { grid-template-columns:repeat(2,minmax(0,1fr)); } .ev-ins4 { grid-template-columns:1fr !important; } }
@media (max-width: 1200px) { .ev-ctx { grid-template-columns:repeat(3,minmax(0,1fr)); } .ev-bias { grid-template-columns:repeat(3,minmax(0,1fr)); } }
@media (max-width: 1100px) {
  .ev-ins, .ev-checks { grid-template-columns:1fr; }
  [class*="st-key-ds_panel_ev"] [data-testid="stHorizontalBlock"], .st-key-ev_ctx [data-testid="stHorizontalBlock"] { flex-wrap:wrap; row-gap:8px; }
  [class*="st-key-ds_panel_ev"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"], .st-key-ev_ctx [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex:1 1 100% !important; min-width:100% !important; }
}
@media (max-width: 640px) { .ev-ctx, .ev-bias { grid-template-columns:repeat(2,minmax(0,1fr)); } }
</style>
"""

EMPTY_ART = ('<svg class="art" viewBox="0 0 150 110" fill="none" aria-hidden="true"><rect x="8" y="14" width="134" height="86" rx="12" fill="#e1fbf1" stroke="#c6f0dc"/>'
             '<rect x="24" y="64" width="16" height="24" rx="3" fill="#d1fae5"/><rect x="48" y="50" width="16" height="38" rx="3" fill="#18d993"/>'
             '<rect x="72" y="38" width="16" height="50" rx="3" fill="#10b981"/><rect x="96" y="56" width="16" height="32" rx="3" fill="#d1fae5" stroke="#10b981" stroke-dasharray="3 3"/>'
             '<circle cx="118" cy="30" r="14" fill="#fff" stroke="#059669" stroke-width="2"/><path d="m112 30 4 4 8-8" stroke="#059669" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def _box(kind: str, text: str, icon: str = "info") -> str:
    return f'<div class="ev-box {kind}">{ICON[icon]}<span>{text}</span></div>'


def _f(v, d=1) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:,.{d}f}"


# ------------------------------------------------------------------ cached artifacts

@st.cache_data(show_spinner=False)
def _load(mid: str, stamp: float) -> tuple[dict, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    return (registry.load_metadata(mid), registry.load_table(mid, "predictions_holdout"), registry.load_table(mid, "predictions_test"),
            registry.load_table(mid, "predictions_cv"), registry.load_table(mid, "importance"))


@st.cache_data(show_spinner=False)
def _observed(scope: str) -> pd.DataFrame:
    a = analytic()
    if scope == "international_market":
        hm = a.hotel_market[a.hotel_market["split"] == "train"]
        return hm[["Date", "market_key", "Guests"]]
    col = {"international_total": "intl_guests", "domestic": "dom_guests", "overall": "overall_guests"}[scope]
    ht = a.hotel_total[a.hotel_total["split"] == "train"]
    return ht[["Date", col]].rename(columns={col: "Guests"})


@st.cache_data(show_spinner=False)
def _train_rows(scope: str, mode: str) -> pd.DataFrame:
    return training_rows(model_frame(analytic(), scope, mode))


@st.cache_data(show_spinner=False)
def _groups(stamp: float) -> dict[str, tuple]:
    out = {}
    for m in registry.list_models():
        h = registry.load_table(m["model_id"], "predictions_holdout")
        if h is not None:
            out[m["model_id"]] = comparison_key(m, h)
    return out


@st.cache_data(show_spinner="Recomputing permutation importance from the saved pipeline …")
def _permutation(mid: str, grouped: bool, repeats: int, stamp: float) -> tuple[pd.DataFrame | None, str]:
    md = registry.load_metadata(mid)
    if md["algorithm"] == "seasonal_naive":
        return None, "The seasonal-naive baseline has no learned features."
    rows = _train_rows(md["scope"], md["mode"])
    hold = rows[rows["Date"] >= pd.Timestamp(md["config"]["holdout_start"])].reset_index(drop=True)
    model = registry.load_model(mid, "eval")
    saved = registry.load_table(mid, "predictions_holdout")
    keys = ["Date", "market_key"] if md["scope"] == "international_market" else ["Date"]
    j = hold[keys].merge(saved[keys + ["y_pred"]], on=keys, how="left")
    if len(hold) != len(saved) or not np.allclose(model.predict(hold[md["features"]]), j["y_pred"], rtol=1e-7, atol=1e-6):
        return None, ("The current feature data no longer reproduces this model's saved holdout predictions, so importance cannot be recomputed "
                      "faithfully; the values saved at training time are shown instead.")
    feats = md["features"]
    groups = {}
    if grouped:
        for f in feats:
            groups.setdefault(FAMILY[feature_family(f)], []).append(f)
    else:
        groups = {CATALOG_BY_NAME[f].label: [f] for f in feats}
    tab = permutation_table(model, hold[feats], hold["y"].to_numpy(float), groups, n_repeats=repeats, seed=md.get("seed", config.RANDOM_STATE))
    return tab, ""


# ------------------------------------------------------------------ page

def render() -> None:
    design.inject()
    st.markdown(heroes.CSS + CSS, unsafe_allow_html=True)
    design.html(heroes.banner("evaluation", "Section 6", "Model <span>Evaluation</span>", "How well each model predicts unseen days, and where it fails."))
    st.write("")
    stamp = _registry_stamp()
    df = _registry_frame(stamp)
    if df.empty:
        design.html(_box("blue", "No saved models yet — train one in the ML Training Studio."))
        return
    act = active_id()
    ids = df["model_id"].tolist()
    ids = ([act] if act in ids else []) + [m for m in ids if m != act]
    lab = df.set_index("model_id")["model"].to_dict()

    # ---------------------------------------------------------------- model context
    with st.container(key="ev_ctx"):
        c1, c2 = st.columns([1.5, 1], vertical_alignment="center")
        mid = c1.selectbox("Model", ids, key="ev_model", label_visibility="collapsed",
                           format_func=lambda m: ("★ " if m == act else "") + lab[m])
        md, hold, test, cvp, imp_saved = _load(mid, stamp)
        whatif = MI.scenario_ready(md)
        checks = verify_artifacts(md, hold, test, _observed(md["scope"]))
        all_ok = all(c["ok"] for c in checks)
        badges = [f'<span class="ev-chip {"info" if whatif else "warn"}">{ICON["check" if whatif else "lock"]}'
                  f'{"Simulator-ready" if whatif else "Not for the simulator"}</span>',
                  f'<span class="ev-chip {"ok" if all_ok else "bad"}">{ICON["shield" if all_ok else "warning"]}{"Artifacts verified" if all_ok else "Artifact check failed"}</span>']
        if mid == act:
            badges.append(f'<span class="ev-chip info">{ICON["check"]}Active in the simulator</span>')
        if md.get("reference"):
            badges.append(f'<span class="ev-chip grey">{ICON["lock"]}Reference model</span>')
        c2.markdown(f'<div class="ev-badges" style="justify-content:flex-end">{"".join(badges)}</div>', unsafe_allow_html=True)
        per = md["periods"]
        ctx = list(MI.facts(md).items()) + [("Mode", MI.mode(md)), ("Features", f"{len(md['features'])}")]
        design.html('<div class="ev-ctx">' + "".join(f"<div><span>{escape(k)}</span><b>{escape(v)}</b></div>" for k, v in ctx) + "</div>")
        if not whatif and md["mode"] != "competition":
            design.html(_box("amber", escape(MI.not_ready_reason(md) or ""), "lock"))
        if md["mode"] == "competition":
            design.html(_box("amber", "Competition model: it uses New Arrivals / Same-Day Guests, which exist only retrospectively in the official test files. "
                                      "Its scores describe nowcasting with those fields and cannot be compared with What-if forecasts; it cannot drive the simulator.", "warning"))
        with st.expander("Technical metadata and integrity checks", icon=":material/fact_check:"):
            design.html('<div class="ev-checks">' + "".join(
                f'<div class="{"ok" if c["ok"] else "bad"}">{ICON["check" if c["ok"] else "warning"]}<span>{escape(c["check"])}<br><small>{escape(c["detail"])}</small></span></div>'
                for c in checks) + "</div>")
            design.html('<div class="ev-ctx" style="margin:8px 0">' + "".join(f"<div><span>{escape(k)}</span><b>{escape(v)}</b></div>" for k, v in [
                ("Training period", f"{per['eval_fit'][0]} → {per['eval_fit'][1]}"), ("Holdout period", f"{per['holdout'][0]} → {per['holdout'][1]}"),
                ("Holdout rows", f"{len(hold):,}"), ("Unique holdout dates", f"{hold['Date'].nunique():,}"), ("Grain", md.get("grain", "")),
                ("Trained", md["created_at"].replace("T", " ")[:16])]) + "</div>")
            _meta_tables(md)

    m = metrics_from_predictions(hold)
    dg = diagnostics(hold)
    by_market = md["scope"] == "international_market"
    unit = "guests per nationality-day" if by_market else "guests per day"

    # ---------------------------------------------------------------- metric cards
    st.write("")
    kp = [("target", "blue", "MAE", _f(m["MAE"]), unit), ("chart", "violet", "RMSE", _f(m["RMSE"]), "guests · penalises large misses"),
          ("bars", "teal", "R²", _f(m["R2"], 3), "share of variance explained (not accuracy)"), ("scale", "orange", "WMAPE", f"{m['WMAPE']:.1%}", "Σ|actual − predicted| ÷ Σ actual"),
          ("trend", "indigo", "Pearson", _f(m["Pearson"], 3), "linear correlation, actual vs predicted"),
          ("scatter", "slate", "Spearman", _f(m["Spearman"], 3), "rank correlation, actual vs predicted")]
    design.html('<div class="ev-kq"><div class="ev-kpis">' + "".join(
        f'<div class="ev-kpi"><div class="ds-ico {tone}">{ICON[ic]}</div><div><span>{escape(t)}</span><b>{escape(v)}</b><small>{escape(s)}</small></div></div>'
        for ic, tone, t, v, s in kp) + "</div></div>")
    design.html(f'<div class="ev-strip"><span>Mean bias error <b>{dg["mean_residual"]:+,.1f}</b> (actual − predicted)</span>'
                f'<span>Median absolute error <b>{dg["median_abs_error"]:,.1f}</b></span><span>90th percentile absolute error <b>{dg["p90_abs_error"]:,.0f}</b></span>'
                f'<span>Holdout observations <b>{len(hold):,}</b></span><span>Validation period <b>{per["holdout"][0]} → {per["holdout"][1]}</b></span></div>')
    if by_market:
        dm = metrics_from_predictions(daily_aggregate(hold))
        design.html(f'<div class="ev-strip"><span>Summed to daily totals across {hold["market_key"].nunique()} markets: MAE <b>{dm["MAE"]:,.0f}</b> · '
                    f'WMAPE <b>{dm["WMAPE"]:.1%}</b> · R² <b>{dm["R2"]:.3f}</b> — not comparable with daily-total models trained on all 45 nationalities.</span></div>')

    # ---------------------------------------------------------------- tabs
    st.write("")
    with st.container(key="ds_panel_ev_tabs"):
        tabs = st.tabs(["Actual vs predicted", "Residual diagnostics", "Model comparison", "Feature importance", "Ablation study", "Cross-validation & data"],
                       key="ev_tabnav", on_change="rerun")
        if tabs[0].open:
            with tabs[0]:
                _tab_actual(md, hold, test, mid)
        if tabs[1].open:
            with tabs[1]:
                _tab_residuals(md, hold, dg, mid)
        if tabs[2].open:
            with tabs[2]:
                _tab_comparison(md, mid, stamp)
        if tabs[3].open:
            with tabs[3]:
                _tab_importance(md, mid, imp_saved, stamp)
        if tabs[4].open:
            with tabs[4]:
                _tab_ablation(md, mid)
        if tabs[5].open:
            with tabs[5]:
                _tab_cv(md, cvp, m, mid)

    # ---------------------------------------------------------------- insights
    st.write("")
    groups = _groups(stamp)
    base_id = next((x for x, k in groups.items() if k == groups.get(mid) and x.split("-")[2] == "seasonal_naive" and x != mid), None)
    base = metrics_from_predictions(registry.load_table(base_id, "predictions_holdout")) if base_id else None
    ins = insights(md, hold, base)
    primary = {"Overall performance": "Predictive performance", "Systematic bias": "Prediction bias", "Demand spikes": "Peak demand",
               "Scenario suitability": "Scenario suitability"}
    card = lambda i, title: f'<div class="{i["tone"]}"><div class="t">{escape(title)}</div><p>{escape(i["text"])}</p></div>'
    with design.panel("ev_insights", "Key insights", "Written by fixed rules from the metrics and predictions above — no language model, no external data."):
        design.html('<div class="ev-kq"><div class="ev-ins ev-ins4">' + "".join(card(i, primary[i["title"]]) for i in ins if i["title"] in primary) + "</div></div>")
        rest = [i for i in ins if i["title"] not in primary]
        if rest:
            with st.expander("More insights", icon=":material/insights:"):
                design.html('<div class="ev-ins">' + "".join(card(i, i["title"]) for i in rest) + "</div>")

    # ---------------------------------------------------------------- exports
    st.write("")
    with design.panel("ev_exports", "Exports", "Every file reflects the selected model; IDs and periods are included so each evaluation can be reproduced."):
        mrow = pd.DataFrame([{"model_id": mid, "algorithm": md["algorithm_label"], "scope": md["scope"], "mode": md["mode"], "holdout_start": per["holdout"][0],
                              "holdout_end": per["holdout"][1], **m, **{k: v for k, v in dg.items() if k != "n"}}])
        with st.container(horizontal=True, gap="small"):
            csv_button(hold, f"{mid}_predictions_holdout.csv", "Holdout predictions (CSV)", key="ev_dl_hold")
            csv_button(mrow, f"{mid}_metrics.csv", "Evaluation metrics (CSV)", key="ev_dl_metrics")
            st.download_button("Evaluation summary (Markdown)", _report(md, m, dg, checks, insights(md, hold, base)).encode(), file_name=f"{mid}_evaluation.md",
                               mime="text/markdown", icon=":material/description:", key="ev_dl_report", on_click="ignore")
    next_step("eval")



def _meta_tables(md: dict) -> None:
    ident = {"Model ID": md["model_id"], "Label": md.get("label", ""), "Created": md["created_at"], "Reference model": "yes" if md.get("reference") else "no",
             "Target transform": md.get("target_transform", "log1p/expm1" if md.get("log_target") else "none"),
             "Negative-prediction policy": md.get("negative_policy", "none (raw predictions)"), "Monotone constraint": "yes" if md.get("monotone") else "no",
             "Random seed": str(md.get("seed", md.get("config", {}).get("seed", ""))), "Runtime": f"{md.get('runtime_seconds', 0):.1f} s",
             "CV configuration": f"{md['config'].get('cv_folds')} folds × {md['config'].get('cv_horizon_days')} days (as saved)",
             "Holdout start (as saved)": md["config"].get("holdout_start"), "Matrix fingerprint": md.get("matrix_fingerprint", "not recorded (older run)")}
    st.dataframe(pd.DataFrame(ident.items(), columns=["Field", "Value"]).astype(str), hide_index=True, width="stretch")
    st.dataframe(pd.DataFrame([(k, str(v)) for k, v in md["params"].items()], columns=["Hyperparameter", "Value"]), hide_index=True, width="stretch")
    st.caption("Features in pipeline order: " + ", ".join(md["features"]))
    st.caption("Software: " + " · ".join(f"{k} {v}" for k, v in md["versions"].items()))


def _tab_actual(md: dict, hold: pd.DataFrame, test: pd.DataFrame | None, mid: str) -> None:
    by_market = md["scope"] == "international_market"
    c1, c2 = st.columns([1, 1.35], gap="medium")
    with c1:
        design.html('<div class="ev-title">Actual vs predicted</div><div class="ev-sub">Equal scales on both axes; distance from the dashed line is the error. '
                    'Colour = absolute residual; circles mark the 10 largest errors.</div>')
        f1 = EC.actual_vs_pred(hold, height=430)
        plot(f1, key="ev_avp")
        with st.container(horizontal=True, gap="small"):
            png_button(f1, f"{mid}_actual_vs_predicted.png", "PNG", key="ev_png_avp", width=800)
            csv_button(hold, f"{mid}_predictions_holdout.csv", "CSV", key="ev_csv_avp")
    with c2:
        design.html('<div class="ev-title">Actual vs predicted over time</div><div class="ev-sub">Green shading: the model under-predicted (actual above); '
                    'grey shading: over-predicted. Circles mark the 5 largest errors.</div>')
        o = st.columns([1.3, 1, 1.2], vertical_alignment="bottom")
        if by_market:
            mk = o[0].selectbox("Series", ["All markets (daily total)"] + sorted(hold["market_key"].unique()), key="ev_mk",
                                format_func=lambda s: s if s.startswith("All") else s.title())
            ts = daily_aggregate(hold) if mk.startswith("All") else hold[hold["market_key"] == mk]
            ext = (test.groupby("Date", as_index=False)["y_pred"].sum() if mk.startswith("All") else test[test["market_key"] == mk]) if test is not None and len(test) else None
        else:
            ts, ext = hold, test
        roll = o[1].toggle("7-day trend", value=False, key="ev_roll")
        show_ext = o[2].toggle("Test-period forecasts", value=False, key="ev_ext", help="Forecasts for 1 Aug 2025 → 28 Feb 2026; no actual Guests exist for these dates.")
        d0, d1 = ts["Date"].min().date(), ts["Date"].max().date()
        rng = st.slider("Holdout dates", min_value=d0, max_value=d1, value=(d0, d1), format="YYYY-MM-DD", key=f"ev_rng_{mid}", label_visibility="collapsed")
        tsel = ts[(ts["Date"] >= pd.Timestamp(rng[0])) & (ts["Date"] <= pd.Timestamp(rng[1]))]
        f2 = EC.over_time(tsel, extra=ext if show_ext else None, roll=7 if roll else 0, height=400)
        plot(f2, key="ev_ts")
        if show_ext:
            design.html(_box("blue", "Dotted line: forecasts from the deployment model for the official test dates. No actual Guests exist there, so no error, "
                                     "R² or correlation is — or can be — computed for them; they are not part of any statistic on this page."))
        with st.container(horizontal=True, gap="small"):
            png_button(f2, f"{mid}_over_time.png", "PNG", key="ev_png_ts", width=1100)
            csv_button(tsel, f"{mid}_timeseries.csv", "CSV", key="ev_csv_ts")
            if show_ext and ext is not None and len(ext):
                csv_button(ext, f"{mid}_official_test_forecasts.csv", "Test forecasts (CSV)", key="ev_csv_ext")


def _tab_residuals(md: dict, hold: pd.DataFrame, dg: dict, mid: str) -> None:
    lean = "under-predicts" if dg["mean_residual"] > 0 else "over-predicts"
    design.html('<div class="ev-title">Prediction bias</div><div class="ev-sub">Residual = actual − predicted: positive means the model under-predicted.</div>'
                '<div class="ev-bias">'
                f'<div><span>Mean residual</span><b>{dg["mean_residual"]:+,.1f}</b></div><div><span>Median residual</span><b>{dg["median_residual"]:+,.1f}</b></div>'
                f'<div><span>Under-predicted</span><b>{dg["under_share"]:.0%}</b></div><div><span>Over-predicted</span><b>{dg["over_share"]:.0%}</b></div>'
                f'<div><span>90th pct |error|</span><b>{dg["p90_abs_error"]:,.0f}</b></div><div><span>Largest |error|</span><b>{dg["max_abs_error"]:,.0f}</b></div></div>')
    design.html(f'<div class="ev-note">On average the model <b>{lean}</b> by {abs(dg["mean_residual"]):,.0f} guests; '
                f'{"both mean and median point the same way, so the lean is not driven by a few outliers" if np.sign(dg["mean_residual"]) == np.sign(dg["median_residual"]) else "mean and median disagree in sign — a few large errors drive the average"}. '
                'A small average error does not make a model unbiased: look at where the errors fall below.</div>')
    daily = daily_aggregate(hold) if "market_key" in hold else hold
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        plot(EC.residuals_vs_pred(hold), key="ev_rvp")
    with c2:
        plot(EC.residual_hist(hold), key="ev_rh")
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        design.html('<div class="ev-note"><b>Residuals over time</b>' + (" (market predictions summed per day)" if "market_key" in hold else "") + "</div>")
        plot(EC.residual_ts(daily), key="ev_rts")
    with c4:
        design.html('<div class="ev-note"><b>Absolute error over time</b> with a 14-day mean</div>')
        plot(EC.residual_ts(daily, absolute=True), key="ev_ats")
    opts = ["Month", "Weekday"] + (["Market"] if "market_key" in hold else [])
    b1, b2 = st.columns([1, 1], vertical_alignment="bottom")
    by = b1.segmented_control("Break errors down by", opts, default="Month", key="ev_by") or "Month"
    metric = b2.segmented_control("Metric", ["MAE", "Mean residual", "WMAPE"], default="MAE", key="ev_by_metric") or "MAE"
    h = hold.copy()
    h["Month"] = h["Date"].dt.strftime("%Y-%m")
    h["Weekday"] = h["Date"].dt.dayofweek.map(dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])))
    if by == "Market":
        h["Market"] = h["market_key"].str.title()
    g = h.groupby(by, sort=False).apply(lambda x: pd.Series({"MAE": (x["y_true"] - x["y_pred"]).abs().mean(), "Mean residual": (x["y_true"] - x["y_pred"]).mean(),
                                                             "WMAPE": (x["y_true"] - x["y_pred"]).abs().sum() / x["y_true"].abs().sum(), "n": len(x)}), include_groups=False).reset_index()
    if by == "Weekday":
        g[by] = pd.Categorical(g[by], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], ordered=True)
        g = g.sort_values(by)
    elif by == "Market":
        g = g.sort_values("MAE", ascending=False).head(15)
    else:
        g = g.sort_values(by)
    plot(EC.breakdown(g, by, metric), key="ev_breakdown")
    csv_button(g, f"{mid}_errors_by_{by.lower()}.csv", f"Errors by {by.lower()} (CSV)", key="ev_dl_by")


def _tab_comparison(md: dict, mid: str, stamp: float) -> None:
    groups = _groups(stamp)
    df = _registry_frame(stamp).set_index("model_id")
    mode_of = {x: x.split("-")[1] for x in groups}
    same = [x for x, k in groups.items() if k == groups.get(mid) and mode_of[x] == mid.split("-")[1]]
    design.html('<div class="ev-title">Model comparison</div><div class="ev-sub">Models are ranked only when they predict the same target at the same grain for '
                'exactly the same evaluation rows (same dates, same nationalities, same actual values). Other models can be added but are flagged.</div>')
    picked = st.multiselect("Models", list(df.index), default=same, key=f"ev_cmp_{mid}", format_func=lambda x: df.loc[x, "model"] + f" · {x[-6:]}",
                            label_visibility="collapsed", placeholder="Choose models to compare")
    if mid not in picked:
        picked = [mid] + picked
    rows = []
    for x in picked:
        mdx = registry.load_metadata(x)
        hx = registry.load_table(x, "predictions_holdout")
        mx = metrics_from_predictions(hx)
        ft = fold_table(registry.load_table(x, "predictions_cv"), mdx)
        rows.append({"model_id": x, "model": df.loc[x, "model"] + f" · {x[-6:]}", "algorithm": mdx["algorithm_label"].split(" (")[0],
                     "mode": "What-if" if mdx["mode"] == "whatif" else "Competition",
                     "comparable": groups.get(x) == groups.get(mid), **{k: mx[k] for k in ("MAE", "RMSE", "R2", "WMAPE", "Pearson", "Spearman")},
                     "CV_WMAPE_sd": ft["WMAPE"].std() if len(ft) > 1 else np.nan, "runtime_s": mdx.get("runtime_seconds"), "holdout_rows": len(hx)})
    tab = pd.DataFrame(rows)
    inc = tab[~tab["comparable"]]
    if len(inc):
        design.html(_box("amber", f"{len(inc)} selected model(s) are not comparable with this one (different scope, grain or evaluation rows) and are excluded from the "
                                  "ranking: " + escape(", ".join(inc["model"])), "warning"))
    comp = tab[tab["comparable"]]
    if comp["mode"].nunique() > 1:
        design.html(_box("amber", "The ranking mixes What-if and Competition models. They are scored on identical rows, but Competition models also see "
                                  "same-day hotel observations (New Arrivals), so their lower errors reflect extra information that a forecast or scenario never has.", "warning"))
    c1, c2 = st.columns([1, 1.1], vertical_alignment="bottom")
    metric = c1.segmented_control("Metric", ["MAE", "RMSE", "WMAPE", "R2", "Pearson", "Spearman", "runtime_s"], default="MAE", key="ev_cmp_metric") or "MAE"
    c2.markdown('<div class="ev-note">Lower is better for MAE, RMSE, WMAPE and runtime; higher for R² and correlations. The best R² is not automatically the '
                'best model for every use — check bias, peaks and scenario eligibility too.</div>', unsafe_allow_html=True)
    if len(comp) >= 1:
        plot(EC.metric_bars(comp, metric, "model", highlight=mid, pct=metric == "WMAPE"), key="ev_cmp_fig")
    st.dataframe(tab.drop(columns=["model_id"]), hide_index=True, width="stretch", column_config={
        "comparable": st.column_config.CheckboxColumn("Comparable"), "MAE": st.column_config.NumberColumn(format="%,.1f"), "RMSE": st.column_config.NumberColumn(format="%,.1f"),
        "R2": st.column_config.NumberColumn("R²", format="%.3f"), "WMAPE": st.column_config.NumberColumn(format="percent"),
        "Pearson": st.column_config.NumberColumn(format="%.3f"), "Spearman": st.column_config.NumberColumn(format="%.3f"),
        "CV_WMAPE_sd": st.column_config.NumberColumn("CV WMAPE sd", format="percent"), "runtime_s": st.column_config.NumberColumn("Runtime (s)", format="%.1f")})
    csv_button(tab, f"model_comparison_{md['scope']}.csv", "Download comparison (CSV)", key="ev_dl_cmp")
    others = [x for x in comp["model_id"] if x != mid]
    if others:
        with st.expander("Model-versus-model predictions", icon=":material/compare_arrows:"):
            other = st.selectbox("Compare with", others, format_func=lambda x: df.loc[x, "model"] + f" · {x[-6:]}", key="ev_vs")
            h1, h2 = registry.load_table(mid, "predictions_holdout"), registry.load_table(other, "predictions_holdout")
            keys = ["Date", "market_key"] if "market_key" in h1 else ["Date"]
            j = h1.merge(h2, on=keys + ["y_true"], suffixes=("_a", "_b"))
            import plotly.graph_objects as go
            fig = go.Figure(go.Scattergl(x=(j["y_true"] - j["y_pred_a"]).abs(), y=(j["y_true"] - j["y_pred_b"]).abs(), mode="markers",
                                         marker=dict(size=6, color=EC.BLUE, opacity=0.6), customdata=j["Date"].dt.strftime("%Y-%m-%d"),
                                         hovertemplate="%{customdata}<br>|error| this model %{x:,.0f}<br>|error| other %{y:,.0f}<extra></extra>"))
            mx_ = float(max((j["y_true"] - j["y_pred_a"]).abs().max(), (j["y_true"] - j["y_pred_b"]).abs().max()))
            fig.add_trace(go.Scatter(x=[0, mx_], y=[0, mx_], mode="lines", line=dict(color=EC.MUTED, dash="dash"), hoverinfo="skip", showlegend=False))
            design.style_fig(fig, height=360, legend=False)
            fig.update_xaxes(title_text="|error| — selected model")
            fig.update_yaxes(title_text="|error| — comparison model")
            plot(fig, key="ev_vs_fig")
            better = ((j["y_true"] - j["y_pred_a"]).abs() < (j["y_true"] - j["y_pred_b"]).abs()).mean()
            design.html(f'<div class="ev-note">The selected model has the smaller error on <b>{better:.0%}</b> of the {len(j):,} shared rows '
                        '(points above the dashed line).</div>')


def _tab_importance(md: dict, mid: str, imp_saved: pd.DataFrame | None, stamp: float) -> None:
    if md["algorithm"] == "seasonal_naive":
        design.html(_box("blue", "The seasonal-naive baseline uses no learned features, so it has no feature importance."))
        return
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        design.html('<div class="ev-title">Permutation importance (holdout)</div><div class="ev-sub">Increase in holdout RMSE (guests) when a feature is '
                    'shuffled, using the saved evaluation pipeline. Mean ± SD over repeats; negative values are kept.</div>')
        o = st.columns([1, 1], vertical_alignment="bottom")
        grouped = o[0].toggle("Group correlated features", value=False, key="ev_pi_group", help="Shuffle each feature family together (aviation, calendar, guest history, …).")
        repeats = o[1].select_slider("Repeats", [3, 5, 10], value=5, key="ev_pi_rep")
        tab, why = _permutation(mid, grouped, repeats, stamp)
        if tab is not None:
            t2 = tab.rename(columns={"name": "label"})
            plot(EC.importance_bars(t2, "mean", "sd", xlab="RMSE increase (guests)"), key="ev_pi")
            design.html(f'<div class="ev-note">Base holdout RMSE {tab["base_rmse"].iloc[0]:,.1f} on {tab["rows_used"].iloc[0]:,} rows'
                        f'{" (random sample)" if tab["rows_used"].iloc[0] < md["n_rows"]["holdout"] else ""}. Correlated aviation features share credit — '
                        'shuffling one leaves its near-duplicates intact, so individual values understate the family; use the grouped view. This is a description '
                        'of the fitted model on the holdout, not a guide for re-tuning on it.</div>')
            csv_button(tab, f"{mid}_permutation_importance.csv", "Download (CSV)", key="ev_dl_pi")
        else:
            design.html(_box("amber", escape(why), "warning"))
            if imp_saved is not None and imp_saved["permutation_rmse_increase"].notna().any():
                plot(EC.importance_bars(imp_saved, "permutation_rmse_increase", xlab="RMSE increase (guests), saved at training time"), key="ev_pi_saved")
    with c2:
        kind = importance_kind(md["algorithm"])
        design.html(f'<div class="ev-title">Native model importance</div><div class="ev-sub">{escape(kind)}. One-hot and missing-indicator columns are summed back '
                    'to the original feature.</div>')
        nat = None
        try:
            nat = native_importance(registry.load_model(mid, "eval"), md["features"])
        except Exception:
            nat = None
        if nat:
            nt = pd.DataFrame({"feature": list(nat), "native": list(nat.values())})
            nt["label"] = nt["feature"].map(lambda f: CATALOG_BY_NAME[f].label)
            xl = ("Coefficient per +1 SD of the input" + (" (log-guests scale)" if md.get("log_target") else " (guests)")) if md["algorithm"] == "ridge" else "Importance"
            plot(EC.importance_bars(nt, "native", xlab=xl), key="ev_ni")
            if md["algorithm"] == "ridge":
                design.html('<div class="ev-note">Inputs are standardised inside the pipeline, so coefficients are per standard deviation. With strongly '
                            'correlated inputs, individual coefficients can take opposite signs that cancel — read them together, not one by one.</div>')
            csv_button(nt, f"{mid}_native_importance.csv", "Download (CSV)", key="ev_dl_ni")
        else:
            design.html(_box("blue", "Native importance is not available for this algorithm (an RBF-kernel SVR has no per-feature weights)."))
    design.html(_box("amber", "Importance shows what the fitted model relies on to predict — not causal influence. A feature can rank highly because it moves "
                              "with demand (season, trend), not because changing it would change hotel guests.", "info"))


def _tab_ablation(md: dict, mid: str) -> None:
    found = _ablations_for(mid, md)
    if not found:
        design.html('<div class="ev-empty">' + EMPTY_ART + '<div><h4>No ablation study for this model yet</h4><p>A feature ablation retrains the same algorithm with '
                    'feature families removed or isolated (aviation, calendar, guest history, hotel observations) on the identical chronological holdout, and '
                    'reports how MAE, RMSE, R² and WMAPE change.</p><p>It measures predictive contribution under this setup, not causal importance. Studies are '
                    'never copied from a different model.</p></div></div>')
        from .pages import PAGES
        st.page_link(PAGES["train"], label="Configure an ablation study in the ML Training Studio", icon=":material/science:")
        return
    pick = st.selectbox("Saved study", found, key="ev_abl_pick") if len(found) > 1 else found[0]
    _ablation_result(pick)


def _tab_cv(md: dict, cvp: pd.DataFrame | None, m: dict, mid: str) -> None:
    ft = fold_table(cvp, md)
    cfg = md["config"]
    design.html(f'<div class="ev-title">Cross-validation & data</div><div class="ev-sub">As saved with this model: {cfg.get("cv_folds")} expanding-window folds × '
                f'{cfg.get("cv_horizon_days")} days before a holdout starting {cfg.get("holdout_start")}. Fold metrics are recomputed from predictions_cv.csv.</div>')
    if len(ft):
        if ft["train_rows"].isna().any():
            rows = _train_rows(md["scope"], md["mode"])
            start = pd.Timestamp(md["periods"]["eval_fit"][0])
            ft["train_rows"] = [int(((rows["Date"] >= start) & (rows["Date"] < pd.Timestamp(v))).sum()) for v in ft["val_start"]]
        plot(EC.timeline(md["periods"]["eval_fit"][0], ft.to_dict("records"), tuple(md["periods"]["holdout"]), height=150 + 28 * len(ft)), key="ev_timeline")
        c1, c2 = st.columns([1, 1], vertical_alignment="bottom")
        metric = c1.segmented_control("Fold metric", ["MAE", "RMSE", "WMAPE", "R2"], default="MAE", key="ev_fold_metric") or "MAE"
        plot(EC.fold_bars(ft, metric, m[metric], pct=metric == "WMAPE"), key="ev_folds")
        sd = ft[metric].std() if len(ft) > 1 else float("nan")
        design.html(f'<div class="ev-note">CV mean {metric} {ft[metric].mean():,.3f} ± {sd:,.3f} (SD across folds) vs holdout {m[metric]:,.3f}. '
                    'Folds train on all earlier dates only; the holdout was never used for fitting or model selection.</div>' if metric != "WMAPE" else
                    f'<div class="ev-note">CV mean WMAPE {ft["WMAPE"].mean():.1%} ± {sd:.1%} vs holdout {m["WMAPE"]:.1%}.</div>')
        st.dataframe(ft[["fold", "train_start", "train_end", "val_start", "val_end", "train_rows", "val_rows", "val_dates", "MAE", "RMSE", "R2", "WMAPE", "Pearson", "Spearman"]],
                     hide_index=True, width="stretch", column_config={"MAE": st.column_config.NumberColumn(format="%,.1f"), "RMSE": st.column_config.NumberColumn(format="%,.1f"),
                                                                      "R2": st.column_config.NumberColumn("R²", format="%.3f"), "WMAPE": st.column_config.NumberColumn(format="percent"),
                                                                      "Pearson": st.column_config.NumberColumn(format="%.3f"), "Spearman": st.column_config.NumberColumn(format="%.3f")})
        csv_button(ft, f"{mid}_cv_folds.csv", "Download CV folds (CSV)", key="ev_dl_cv")
    else:
        design.html(_box("blue", "This model was trained without cross-validation folds."))


def _report(md: dict, m: dict, dg: dict, checks: list[dict], ins: list[dict]) -> str:
    per = md["periods"]
    lines = [f"# Model evaluation — {md['algorithm_label']}", "", f"- Model ID: `{md['model_id']}`", f"- Scope: {md['scope_label']} · grain {md.get('grain', '')}",
             f"- Mode: {md['mode_label']}", f"- Training (evaluation fit): {per['eval_fit'][0]} → {per['eval_fit'][1]}",
             f"- Holdout: {per['holdout'][0]} → {per['holdout'][1]} ({md['n_rows']['holdout']:,} rows)", f"- Features ({len(md['features'])}): {', '.join(md['features'])}",
             "", "## Holdout metrics (recomputed from predictions_holdout.csv)", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v:,.4f} |" for k, v in m.items() if k != "n"]
    lines += [f"| Mean residual (actual − predicted) | {dg['mean_residual']:+,.2f} |", f"| Median absolute error | {dg['median_abs_error']:,.2f} |",
              f"| 90th percentile absolute error | {dg['p90_abs_error']:,.2f} |", f"| Under-predicted share | {dg['under_share']:.1%} |", "",
              "## Integrity checks", ""] + [f"- {'PASS' if c['ok'] else 'FAIL'} — {c['check']} ({c['detail']})" for c in checks]
    lines += ["", "## Interpretation (rule-based)", ""] + [f"- **{i['title']}.** {i['text']}" for i in ins]
    lines += ["", "Official-test forecasts have no actual values and are not scored. Associations are not causal effects."]
    return "\n".join(lines)
