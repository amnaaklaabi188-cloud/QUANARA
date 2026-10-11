"""Section 5 — ML Training Studio.

Every number on this page is either a configuration the user chose or a value computed from a
real training run (saved predictions, saved metadata). Training is explicit: nothing retrains
when a control changes.
"""
from __future__ import annotations

import json
import math
import time
import traceback
from datetime import datetime
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from .. import config, registry
from ..evaluation import comparison_key, diagnostics, fold_table, metrics_from_predictions
from ..features import CATALOG_BY_NAME, available_features, default_features
from .. import model_identity as MI
from ..models import (ALGORITHM_INFO, HYPERPARAMS, MONOTONE_ALGORITHMS, NEGATIVE_POLICIES, available_algorithms,
                      importance_kind, monotone_features)
from ..training import ABLATION_EXPERIMENTS, ABLATIONS_DIR, TrainConfig, cv_folds, run_ablation, train
from . import design, eval_charts as EC, heroes
from .common import active_id, analytic, csv_button, plot, next_step
from .design import ICON
from .features_page import _matrix as fe_matrix, _rows as fe_rows, selected_features

GROUP_TABS = {"all": "All", "aviation": "Aviation", "time": "Time", "derived": "Derived", "historical": "Historical", "market": "Market",
              "competition": "Hotel obs."}
SCOPE_SHORT = {"international_total": "International guests · daily total", "international_market": "International guests · by nationality",
               "domestic": "Domestic guests · daily total", "overall": "All guests · daily total"}
GRAIN = {"international_total": "Target: Guests per day, summed over every reported nationality.",
         "international_market": "Target: Guests per nationality per day (33 matched source markets in What-if mode).",
         "domestic": "Target: domestic Guests per day.", "overall": "Target: international + domestic Guests per day."}

CSS = """
<style>
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-ts_c_"]) { height:100%; }
.st-key-ts_cards [data-testid="stHorizontalBlock"] { align-items:stretch; }
.st-key-ts_cards [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { height:100%; }
[class*="st-key-ts_c_"] { height:100%; background:#fff; border:1px solid #e7e8e7; border-radius:16px; padding:13px 15px 10px; gap:.4rem;
   box-shadow:0 1px 2px rgba(15,23,20,.04),0 8px 22px rgba(15,23,20,.05); animation:dsFadeUp .5s ease both; }
.st-key-ts_c_mode { animation-delay:.05s; } .st-key-ts_c_target { animation-delay:.1s; } .st-key-ts_c_algo { animation-delay:.15s; }
[class*="st-key-ts_c_"] [data-testid="stMarkdownContainer"], [class*="st-key-ds_panel_ts"] [data-testid="stMarkdownContainer"] { margin-bottom:0 !important; }
.ts-ch { display:flex; gap:10px; align-items:center; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.ts-ch b { font-size:13.8px; color:#101614; font-weight:750; } .ts-ch .ds-ico { width:34px; height:34px; } .ts-ch .ds-ico svg { width:18px; height:18px; }
.ts-cap { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:11.8px; color:#6f7572; line-height:1.4; }
.ts-cap b { color:#313432; }
.ts-chip { display:inline-flex; gap:5px; align-items:center; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12px; font-weight:650;
   border-radius:999px; padding:3px 10px; } .ts-chip svg { width:13px; height:13px; }
.ts-chip.ok { color:#047857; background:#e3f8ee; } .ts-chip.warn { color:#8a5a00; background:#fff2d6; } .ts-chip.bad { color:#a61b1b; background:#fde8e8; }
.ts-chip.info { color:#047857; background:#ecfdf5; } .ts-chip.grey { color:#606562; background:#e9f3ef; }
.st-key-ts_pills button { font-size:12.4px; transition:background .15s, color .15s, transform .15s; }
.st-key-ts_pills button:is([aria-checked="true"],[aria-pressed="true"]) { background:#059669 !important; border-color:#059669 !important; color:#ffffff !important;
   animation:tsChipIn .28s ease both; }
.st-key-ts_pills button:is([aria-checked="true"],[aria-pressed="true"]) p { color:#ffffff !important; }
.st-key-ts_pills button:is([aria-checked="true"],[aria-pressed="true"])::after { content:"×"; margin-left:7px; font-weight:700; opacity:.85; }
.st-key-ts_pills button:hover { transform:translateY(-1px); }
@keyframes tsChipIn { from { transform:scale(.92); opacity:.6; } to { transform:none; opacity:1; } }
.st-key-ts_feat_actions button p { white-space:nowrap; font-size:13px; }
[class*="st-key-ds_panel_ts"] [role="tablist"] { flex-wrap:wrap; overflow:visible !important; }
.st-key-tr_abl_exps button { font-size:12.4px; } .st-key-tr_abl_exps button:is([aria-checked="true"],[aria-pressed="true"]) { background:#ecfdf5 !important; border-color:#059669 !important; color:#047857 !important; }
.ts-title { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:17px; font-weight:800; color:#101614; letter-spacing:-.01em; }
.ts-sub { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.8px; color:#6f7572; line-height:1.45; margin-top:2px; }
.ts-note { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.3px; color:#6f7572; line-height:1.5; } .ts-note b { color:#313432; }
.ts-box { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.4px; line-height:1.45; border-radius:10px; padding:8px 11px; display:flex; gap:8px; margin:4px 0; }
.ts-box svg { width:15px; height:15px; flex:none; margin-top:1px; }
.ts-box.amber { background:#fff7e8; color:#8a5a00; border:1px solid #fbe3b4; } .ts-box.blue { background:#ecfdf5; color:#047857; border:1px solid #c6f0dc; }
.ts-box.red { background:#fdeeee; color:#9b1c1c; border:1px solid #f6caca; } .ts-box.green { background:#eaf8f2; color:#065f46; border:1px solid #c9ecdd; }
.ts-metrics { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:10px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.ts-m { border:1px solid #eceeed; border-radius:12px; padding:9px 12px; background:#f8fdfb; min-width:0; animation:dsFadeUp .5s ease both; }
.ts-m span { display:block; font-size:11.8px; color:#515553; font-weight:600; } .ts-m b { display:block; font-size:23px; font-weight:800; color:#101614; line-height:1.2; }
.ts-m small { display:block; font-size:11px; color:#6f7572; line-height:1.35; }
.ts-ctx { display:flex; flex-wrap:wrap; gap:4px 16px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.3px; color:#6f7572; margin:8px 0 4px; }
.ts-ctx b { color:#313432; font-weight:650; }
.ts-stage { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.6px; color:#313432; }
@media (max-width: 1100px) {
  .st-key-ts_cards [data-testid="stHorizontalBlock"], .st-key-ts_main > div > [data-testid="stHorizontalBlock"] { flex-wrap:wrap; row-gap:10px; }
  .st-key-ts_cards [data-testid="stColumn"], .st-key-ts_main > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex:1 1 100% !important; min-width:100% !important; }
  .ts-metrics { grid-template-columns:repeat(3,minmax(0,1fr)); }
  [class*="st-key-ds_panel_ts"] [data-testid="stHorizontalBlock"] { flex-wrap:wrap; row-gap:8px; }
  [class*="st-key-ds_panel_ts"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex:1 1 45% !important; min-width:45% !important; }
}
</style>
"""


# ------------------------------------------------------------------ helpers

def _head(icon: str, tone: str, title: str) -> None:
    design.html(f'<div class="ts-ch"><div class="ds-ico {tone}">{ICON[icon]}</div><b>{escape(title)}</b></div>')


def _box(kind: str, text: str, icon: str = "info") -> str:
    return f'<div class="ts-box {kind}">{ICON[icon]}<span>{text}</span></div>'


def _param_widget(algo: str, name: str, spec: tuple, key: str):
    label, typ, default, lo, hi, step, help_ = spec
    if typ == "int":
        return st.slider(label, int(lo), int(hi), int(default), int(step), help=help_, key=key)
    if typ == "choice":
        return st.selectbox(label, lo, index=lo.index(default), help=help_, key=key)
    if step is None:  # log-scale float
        exps = [round(x, 4) for x in [lo * (hi / lo) ** (i / 40) for i in range(41)]]
        idx = min(range(len(exps)), key=lambda i: abs(math.log(exps[i]) - math.log(default)))
        return st.select_slider(label, options=exps, value=exps[idx], help=help_ + " (log scale)", key=key)
    return st.slider(label, float(lo), float(hi), float(default), float(step), help=help_, key=key)


def _sel_key(scope: str, mode: str) -> str:
    return f"{scope}|{mode}"


def _selection(scope: str, mode: str) -> list[str]:
    d = st.session_state.setdefault("tr_sel", {})
    allowed = set(available_features(scope, mode))
    if _sel_key(scope, mode) not in d:
        d[_sel_key(scope, mode)] = [f for f in selected_features(scope, mode) if f in allowed]
    return [f for f in d[_sel_key(scope, mode)] if f in allowed]


def _set_selection(scope: str, mode: str, feats: list[str]) -> None:
    order = list(CATALOG_BY_NAME)
    st.session_state.setdefault("tr_sel", {})[_sel_key(scope, mode)] = sorted(set(feats), key=order.index)
    st.session_state["tr_pills_v"] = st.session_state.get("tr_pills_v", 0) + 1


def _pills_changed(scope: str, mode: str, visible: list[str], key: str) -> None:
    cur = set(_selection(scope, mode)) - set(visible)
    _set_selection(scope, mode, list(cur | set(st.session_state.get(key) or [])))


def _reset_hp(algo: str) -> None:
    for k in [k for k in st.session_state if isinstance(k, str) and k.startswith(f"hp_{algo}_")]:
        del st.session_state[k]


def _fmt(v, d=1) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:,.{d}f}"


def _label(md: dict) -> str:
    """The app-wide display name (quanara.model_identity), disambiguated by training time when two names coincide."""
    return MI.labels(registry.list_models()).get(md["model_id"], MI.name(md))


@st.cache_data(show_spinner=False)
def _registry_frame(stamp: float) -> pd.DataFrame:
    rows = []
    labels = MI.labels(registry.list_models())
    for m in registry.list_models():
        h = m.get("metrics_holdout", {})
        cvm = m.get("metrics_cv_mean") or {}
        rows.append({"model_id": m["model_id"], "model": labels[m["model_id"]], "algorithm": MI.algorithm(m), "scope": m["scope"],
                     "mode": "What-if" if m["mode"] == "whatif" else "Competition", "features": len(m["features"]),
                     "MAE": h.get("MAE"), "RMSE": h.get("RMSE"), "R2": h.get("R2"), "WMAPE": h.get("WMAPE"), "CV_WMAPE": cvm.get("WMAPE"),
                     "holdout": " → ".join(m["periods"]["holdout"]), "n_holdout": m["n_rows"]["holdout"],
                     "trained": m["created_at"].replace("T", " ")[:16], "runtime_s": m.get("runtime_seconds"),
                     "scenario_ready": MI.scenario_ready(m), "reference": bool(m.get("reference"))})
    return pd.DataFrame(rows)


def _registry_stamp() -> float:
    d = config.MODELS_DIR
    return max([p.stat().st_mtime for p in d.iterdir()] + [0]) if d.exists() else 0.0


# ------------------------------------------------------------------ page

def render() -> None:
    design.inject()
    st.markdown(heroes.CSS + CSS, unsafe_allow_html=True)
    design.html(heroes.banner("training", "Section 5", 'ML <span>Training Studio</span>', "Train, validate and register predictive models with chronological splits."))
    st.write("")
    a = analytic()
    algos = available_algorithms()

    # ---------------------------------------------------------------- row 1: configuration cards
    with st.container(key="ts_cards"):
        c = st.columns([1.4, 1.5, 1.25], gap="small")
        with c[0], st.container(key="ts_c_scope"):
            _head("globe", "blue", "Prediction scope")
            scope = st.selectbox("Prediction scope", list(config.SCOPES), format_func=SCOPE_SHORT.get, key="tr_scope", label_visibility="collapsed")
            design.html(f'<div class="ts-cap"><b>Grain:</b> {escape(GRAIN[scope])}</div>')
        with c[1], st.container(key="ts_c_mode"):
            _head("database", "violet", "Prediction mode")
            mode = st.radio("Prediction mode", list(config.MODES), key="tr_mode", label_visibility="collapsed",
                            format_func=lambda m: "What-if :gray[— aviation, calendar, last year's guests]" if m == "whatif"
                            else "Competition :gray[— also same-day hotel fields]")
            design.html('<div class="ts-cap">' + ("Models trained in this mode can drive the What-if Simulator."
                        if mode == "whatif" else "A <b>nowcast</b> for scoring the official test dates; it cannot drive the simulator.") + "</div>")
        with c[2], st.container(key="ts_c_algo"):
            _head("gear", "indigo", "Model algorithm")
            order = [k for k in ["catboost", "xgboost", "random_forest", "extra_trees", "ridge", "svr", "seasonal_naive"] if k in algos]
            algo = st.selectbox("Model algorithm", order, format_func=lambda k: MI.ALGORITHMS.get(k, algos[k].split(" (")[0]), key="tr_algo",
                                label_visibility="collapsed")
            info, scen_ok, mono_ok = ALGORITHM_INFO[algo]
            design.html(f'<div class="ts-cap">{escape(info)}</div>')

    allowed = available_features(scope, mode)
    sel = _selection(scope, mode)
    rows = fe_rows(scope, mode).set_index("name")
    tr = fe_matrix(scope, mode)

    st.write("")
    with st.container(key="ts_main"):
        left, right = st.columns([2.05, 1], gap="medium")

        # ------------------------------------------------------------ feature subset
        with left, st.container(key="ds_panel_ts_feats"):
            design.html('<div class="ts-title">Feature subset <span style="font-weight:600;color:#6f7572;font-size:14px">(defaults from Feature Engineering)</span></div>')
            groups = ["all"] + [g for g in GROUP_TABS if g != "all" and any(CATALOG_BY_NAME[f].group == g for f in allowed)]  # only groups usable in this mode
            grp = st.segmented_control("Group", groups, default="all", key="tr_grp", label_visibility="collapsed",
                                       format_func=lambda g: f"{GROUP_TABS[g]} ({len(allowed) if g == 'all' else sum(CATALOG_BY_NAME[f].group == g for f in allowed)})") or "all"
            s1, s2 = st.columns([2.4, 1], vertical_alignment="center")
            q = s1.text_input("Search", placeholder="Search features", icon=":material/search:", key="tr_q", label_visibility="collapsed")
            only = s2.toggle("Selected only", key="tr_only")
            vis = [f for f in allowed if (grp == "all" or CATALOG_BY_NAME[f].group == grp)]
            if q:
                ql = q.lower()
                vis = [f for f in vis if ql in (f + " " + CATALOG_BY_NAME[f].label + " " + CATALOG_BY_NAME[f].source).lower()]
            if only:
                vis = [f for f in vis if f in sel]
            disabled = algo == "seasonal_naive"
            pk = f"tr_pills_{scope}_{mode}_{grp}_{abs(hash(q)) % 10**6}_{int(only)}_{st.session_state.get('tr_pills_v', 0)}"
            with st.container(key="ts_pills"):
                if vis:
                    st.pills("Features", vis, selection_mode="multi", default=[f for f in vis if f in sel], key=pk, label_visibility="collapsed",
                             format_func=lambda f: f"{CATALOG_BY_NAME[f].label}  [{GROUP_TABS[CATALOG_BY_NAME[f].group].lower()}]",
                             on_change=_pills_changed, args=(scope, mode, vis, pk), disabled=disabled)
                else:
                    design.html('<div class="ts-note">No feature matches these filters.</div>')
            sel = _selection(scope, mode)
            hidden = [f for f in CATALOG_BY_NAME if CATALOG_BY_NAME[f].group == "competition" and f not in allowed]
            if hidden:
                design.html(f'<div class="ts-note">{len(hidden)} same-day hotel observations (New Arrivals, Same-Day Guests) are not offered in What-if mode: '
                            'they are unknown on any scenario date.</div>')
            with st.container(horizontal=True, gap="small", vertical_alignment="center", key="ts_feat_actions"):
                if disabled:
                    st.markdown(f'<span class="ts-chip grey">{ICON["info"]}Baseline uses no features</span>', unsafe_allow_html=True)
                elif not sel:
                    st.markdown(f'<span class="ts-chip bad">{ICON["warning"]}Select at least one feature</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="ts-chip ok">{ICON["check"]}{len(sel)} features selected · ready for training</span>', unsafe_allow_html=True)
                st.button("Clear", key="tr_clear", on_click=_set_selection, args=(scope, mode, []), disabled=disabled)
                st.button("All compatible", key="tr_all", on_click=_set_selection, args=(scope, mode, list(allowed)), disabled=disabled)
                st.button("Recommended set", key="tr_rec", on_click=_set_selection, args=(scope, mode, default_features(scope, mode)), disabled=disabled,
                          help="The catalogue's recommended defaults for this scope and mode")
                st.button("From Feature Engineering", key="tr_fe", on_click=_set_selection, args=(scope, mode, selected_features(scope, mode)),
                          disabled=disabled, help="The selection applied on the Feature Engineering page", icon=":material/arrow_back:")
            with st.expander(f"Feature details — {len(vis)} shown", icon=":material/table_rows:"):
                det = pd.DataFrame([{"selected": f in sel, "feature": f, "label": CATALOG_BY_NAME[f].label, "group": GROUP_TABS[CATALOG_BY_NAME[f].group],
                                     "definition": CATALOG_BY_NAME[f].definition, "source": CATALOG_BY_NAME[f].source,
                                     "type": CATALOG_BY_NAME[f].kind, "missing_pct": rows.loc[f, "missing_pct"] if f in rows.index else np.nan,
                                     "whatif_eligible": "whatif" in CATALOG_BY_NAME[f].modes, "leakage": rows.loc[f, "leakage"] if f in rows.index else "",
                                     "changes_in_scenarios": rows.loc[f, "scenario"] if f in rows.index else ""} for f in vis])
                st.dataframe(det, hide_index=True, width="stretch", height=300, column_config={
                    "selected": st.column_config.CheckboxColumn("Sel."), "definition": st.column_config.TextColumn(width="large"),
                    "missing_pct": st.column_config.NumberColumn("Missing %", format="%.2f"), "whatif_eligible": st.column_config.CheckboxColumn("What-if")})

        # ------------------------------------------------------------ hyperparameters + train
        with right, st.container(key="ds_panel_ts_hp"):
            h1, h2 = st.columns([2, 1], vertical_alignment="center")
            h1.markdown('<div class="ts-title">Hyperparameters</div>', unsafe_allow_html=True)
            h2.button("Reset", icon=":material/restart_alt:", key="tr_hp_reset", on_click=_reset_hp, args=(algo,), width="stretch")
            params = {}
            hp = HYPERPARAMS[algo]
            for name, spec in hp.items():
                params[name] = _param_widget(algo, name, spec, key=f"hp_{algo}_{name}")
            if not hp:
                design.html('<div class="ts-note">The seasonal-naive baseline has no hyperparameters: ŷ(t) = Guests(t − 364), falling back to last '
                            "year's 7-day mean and then the training median.</div>")
            mono_feats = monotone_features(algo, sel, True)
            monotone = st.toggle("Scenario-safe shape constraint", value=algo in MONOTONE_ALGORITHMS, key=f"tr_mono_{algo}", disabled=algo not in MONOTONE_ALGORITHMS,
                                 help=("Holding everything else fixed, more arriving seats / passengers / flights cannot lower predicted guests. Applied only "
                                       f"to these selected features: {', '.join(mono_feats) if mono_feats else 'none selected'}. Ratios, growth, calendar and "
                                       "history stay unconstrained.") if algo in MONOTONE_ALGORITHMS else
                                 f"{MI.ALGORITHMS.get(algo, algo)} does not support monotone constraints in this implementation.")
            adv = st.expander("Advanced options", icon=":material/tune:")
            seed = adv.number_input("Random seed", min_value=0, max_value=10_000, value=config.RANDOM_STATE, step=1, key="tr_seed",
                                   disabled=algo in ("ridge", "seasonal_naive", "svr"), help="Seeds tree sampling; Ridge, SVR and the baseline are deterministic.")
            log_target = adv.toggle("Log-transform target (Guests)", value=scope == "international_market" and algo not in ("ridge", "seasonal_naive"),
                                   key=f"tr_log_{scope}_{algo}", disabled=algo == "seasonal_naive",
                                   help="Fits log(1 + Guests) and inverts predictions with exp(·) − 1; every metric is in guests. Useful when market sizes "
                                        "differ by 100×. Not recommended for Ridge, whose features are not logged.")
            neg = adv.selectbox("Negative predictions", list(NEGATIVE_POLICIES), format_func=NEGATIVE_POLICIES.get, key=f"tr_neg_{algo}",
                               index=0 if algo in ("ridge", "svr") else 1, disabled=algo == "seasonal_naive",
                               help="Ridge and SVR can predict below zero for small markets. The policy is saved with the model; the raw count is always reported.")
            label = st.text_input("Model label (optional)", key="tr_label", placeholder="e.g. CatBoost — seats + calendar")
            go_train = st.button("Train model", type="primary", icon=":material/play_arrow:", key="tr_go", width="stretch",
                                 disabled=(not sel and algo != "seasonal_naive"))
            design.html('<div class="ts-note" style="text-align:center">Nothing trains until you press the button.</div>')
            if algo == "svr" and scope == "international_market":
                design.html(_box("amber", "SVR on the 29k-row market panel takes about 7–8 minutes (kernel methods scale quadratically).", "clock"))

        # ------------------------------------------------------------ validation settings
        with left, st.container(key="ds_panel_ts_val"):
            design.html('<div class="ts-title">Validation settings</div><div class="ts-sub">Chronological only — no random splits. Every row of a date stays on one '
                        'side of every split; preprocessing is fitted inside each fold.</div>')
            v = st.columns([1.1, 1, 1], vertical_alignment="bottom")
            hs = v[0].date_input("Chronological holdout starts", config.DEFAULT_HOLDOUT_START.date(),
                                 min_value=(config.MODEL_START + pd.Timedelta(days=270)).date(), max_value=(config.MODEL_END - pd.Timedelta(days=28)).date(), key="tr_hs")
            folds_n = v[1].slider("Expanding-window CV folds", 0, 8, 4 if scope != "international_market" else 3, key="tr_folds")
            horizon = v[2].slider("CV fold length (days)", 14, 180, 91, 7, key="tr_h")
            hs_t = pd.Timestamp(hs)
            fit_rows = tr[tr["Date"] < hs_t]
            hold = tr[tr["Date"] >= hs_t]
            folds = cv_folds(fit_rows["Date"], hs_t, folds_n, horizon) if folds_n else []
            fold_rows = [{"fold": i + 1, "train_start": str(fit_rows["Date"].min().date()), "train_end": str((vs - pd.Timedelta(days=1)).date()),
                          "val_start": str(vs.date()), "val_end": str(ve.date()), "train_rows": int((fit_rows["Date"] < vs).sum()),
                          "val_rows": int(fit_rows["Date"].between(vs, ve).sum()), "val_dates": int(fit_rows.loc[fit_rows["Date"].between(vs, ve), "Date"].nunique())}
                         for i, (vs, ve) in enumerate(folds)]
            plot(EC.timeline(str(config.MODEL_START.date()), fold_rows, (str(hs), str(config.MODEL_END.date())), height=150 + 26 * len(fold_rows)), key="tr_timeline")
            dropped = folds_n - len(folds)
            design.html(
                f'<div class="ts-ctx"><span>Modelling rows <b>{len(tr):,}</b> ({tr["Date"].nunique():,} dates)</span>'
                f'<span>Fit <b>{config.MODEL_START:%Y-%m-%d} → {(hs_t - pd.Timedelta(days=1)):%Y-%m-%d}</b> · {len(fit_rows):,} rows</span>'
                f'<span>Holdout <b>{hs} → {config.MODEL_END:%Y-%m-%d}</b> · {len(hold):,} rows, {hold["Date"].nunique():,} dates</span>'
                f'<span>CV folds <b>{len(folds)}</b>{f" ({dropped} dropped: each fold needs ≥ 180 training days)" if dropped else ""}</span></div>')
            if fold_rows:
                with st.expander("Fold dates and row counts", icon=":material/table_rows:"):
                    st.dataframe(pd.DataFrame(fold_rows).rename(columns={"fold": "Fold", "train_start": "Train from", "train_end": "Train to", "val_start": "Validate from",
                                                                         "val_end": "Validate to", "train_rows": "Train rows", "val_rows": "Validation rows",
                                                                         "val_dates": "Validation dates"}),
                                 hide_index=True, width="stretch")
            design.html('<div class="ts-note">The holdout is never used to fit, select features or tune. Tuning by watching holdout scores would leak it — '
                        'compare settings on the CV mean instead. The saved deployment model is refitted on all modelling rows after evaluation.</div>')

    # ---------------------------------------------------------------- training run (real monitor)
    if go_train:
        cfg = TrainConfig(scope=scope, mode=mode, algorithm=algo, features=sel if algo != "seasonal_naive" else [], params=params,
                          log_target=log_target, monotone=monotone, holdout_start=str(hs), cv_folds=folds_n, cv_horizon_days=horizon, label=label,
                          seed=int(seed), negative_policy=neg if algo != "seasonal_naive" else "none")
        hist = st.session_state.setdefault("tr_history", [])
        with st.status(f"Training {MI.ALGORITHMS.get(algo, algo)} · {SCOPE_SHORT[scope]} · {len(cfg.features)} features", expanded=True) as status:
            t0 = time.perf_counter()

            def progress(frac: float, msg: str) -> None:
                status.update(label=f"Running — {msg}")
                status.markdown(f'<div class="ts-stage"><b>{time.perf_counter() - t0:6.1f}s</b> · {escape(msg)}</div>', unsafe_allow_html=True)
            try:
                res = train(cfg, a, progress)
                st.session_state["last_trained"] = res.model_id
                hist.insert(0, {"finished": datetime.now().strftime("%H:%M:%S"), "model_id": res.model_id, "status": "completed",
                                "runtime_s": res.metadata["runtime_seconds"], "algorithm": algos[algo].split(" (")[0], "scope": scope, "features": len(cfg.features)})
                status.update(label=f"Training complete in {res.metadata['runtime_seconds']:.1f}s", state="complete", expanded=False)
            except Exception as e:  # surface real errors, never fake success
                hist.insert(0, {"finished": datetime.now().strftime("%H:%M:%S"), "model_id": "", "status": f"failed: {e}", "runtime_s": round(time.perf_counter() - t0, 1),
                                "algorithm": algos[algo].split(" (")[0], "scope": scope, "features": len(cfg.features)})
                status.update(label=f"Training failed: {e}", state="error")
                st.error(f"{type(e).__name__}: {e}")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc(limit=3))

    # ---------------------------------------------------------------- latest result + diagnostics
    last = st.session_state.get("last_trained")
    md = None
    if last:
        try:
            md = registry.load_metadata(last)
        except FileNotFoundError:
            md = None
    if md:
        _result_panel(md)

    hist = st.session_state.get("tr_history", [])
    if hist:
        with st.expander(f"Training runs this session ({len(hist)})", icon=":material/history:"):
            st.dataframe(pd.DataFrame(hist), hide_index=True, width="stretch")

    # ---------------------------------------------------------------- registry + ablation (tabs)
    st.write("")
    with st.container(key="ts_lower"):
        lower = st.tabs(["Model registry", "Feature ablation study"], key="tr_lower_nav", on_change="rerun")
        if lower[0].open:
            with lower[0]:
                _registry_panel()
        if lower[1].open:
            with lower[1]:
                _ablation_panel(a)
    next_step("train")



def _result_panel(md: dict) -> None:
    mid = md["model_id"]
    hold = registry.load_table(mid, "predictions_holdout")
    m = metrics_from_predictions(hold)
    cvp = registry.load_table(mid, "predictions_cv")
    ft = fold_table(cvp, md)
    dg = diagnostics(hold)
    st.write("")
    with st.container(key="ds_panel_ts_result"):
        top = st.columns([3, 1.2], vertical_alignment="center")
        top[0].markdown(f'<div class="ts-title">Latest training result <span class="ts-chip ok" style="margin-left:8px">{ICON["check"]}Completed</span></div>'
                        f'<div class="ts-sub">{escape(_label(md))} · {escape(MI.validation(md))}</div>', unsafe_allow_html=True)
        with top[1]:
            if MI.scenario_ready(md):
                if mid == active_id():
                    st.markdown(f'<span class="ts-chip info">{ICON["check"]}Active in the What-if Simulator</span>', unsafe_allow_html=True)
                elif st.button("Activate for Simulator", type="primary", icon=":material/bolt:", key="act_last", width="stretch"):
                    registry.activate(mid)
                    st.toast("Activated for the What-if Simulator.", icon=":material/bolt:")
                    st.rerun()
            else:
                st.markdown(f'<span class="ts-chip warn" title="{escape(MI.not_ready_reason(md) or "")}">{ICON["lock"]}Not eligible for the simulator</span>',
                            unsafe_allow_html=True)
        cv_w = ft["WMAPE"] if len(ft) else pd.Series(dtype=float)
        design.html(
            '<div class="ts-metrics">'
            f'<div class="ts-m"><span>Holdout MAE</span><b>{_fmt(m["MAE"])}</b><small>guests per {"nationality-day" if md["scope"] == "international_market" else "day"}</small></div>'
            f'<div class="ts-m"><span>Holdout RMSE</span><b>{_fmt(m["RMSE"])}</b><small>guests (penalises large misses)</small></div>'
            f'<div class="ts-m"><span>Holdout R²</span><b>{_fmt(m["R2"], 3)}</b><small>{md["periods"]["holdout"][0]} → {md["periods"]["holdout"][1]}</small></div>'
            f'<div class="ts-m"><span>Holdout WMAPE</span><b>{m["WMAPE"]:.1%}</b><small>Σ|error| ÷ Σ actual</small></div>'
            f'<div class="ts-m"><span>CV WMAPE</span><b>{(f"{cv_w.mean():.1%}" if len(cv_w) else "—")}</b><small>{(f"± {cv_w.std():.1%} across {len(cv_w)} folds" if len(cv_w) > 1 else "no CV folds")}</small></div>'
            f'<div class="ts-m"><span>Training time</span><b>{md["runtime_seconds"]:.1f}s</b><small>{md["n_rows"]["modelling"]:,} rows · {len(md["features"])} features</small></div></div>')
        neg = md.get("negative_predictions_raw")
        design.html(f'<div class="ts-ctx"><span>{escape(MI.target(md))} · {escape(MI.scope(md).lower())}</span>'
                    f'<span>Scored on <b>{len(hold):,}</b> unseen {MI.row_unit(md)}</span>'
                    f'<span>On average the model {"under" if dg["mean_residual"] > 0 else "over"}-predicts by <b>{abs(dg["mean_residual"]):,.0f}</b> guests</span></div>')
        with st.expander("Developer diagnostics", icon=":material/code:"):
            st.markdown(f"- **Model ID:** `{mid}`\n- **Grain:** {md.get('grain', '')}\n"
                        f"- **Target transform:** {md.get('target_transform', 'log1p/expm1' if md.get('log_target') else 'none')}\n"
                        f"- **Monotone features:** {len(md.get('monotone_features') or []) if md.get('monotone') else 0}\n"
                        f"- **Negative predictions (raw, holdout):** {neg['holdout'] if neg else dg.get('negative_predictions', 0)} · policy: "
                        f"{NEGATIVE_POLICIES.get(md.get('negative_policy', 'none'), 'keep raw')}\n"
                        f"- **Parallelism:** tree models use all CPU cores (n_jobs = −1); results are reproducible for a fixed seed ({md.get('seed', '—')})\n"
                        f"- **Software:** " + ", ".join(f"{k} {v}" for k, v in (md.get('versions') or {}).items()))
        if md["mode"] == "competition":
            design.html(_box("amber", "Competition model: it uses same-day hotel observations that only exist retrospectively (New Arrivals, Same-Day Guests). "
                             "Its accuracy reflects nowcasting with those fields and is not comparable with What-if models.", "warning"))
        tabs = st.tabs(["Actual vs predicted", "Residuals vs predicted", "Residual distribution", "Over time", "CV folds", "Feature importance"])
        daily = hold.groupby("Date", as_index=False)[["y_true", "y_pred"]].sum() if "market_key" in hold else hold
        with tabs[0]:
            plot(EC.actual_vs_pred(hold, height=400), key="tr_avp")
        with tabs[1]:
            plot(EC.residuals_vs_pred(hold), key="tr_rvp")
        with tabs[2]:
            plot(EC.residual_hist(hold), key="tr_rh")
        with tabs[3]:
            plot(EC.over_time(daily), key="tr_ts")
            if "market_key" in hold:
                design.html('<div class="ts-note">Market-level predictions summed to daily totals.</div>')
        with tabs[4]:
            if len(ft):
                plot(EC.fold_bars(ft, "MAE", m["MAE"]), key="tr_folds_fig")
                st.dataframe(ft[["fold", "train_start", "train_end", "val_start", "val_end", "train_rows", "val_rows", "val_dates", "MAE", "RMSE", "R2", "WMAPE"]],
                             hide_index=True, width="stretch", column_config={"WMAPE": st.column_config.NumberColumn(format="percent"),
                                                                             "MAE": st.column_config.NumberColumn(format="%,.1f"), "RMSE": st.column_config.NumberColumn(format="%,.1f"),
                                                                             "R2": st.column_config.NumberColumn(format="%.3f")})
            else:
                design.html('<div class="ts-note">This run had no CV folds.</div>')
        with tabs[5]:
            imp = registry.load_table(mid, "importance")
            if imp is not None and imp["permutation_rmse_increase"].notna().any():
                plot(EC.importance_bars(imp, "permutation_rmse_increase", xlab="Holdout RMSE increase when the feature is shuffled (guests)"), key="tr_imp")
                design.html('<div class="ts-note">Permutation importance shows what this model relies on for prediction, not causal influence; correlated '
                            f'aviation features share credit. Native view ({escape(importance_kind(md["algorithm"]))}) is on the Model Evaluation page.</div>')
            else:
                design.html('<div class="ts-note">No learned feature importance for this model.</div>')
        e1, e2, e3 = st.columns([1, 1, 2.2])
        with e1:
            csv_button(hold, f"{mid}_predictions_holdout.csv", "Holdout predictions (CSV)", key="tr_dl_hold")
        with e2:
            st.page_link(_eval_page(), label="Open in Model Evaluation", icon=":material/monitoring:")


def _eval_page():
    from .pages import PAGES
    return PAGES["eval"]


def _ablation_panel(a) -> None:
    with st.container(key="ds_panel_ts_abl"):
        design.html('<div class="ts-title">Feature ablation study</div><div class="ts-sub">Retrains the chosen model with feature families removed or isolated, '
                    'on the identical chronological split (no CV). It shows predictive contribution under this setup — not causal importance.</div>')
        df = _registry_frame(_registry_stamp())
        df = df[df["model_id"].map(lambda m: not m.split("-")[2] == "seasonal_naive")] if len(df) else df
        if df.empty:
            design.html('<div class="ts-note">Train a model first.</div>')
            return
        ids = df["model_id"].tolist()
        last = st.session_state.get("last_trained")
        x1, x2 = st.columns([1.6, 2])
        base_id = x1.selectbox("Base model", ids, index=ids.index(last) if last in ids else 0, key="tr_abl_base",
                               format_func=lambda m: df.set_index("model_id").loc[m, "model"])
        with x2:
            exps = st.pills("Experiments", list(ABLATION_EXPERIMENTS), selection_mode="multi", default=list(ABLATION_EXPERIMENTS), key="tr_abl_exps") or []
        bmd = registry.load_metadata(base_id)
        if st.button("Run ablation study", icon=":material/science:", key="tr_abl_go"):
            cfg = TrainConfig(**{**{k: v for k, v in bmd["config"].items() if k in TrainConfig.__dataclass_fields__}, "label": f"ablation of {base_id}"})
            with st.status("Running ablation — one model per experiment", expanded=True) as status:
                t0 = time.perf_counter()
                abl_id, summary = run_ablation(cfg, a, lambda f, msg: status.markdown(f'<div class="ts-stage"><b>{time.perf_counter() - t0:6.1f}s</b> · {escape(msg)}</div>',
                                                                                        unsafe_allow_html=True),
                                               experiments=exps, source_model_id=base_id)
                status.update(label=f"Ablation saved as {abl_id}", state="complete", expanded=False)
            st.session_state["tr_last_abl"] = abl_id
        abl_id = st.session_state.get("tr_last_abl")
        saved = _ablations_for(base_id, bmd)
        if not abl_id or abl_id not in saved:
            abl_id = saved[0] if saved else None
        if abl_id:
            _ablation_result(abl_id)
        else:
            design.html('<div class="ts-note">No ablation study saved for this model yet.</div>')


def _ablations_for(base_id: str, md: dict) -> list[str]:
    if not ABLATIONS_DIR.exists():
        return []
    out = []
    for p in sorted(ABLATIONS_DIR.glob("*"), reverse=True):
        try:
            c = json.loads((p / "config.json").read_text())
        except Exception:
            continue
        same_src = c.get("source_model_id") == base_id
        same_cfg = c.get("source_model_id") is None and all(c.get(k) == md["config"].get(k) for k in ("scope", "mode", "algorithm", "holdout_start")) \
            and set(c.get("features", [])) == set(md["features"])
        if same_src or same_cfg:
            out.append(p.name)
    return out


def _ablation_result(abl_id: str) -> None:
    d = ABLATIONS_DIR / abl_id
    pred = pd.read_csv(d / "predictions_holdout.csv", parse_dates=["Date"])
    cfg = json.loads((d / "config.json").read_text())
    rows = [{"feature_set": fs, "n_features": None, **metrics_from_predictions(g)} for fs, g in pred.groupby("feature_set", sort=False)]
    summ = pd.DataFrame(rows)
    info = pd.read_csv(d / "summary.csv")[["feature_set", "n_features", "features"]]
    summ = summ.drop(columns="n_features").merge(info, on="feature_set", how="left")
    base_name = "All selected features" if "All selected features" in set(summ["feature_set"]) else summ.iloc[-1]["feature_set"]
    b = summ.set_index("feature_set").loc[base_name]
    for k in ("MAE", "RMSE", "R2", "WMAPE"):
        summ[f"Δ{k}"] = summ[k] - b[k]
    design.html(f'<div class="ts-ctx"><span>Study <b>{escape(abl_id)}</b></span><span>Algorithm <b>{escape(cfg["algorithm"])}</b></span>'
                f'<span>Identical holdout <b>{pred["Date"].min():%Y-%m-%d} → {pred["Date"].max():%Y-%m-%d}</b></span>'
                f'<span>Baseline <b>{escape(base_name)}</b></span><span>Metrics recomputed from the saved predictions</span></div>')
    plot(EC.metric_bars(summ.assign(label=summ["feature_set"]), "MAE", "label", highlight=base_name), key=f"tr_abl_fig_{abl_id}")
    st.dataframe(summ[["feature_set", "n_features", "MAE", "ΔMAE", "RMSE", "ΔRMSE", "R2", "ΔR2", "WMAPE", "ΔWMAPE", "features"]], hide_index=True, width="stretch",
                 column_config={"MAE": st.column_config.NumberColumn(format="%,.1f"), "ΔMAE": st.column_config.NumberColumn(format="%+,.1f"),
                                "RMSE": st.column_config.NumberColumn(format="%,.1f"), "ΔRMSE": st.column_config.NumberColumn(format="%+,.1f"),
                                "R2": st.column_config.NumberColumn(format="%.3f"), "ΔR2": st.column_config.NumberColumn(format="%+.3f"),
                                "WMAPE": st.column_config.NumberColumn(format="percent"), "ΔWMAPE": st.column_config.NumberColumn(format="%+.4f"),
                                "features": st.column_config.TextColumn(width="large")})
    csv_button(summ, f"{abl_id}_summary.csv", "Download ablation table (CSV)", key=f"tr_abl_csv_{abl_id}")


def _registry_panel() -> None:
    with st.container(key="ds_panel_ts_reg"):
        design.html('<div class="ts-title">Model registry</div><div class="ts-sub">Every saved pipeline with its holdout metrics. Select rows to view details, '
                    'activate, export, compare or delete. Reference models are protected so the Judges’ Demo stays reproducible.</div>')
        df = _registry_frame(_registry_stamp())
        if df.empty:
            design.html('<div class="ts-note">No saved models yet.</div>')
            return
        act = active_id()
        df.insert(0, "active", df["model_id"].eq(act).map({True: "● active", False: ""}))
        f = st.columns([1.6, 1, 1, 1, 0.9], vertical_alignment="bottom")
        q = f[0].text_input("Search", placeholder="Search model, algorithm or id…", icon=":material/search:", key="reg_q", label_visibility="collapsed")
        sc = f[1].multiselect("Scope", list(config.SCOPES), format_func=lambda s: SCOPE_SHORT[s].split(" (")[0], key="reg_scope", placeholder="All scopes", label_visibility="collapsed")
        mo = f[2].multiselect("Mode", ["What-if", "Competition"], key="reg_mode", placeholder="All modes", label_visibility="collapsed")
        al = f[3].multiselect("Algorithm", sorted(df["algorithm"].unique()), key="reg_algo", placeholder="All algorithms", label_visibility="collapsed")
        kind = f[4].selectbox("Type", ["All", "User-trained", "Reference"], key="reg_kind", label_visibility="collapsed")
        v = df.copy()
        if q:
            ql = q.lower()
            v = v[(v["model"] + " " + v["model_id"] + " " + v["algorithm"]).str.lower().str.contains(ql, regex=False)]
        if sc:
            v = v[v["scope"].isin(sc)]
        if mo:
            v = v[v["mode"].isin(mo)]
        if al:
            v = v[v["algorithm"].isin(al)]
        if kind != "All":
            v = v[v["reference"] == (kind == "Reference")]
        size = 10
        pages = max(1, math.ceil(len(v) / size))
        page = min(st.session_state.get("reg_page", 1), pages)
        view = v.iloc[(page - 1) * size: page * size].reset_index(drop=True)
        event = st.dataframe(view[["active", "model", "mode", "MAE", "R2", "WMAPE", "CV_WMAPE", "scenario_ready", "trained"]], hide_index=True,
                             width="stretch", on_select="rerun", selection_mode="multi-row",
                             key=f"reg_table_{page}_{abs(hash((q, tuple(sc), tuple(mo), tuple(al), kind))) % 10**6}", column_config={
            "active": st.column_config.TextColumn("", width="small"), "model": st.column_config.TextColumn("Model", width="large"),
            "MAE": st.column_config.NumberColumn("Holdout MAE", format="%,.1f"), "RMSE": st.column_config.NumberColumn(format="%,.1f"),
            "R2": st.column_config.NumberColumn("R²", format="%.3f"), "WMAPE": st.column_config.NumberColumn(format="percent"),
            "CV_WMAPE": st.column_config.NumberColumn("CV WMAPE", format="percent"), "n_holdout": st.column_config.NumberColumn("Holdout rows"),
            "runtime_s": st.column_config.NumberColumn("Runtime (s)", format="%.1f"),
            "reference": st.column_config.CheckboxColumn("Reference"), "scenario_ready": st.column_config.CheckboxColumn("Simulator-ready"),
            "trained": st.column_config.TextColumn("Trained")})
        p1, p2 = st.columns([2.5, 1.5], vertical_alignment="center")
        p1.caption(f"{len(v)} of {len(df)} models · page {page} of {pages} · click a column header to sort")
        with p2, st.container(horizontal=True, horizontal_alignment="right", gap="small"):
            st.button("", icon=":material/chevron_left:", key="reg_prev", disabled=page <= 1, on_click=lambda: st.session_state.update(reg_page=page - 1))
            st.button("", icon=":material/chevron_right:", key="reg_next", disabled=page >= pages, on_click=lambda: st.session_state.update(reg_page=page + 1))
        picked = [view.loc[i, "model_id"] for i in (event.selection.rows if event and event.selection else [])]
        a = st.columns(5)
        one = picked[0] if len(picked) == 1 else None
        if a[0].button("View details", icon=":material/info:", key="reg_view", disabled=one is None, width="stretch"):
            _details_dialog(one)
        can_act = one is not None and MI.scenario_ready(registry.load_metadata(one))
        if a[1].button("Activate", icon=":material/bolt:", key="reg_act", disabled=not can_act, width="stretch",
                       help=(MI.not_ready_reason(registry.load_metadata(one)) if one is not None and not can_act
                             else "Make this model the one the What-if Simulator and Copilot use.")):
            registry.activate(one)
            st.toast("Activated for the What-if Simulator.", icon=":material/bolt:")
            st.rerun()
        with a[2]:
            if one is not None:
                st.download_button("Export metadata", json.dumps(registry.load_metadata(one), indent=2, default=str).encode(), file_name=f"{one}_metadata.json",
                                   mime="application/json", icon=":material/download:", key="reg_export", width="stretch", on_click="ignore")
            else:
                st.button("Export metadata", icon=":material/download:", key="reg_export_d", disabled=True, width="stretch")
        can_del = one is not None and not registry.load_metadata(one).get("reference")
        if a[3].button("Delete…", icon=":material/delete:", key="reg_del", disabled=not can_del, width="stretch",
                       help="Reference models cannot be deleted." if one and not can_del else None):
            _delete_dialog(one)
        a[4].button(f"Compare ({len(picked)})", icon=":material/compare_arrows:", key="reg_cmp", disabled=len(picked) < 2, width="stretch",
                    on_click=lambda: st.session_state.update(reg_compare=list(picked)))
        if len(picked) < 2 and not st.session_state.get("reg_compare"):
            design.html('<div class="ts-note">Tip: select two or more rows to compare them.</div>')
        cmp = st.session_state.get("reg_compare")
        if cmp:
            _comparison(cmp)


def _comparison(ids: list[str]) -> None:
    rows, groups = [], {}
    for mid in ids:
        try:
            md = registry.load_metadata(mid)
        except FileNotFoundError:
            continue
        h = registry.load_table(mid, "predictions_holdout")
        k = comparison_key(md, h)
        groups.setdefault(k, []).append(mid)
        m = metrics_from_predictions(h)
        cvp = registry.load_table(mid, "predictions_cv")
        ft = fold_table(cvp, md)
        rows.append({"model_id": mid, "model": _label(md), "algorithm": MI.algorithm(md), "scope": md["scope"],
                     "mode": md["mode"], "holdout": " → ".join(md["periods"]["holdout"]), "features": len(md["features"]),
                     **{k_: m[k_] for k_ in ("MAE", "RMSE", "R2", "WMAPE")}, "CV_WMAPE_mean": ft["WMAPE"].mean() if len(ft) else np.nan,
                     "CV_WMAPE_sd": ft["WMAPE"].std() if len(ft) > 1 else np.nan, "runtime_s": md.get("runtime_seconds"),
                     "scenario_ready": MI.scenario_ready(md), "group": None})
    tab = pd.DataFrame(rows)
    gid = {k: i + 1 for i, k in enumerate(groups)}
    for k, members in groups.items():
        tab.loc[tab["model_id"].isin(members), "group"] = gid[k]
    st.write("")
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    c1.markdown('<div class="ts-title">Model comparison</div>', unsafe_allow_html=True)
    c2.button("Close comparison", icon=":material/close:", key="reg_cmp_close", on_click=lambda: st.session_state.pop("reg_compare", None), width="stretch")
    if len(groups) > 1:
        design.html(_box("amber", f"These models fall into {len(groups)} comparison groups (different scope, target grain or evaluation rows). "
                         "They are ranked only within their own group — comparing across groups is not apples-to-apples.", "warning"))
    else:
        design.html(_box("green", "All selected models share the same target, grain and evaluation rows, so they can be ranked directly.", "check"))
    for k, members in groups.items():
        g = tab[tab["model_id"].isin(members)].sort_values("MAE")
        if len(g) > 1:
            design.html(f'<div class="ts-note" style="margin-top:6px"><b>Group {gid[k]}</b> · {escape(SCOPE_SHORT[g.iloc[0]["scope"]])} · holdout {escape(g.iloc[0]["holdout"])} · '
                        f'best MAE within this group: <b>{escape(g.iloc[0]["model"])}</b></div>')
            plot(EC.metric_bars(g, "MAE", "model", highlight=g.iloc[0]["model_id"]), key=f"reg_cmp_fig_{gid[k]}")
    st.dataframe(tab.drop(columns=["model_id"]), hide_index=True, width="stretch", column_config={
        "MAE": st.column_config.NumberColumn(format="%,.1f"), "RMSE": st.column_config.NumberColumn(format="%,.1f"),
        "R2": st.column_config.NumberColumn(format="%.3f"), "WMAPE": st.column_config.NumberColumn(format="percent"),
        "CV_WMAPE_mean": st.column_config.NumberColumn("CV WMAPE mean", format="percent"), "CV_WMAPE_sd": st.column_config.NumberColumn("CV WMAPE sd", format="percent"),
        "scenario_ready": st.column_config.CheckboxColumn("Simulator-ready"), "group": st.column_config.NumberColumn("Comparison group")})
    csv_button(tab, "model_comparison.csv", "Download comparison (CSV)", key="reg_cmp_csv")


@st.dialog("Model details", width="large")
def _details_dialog(mid: str) -> None:
    md = registry.load_metadata(mid)
    st.markdown(f"**{_label(md)}**  \n`{mid}`")
    ident = {"Algorithm": MI.algorithm(md), "Prediction target": MI.target(md), "Model scope": MI.scope(md), "Validation": MI.validation(md),
             "Scope": md["scope_label"], "Grain": md.get("grain"), "Mode": md["mode_label"], "Target": md.get("target"),
             "Created": md["created_at"], "Reference model": "yes" if md.get("reference") else "no",
             "Simulator-ready": "yes" if MI.scenario_ready(md) else f"no — {MI.not_ready_reason(md)}",
             "Target transform": md.get("target_transform", "log1p/expm1" if md.get("log_target") else "none"),
             "Negative-prediction policy": NEGATIVE_POLICIES.get(md.get("negative_policy", "none")),
             "Monotone features": ", ".join(md.get("monotone_features") or []) or ("yes (recorded before feature lists were saved)" if md.get("monotone") else "none"),
             "Random seed": md.get("seed", md.get("config", {}).get("seed")), "Matrix fingerprint": md.get("matrix_fingerprint", "—"),
             "Data fingerprint": (md.get("data_fingerprint") or {}).get("digest", "—")}
    st.dataframe(pd.DataFrame(ident.items(), columns=["Field", "Value"]).astype(str), hide_index=True, width="stretch")
    per = md["periods"]
    st.dataframe(pd.DataFrame([("Modelling", *per["modelling"], md["n_rows"]["modelling"]), ("Evaluation fit", *per["eval_fit"], md["n_rows"]["eval_fit"]),
                               ("Holdout", *per["holdout"], md["n_rows"]["holdout"]), ("Deployment fit", *per["deployment_fit"], md["n_rows"]["modelling"])],
                              columns=["Period", "From", "To", "Rows"]), hide_index=True, width="stretch")
    st.markdown("**Features (in pipeline order)**")
    st.write(", ".join(md["features"]) or "none")
    st.markdown("**Hyperparameters**")
    st.dataframe(pd.DataFrame([(k, str(v)) for k, v in md["params"].items()], columns=["Parameter", "Value"]), hide_index=True, width="stretch")
    st.markdown("**Software versions**")
    st.dataframe(pd.DataFrame([(k, str(v)) for k, v in md["versions"].items()], columns=["Package", "Version"]), hide_index=True, width="stretch")


@st.dialog("Delete model")
def _delete_dialog(mid: str) -> None:
    md = registry.load_metadata(mid)
    st.markdown(f"Delete **{_label(md)}**?  \n`{mid}`")
    st.caption("This removes the saved pipeline, predictions and metadata from this project. Reference models cannot be deleted.")
    ok = st.checkbox("I understand this cannot be undone", key="reg_del_ok")
    if st.button("Delete permanently", type="primary", disabled=not ok, key="reg_del_confirm"):
        registry.delete(mid)
        if st.session_state.get("last_trained") == mid:
            st.session_state.pop("last_trained", None)
        st.session_state.pop("reg_compare", None)
        st.rerun()
