"""Section 3 — Feature Engineering.

The catalogue table is built from native Streamlit widgets (one checkbox per feature), so every
tick changes the real selection. Two selections are kept per scope × mode:

  draft    — what is ticked on this page (`fe_draft`)
  applied  — what the ML Training Studio uses (`feature_selection`), set by "Apply Selected Features"

Every number on the page comes from `feature_facts`, i.e. from the same model frame the training
code fits on.
"""
from __future__ import annotations

import math
from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from .. import config
from ..feature_facts import (GROUP_LABEL, GROUP_ORDER, assumptions, empty_same_day_cells, feature_rows, matrix_columns,
                             matrix_fingerprint, missing_reason, missing_treatment, scenario_sensitivity, training_matrix, zero_days, _FORMULA, _UNIT,
                             _window)
from ..features import CATALOG_BY_NAME, REJECTED_FEATURES, available_features, default_features
from . import design, heroes
from .common import analytic, next_step
from .design import ICON

SPEC = [0.36, 1.38, 1.9, 0.95, 1.35, 1.12, 1.1, 1.05, 0.5]
LEAK_TONE = {"None": "green", "Low": "blue", "High": "red", "Not allowed": "grey"}
SCEN_TONE = {"Yes": "green", "Indirect": "amber", "No": "grey"}
GROUP_TONE = {"aviation": "blue", "time": "violet", "derived": "teal", "historical": "indigo", "market": "slate", "competition": "orange"}
SCOPE_HINT = {
    "international_total": "Forecast total international guests per day (45 nationalities summed).",
    "international_market": "Forecast hotel demand for each source market × nationality.",
    "domestic": "Forecast domestic (UAE-resident) guests per day.",
    "overall": "Forecast international + domestic guests per day.",
}
PAGE_SIZES = [10, 20, 50, 0]
SCOPE_SHORT = {
    "international_total": "International guests · daily total",
    "international_market": "International guests · by nationality",
    "domestic": "Domestic guests · daily total",
    "overall": "All guests · daily total",
}
# when a feature's value is known, relative to the date being predicted (quanara.leakage.timing_class)
KNOWN = {"calendar": ("In advance", "green"), "identifier": ("In advance", "green"), "schedule": ("Schedule (ahead)", "green"),
         "history": ("364+ days before", "green"), "lagged_realised": ("7+ days before", "green"), "realised": ("On the day", "amber"),
         "retrospective": ("After the day", "red")}
TAB_LABEL = {"aviation": "Aviation", "time": "Time", "derived": "Derived", "historical": "Historical", "market": "Market", "competition": "Hotel"}

CSS = """
<style>
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-fe_card_"]) { height:100%; }
.st-key-fe_cards [data-testid="stHorizontalBlock"] { align-items:stretch; }
.st-key-fe_cards [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { height:100%; }
[class*="st-key-fe_card_"] { height:100%; background:#fff; border:1px solid #e7e8e7; border-radius:16px; padding:14px 16px 12px;
   box-shadow:0 1px 2px rgba(15,23,20,.04),0 8px 22px rgba(15,23,20,.05); animation:dsFadeUp .5s ease both; gap:.45rem; }
.st-key-fe_card_mode { animation-delay:.06s; } .st-key-fe_card_sel { animation-delay:.12s; }
.fe-ch { display:flex; gap:11px; align-items:center; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.fe-ch b { font-size:14px; color:#101614; font-weight:750; }
.fe-ch .ds-ico { width:36px; height:36px; } .fe-ch .ds-ico svg { width:19px; height:19px; }
.fe-cap { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12px; color:#6f7572; line-height:1.4; }
.fe-count { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; display:flex; align-items:baseline; gap:6px; margin-top:2px; }
.fe-count b { font-size:30px; font-weight:800; color:#059669; letter-spacing:-.02em; animation:fePop .45s cubic-bezier(.3,1.4,.6,1); display:inline-block; }
.fe-count span { font-size:22px; font-weight:700; color:#101614; } .fe-count small { font-size:12px; color:#6f7572; font-weight:600; margin-left:2px; }
@keyframes fePop { from { transform:scale(.7); opacity:.2; } to { transform:none; opacity:1; } }
.fe-state { display:inline-flex; gap:5px; align-items:center; font-size:12px; font-weight:650; border-radius:999px; padding:2px 9px; margin-top:4px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.fe-state svg { width:13px; height:13px; } .fe-state.ok { color:#047857; background:#e3f8ee; } .fe-state.warn { color:#8a5a00; background:#fff2d6; } .fe-state.bad { color:#a61b1b; background:#fde8e8; }
/* catalogue card */
.st-key-fe_catalog { background:#fff; border:1px solid #e7e8e7; border-radius:18px; padding:8px 18px 14px; box-shadow:0 1px 2px rgba(15,23,20,.04),0 10px 28px rgba(15,23,20,.06);
   animation:dsFadeUp .5s ease .1s both; gap:0; }
.st-key-fe_toolbar { border-bottom:1px solid #eceeed; padding:4px 0 6px; margin-bottom:6px; }
.st-key-fe_group [role="radiogroup"] { gap:4px; flex-wrap:wrap; background:#edf1ef; border-radius:12px; padding:4px; width:fit-content; }
.st-key-fe_group [data-testid="stRadioOption"] { padding:7px 11px; margin:0; border-radius:9px; transition:background .15s; cursor:pointer; }
.st-key-fe_group [data-testid="stRadioOption"] > div > div:not([data-testid]) { display:none; }
.st-key-fe_group [data-testid="stRadioOption"] p { font-size:13.3px; font-weight:650; color:#2b302d; white-space:nowrap; }
.st-key-fe_group [data-testid="stRadioOption"]:hover p { color:#047857; }
.st-key-fe_group [data-testid="stRadioOption"][data-selected="true"] { background:#047857; box-shadow:0 4px 12px rgba(4,120,87,.22); }
.st-key-fe_group [data-testid="stRadioOption"][data-selected="true"] p { color:#ffffff; }
.st-key-fe_q input { font-size:13.5px; }
/* table */
.st-key-fe_head { background:#f0faf7; border:1px solid #eceeed; border-radius:10px 10px 0 0; padding:2px 8px; }
.st-key-fe_head p, .fe-hd { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.5px !important; font-weight:650; color:#515553; }
.st-key-fe_head button { padding:0 !important; min-height:0 !important; max-width:none; }
.st-key-fe_head button p { font-size:12.3px !important; font-weight:650; color:#515553; white-space:nowrap; overflow:visible; text-overflow:clip; }
.st-key-fe_head button span[data-testid="stIconMaterial"] { font-size:15px !important; color:#9fa3a1; }
.fe-hd { white-space:nowrap; }
[class*="st-key-fe_row_"] [data-testid="stMarkdownContainer"], .st-key-fe_head [data-testid="stMarkdownContainer"] { margin-bottom:0 !important; }
[class*="st-key-fe_card_"] [data-testid="stMarkdownContainer"] { margin-bottom:0 !important; }
.st-key-fe_card_scope [data-baseweb="select"] div { font-size:13px; }
.st-key-fe_card_sel .fe-ch b { white-space:normal; line-height:1.25; }
.st-key-fe_card_mode [data-testid="stRadioOption"] p { font-size:13px; }
.st-key-fe_card_mode [role="radiogroup"] { gap:6px; }
.st-key-fe_card_sel button p { font-size:13.2px; white-space:nowrap; }
.fe-ch b { white-space:nowrap; }
[class*="st-key-fe_row_"] { border:1px solid #e8f2ee; border-top:none; padding:3px 8px; transition:background .15s; background:#fff; }
[class*="st-key-fe_row_"]:hover { background:#effdf8; }
[class*="st-key-fe_row_"].fe-open, [class*="st-key-fe_row_"]:has(+ [class*="st-key-fe_det_"]) { background:#e8fcf5; }
[class*="st-key-fe_row_"] [data-testid="stHorizontalBlock"], .st-key-fe_head [data-testid="stHorizontalBlock"] { gap:.6rem; }
[class*="st-key-fe_row_"] [data-testid="stMarkdownContainer"] p { margin:0; }
[class*="st-key-fe_row_"] [data-testid="stCheckbox"] label { min-height:0; }
[class*="st-key-fe_row_"] button[kind="tertiary"] { min-height:0; padding:2px; color:#6f7572; }
[class*="st-key-fe_row_"] button[kind="tertiary"]:hover { color:#059669; }
.fe-name { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:13.2px; font-weight:750; color:#101614; overflow-wrap:normal; word-break:normal; }
.fe-lab { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.8px; color:#313432; line-height:1.35; }
.fe-num { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:13px; color:#313432; font-variant-numeric:tabular-nums; }
.fe-num.hi { color:#c2410c; font-weight:650; } .fe-num.na { color:#9fa3a1; }
.fe-chip { display:inline-flex; align-items:center; gap:4px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12px; font-weight:650; border-radius:7px; padding:3px 8px; white-space:nowrap; line-height:1.35; }
.fe-chip svg { width:12px; height:12px; }
.fe-chip.blue { color:#047857; background:#ecfdf5; } .fe-chip.green { color:#047857; background:#e3f8ee; } .fe-chip.amber { color:#8a5a00; background:#fff2d6; }
.fe-chip.red { color:#a61b1b; background:#fde8e8; } .fe-chip.grey { color:#606562; background:#e9f3ef; } .fe-chip.violet { color:#065f46; background:#ecfdf5; }
.fe-chip.teal { color:#0f7f74; background:#e1f5f2; } .fe-chip.indigo { color:#1b5841; background:#c4f5e3; } .fe-chip.slate { color:#444846; background:#e9f3ef; }
.fe-chip.orange { color:#b4501f; background:#fdeee6; }
.fe-mono { display:inline-flex; width:22px; height:22px; border-radius:50%; align-items:center; justify-content:center; font-weight:800; font-size:14px; color:#047857; background:#e3f8ee; }
.fe-dash { color:#9fa3a1; font-size:14px; }
/* details panel */
[class*="st-key-fe_det_"] { border:1px solid #b0f5dc; border-top:none; background:#f6fefb; padding:4px 16px 14px; animation:feOpen .32s ease both; }
@keyframes feOpen { from { opacity:0; transform:translateY(-6px); } to { opacity:1; transform:none; } }
[class*="st-key-fe_det_"] [data-testid="stTab"] p { font-size:13px; }
[class*="st-key-fe_det_"] [role="tablist"] { flex-wrap:wrap; overflow:visible !important; }
.fe-dg { display:grid; grid-template-columns:1.25fr 1fr 1fr 1.15fr; gap:22px; padding:6px 2px 2px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.fe-dg h5 { font-size:13px; font-weight:750; color:#101614; margin:0 0 6px; }
.fe-dg p { font-size:12.8px; color:#444846; line-height:1.5; margin:0 0 6px; }
.fe-formula { background:#e6f8f1; border:1px solid #d0f0e4; border-radius:10px; padding:10px 12px; font-family:'Cambria Math','STIX Two Math','Times New Roman',serif;
   font-style:italic; font-size:15px; color:#1f2937; margin-top:4px; overflow-x:auto; white-space:nowrap; }
.fe-frac { display:inline-flex; flex-direction:column; vertical-align:middle; text-align:center; margin:0 4px; }
.fe-frac > span:first-child { border-bottom:1.2px solid #1f2937; padding:0 4px 2px; } .fe-frac > span:last-child { padding:2px 4px 0; }
.fe-src { display:flex; gap:6px; flex-wrap:wrap; margin:2px 0 8px; }
.fe-big { font-size:17px; font-weight:800; color:#101614; } .fe-big small { font-size:13px; font-weight:500; color:#515553; margin-left:4px; }
.fe-box { border-radius:10px; padding:9px 11px; font-size:12.5px; line-height:1.45; margin-top:6px; display:flex; gap:8px; }
.fe-box svg { width:16px; height:16px; flex:none; margin-top:1px; }
.fe-box.amber { background:#fff7e8; color:#8a5a00; border:1px solid #fbe3b4; } .fe-box.blue { background:#ecfdf5; color:#047857; border:1px solid #c6f0dc; }
.fe-box.green { background:#eaf8f2; color:#065f46; border:1px solid #c9ecdd; } .fe-box.red { background:#fdeeee; color:#9b1c1c; border:1px solid #f6caca; }
.fe-list { margin:4px 0 0; padding-left:18px; } .fe-list li { font-size:12.8px; color:#444846; line-height:1.5; margin-bottom:4px; }
.fe-kv { display:grid; grid-template-columns:170px 1fr; gap:6px 14px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.8px; padding:6px 2px; }
.fe-kv dt { color:#6f7572; font-weight:600; } .fe-kv dd { margin:0; color:#313432; line-height:1.5; }
.st-key-fe_foot { padding-top:10px; }
.st-key-fe_pager button { min-width:34px; }
/* matrix preview */
.st-key-ds_panel_matrix { animation-delay:.15s; }
.fe-mx-wrap { overflow-x:auto; border:1px solid #eceeed; border-radius:12px; margin-top:8px; }
.fe-mx { border-collapse:collapse; width:100%; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.8px; }
.fe-mx th { background:#f0faf7; color:#515553; font-weight:650; text-align:right; padding:9px 12px; white-space:nowrap; border-bottom:1px solid #eceeed; position:sticky; top:0; }
.fe-mx th:first-child, .fe-mx td:first-child { text-align:left; position:sticky; left:0; background:inherit; }
.fe-mx td { padding:8px 12px; text-align:right; color:#313432; border-bottom:1px solid #f1f2f2; white-space:nowrap; font-variant-numeric:tabular-nums; background:#fff; }
.fe-mx th.tgt, .fe-mx td.tgt { background:#f0fdf6; color:#1f2937; font-weight:650; }
.fe-mx tr:hover td { background:#effdf8; }
.fe-meta { display:flex; flex-wrap:wrap; gap:6px 16px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.5px; color:#6f7572; margin-top:2px; }
.fe-meta b { color:#313432; font-weight:650; }
.fe-rej { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.fe-rej div { background:#f3fbf8; border:1px solid #e8f2ee; border-radius:12px; padding:10px 12px; }
.fe-rej b { font-size:13px; color:#101614; } .fe-rej p { font-size:12.4px; color:#515553; margin:4px 0 0; line-height:1.45; }
@media (max-width: 1200px) { .fe-dg { grid-template-columns:1fr 1fr; } }
.st-key-fe_mx_actions button p { white-space:nowrap; }
@media (max-width: 1100px) {
  /* narrow screens: keep select · feature · availability · missing · details; the rest is in the details panel */
  [class*="st-key-fe_row_"] [data-testid="stHorizontalBlock"], .st-key-fe_head [data-testid="stHorizontalBlock"] { flex-wrap:nowrap; }
  [class*="st-key-fe_row_"] [data-testid="stColumn"]:is(:nth-child(3),:nth-child(4),:nth-child(7),:nth-child(8)),
  .st-key-fe_head [data-testid="stColumn"]:is(:nth-child(3),:nth-child(4),:nth-child(7),:nth-child(8)) { display:none; }
  [class*="st-key-fe_row_"] [data-testid="stColumn"], .st-key-fe_head [data-testid="stColumn"] { min-width:0; }
  .fe-rej { grid-template-columns:1fr; }
  .st-key-fe_cards [data-testid="stHorizontalBlock"], .st-key-fe_toolbar [data-testid="stHorizontalBlock"],
  .st-key-fe_foot [data-testid="stHorizontalBlock"], .st-key-ds_panel_matrix [data-testid="stHorizontalBlock"] { flex-wrap:wrap; row-gap:10px; }
  .st-key-fe_cards [data-testid="stColumn"], .st-key-fe_toolbar [data-testid="stColumn"],
  .st-key-fe_foot [data-testid="stColumn"], .st-key-ds_panel_matrix [data-testid="stColumn"] { flex:1 1 100% !important; width:100% !important; min-width:100% !important; }
  .st-key-fe_pager, .st-key-fe_mx_actions { justify-content:flex-start !important; }
}
@media (max-width: 640px) { .fe-dg { grid-template-columns:1fr; } .fe-kv { grid-template-columns:1fr; } }
</style>
"""


# ------------------------------------------------------------------ state

def _key(scope: str, mode: str) -> str:
    return f"{scope}|{mode}"


def selected_features(scope: str, mode: str) -> list[str]:
    """The APPLIED selection (what the ML Training Studio uses); defaults until something is applied."""
    sel = st.session_state.setdefault("feature_selection", {})
    allowed = set(available_features(scope, mode))
    return [f for f in sel.get(_key(scope, mode), default_features(scope, mode)) if f in allowed]


def _draft(scope: str, mode: str) -> list[str]:
    d = st.session_state.setdefault("fe_draft", {})
    if _key(scope, mode) not in d:
        d[_key(scope, mode)] = selected_features(scope, mode)
    return d[_key(scope, mode)]


def _set_draft(scope: str, mode: str, feats: list[str]) -> None:
    order = list(CATALOG_BY_NAME)
    st.session_state.setdefault("fe_draft", {})[_key(scope, mode)] = sorted(set(feats), key=order.index)
    prefix = f"fe_cb_{scope}_{mode}_"
    for k in [k for k in st.session_state if isinstance(k, str) and (k.startswith(prefix) or k == f"fe_all_{scope}_{mode}")]:
        del st.session_state[k]


def _toggle(scope: str, mode: str, name: str, key: str) -> None:
    cur = set(_draft(scope, mode))
    (cur.add if st.session_state[key] else cur.discard)(name)
    _set_draft(scope, mode, list(cur))


def _toggle_all(scope: str, mode: str, names: list[str], key: str) -> None:
    cur = set(_draft(scope, mode))
    if st.session_state[key]:
        cur |= set(names)
    else:
        cur -= set(names)
    _set_draft(scope, mode, list(cur))


def _apply(scope: str, mode: str) -> None:
    feats = _draft(scope, mode)
    if not feats:
        st.session_state["fe_flash"] = ("bad", "Select at least one feature before applying.")
        return
    st.session_state.setdefault("feature_selection", {})[_key(scope, mode)] = list(feats)
    for k in [k for k in st.session_state if isinstance(k, str) and k.startswith(f"tr_feats_{scope}_{mode}")]:
        del st.session_state[k]  # the Training Studio picks up the new defaults
    st.session_state["fe_flash"] = ("ok", f"Applied {len(feats)} features — the ML Training Studio now uses them for "
                                          f"{config.SCOPES[scope].split(' (')[0]} · {'What-if' if mode == 'whatif' else 'Competition'}.")


def _reset(scope: str, mode: str) -> None:
    _set_draft(scope, mode, default_features(scope, mode))
    st.session_state["fe_flash"] = ("ok", "Restored the recommended defaults for this scope and mode (not yet applied).")


def _first_page() -> None:
    st.session_state["fe_page"] = 1


def _sort(col: str) -> None:
    c, d = st.session_state.get("fe_sort", (None, None))
    st.session_state["fe_sort"] = (col, "asc") if c != col else ((col, "desc") if d == "asc" else (None, None))
    _first_page()


def _open(name: str) -> None:
    o = st.session_state.setdefault("fe_open", set())
    (o.discard if name in o else o.add)(name)


def _goto(p: int) -> None:
    st.session_state["fe_page"] = p


# ------------------------------------------------------------------ cached facts

@st.cache_data(show_spinner=False)
def _moved() -> dict:
    return scenario_sensitivity(analytic())


@st.cache_data(show_spinner="Measuring features on the training rows …")
def _rows(scope: str, mode: str) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in feature_rows(analytic(), scope, mode, _moved())])


@st.cache_data(show_spinner=False)
def _matrix(scope: str, mode: str) -> pd.DataFrame:
    return training_matrix(analytic(), scope, mode)


@st.cache_data(show_spinner=False)
def _empty_sdg(scope: str, mode: str):
    return empty_same_day_cells(analytic(), scope, mode)


# ------------------------------------------------------------------ cells

def _chip(text: str, tone: str, icon: str | None = None) -> str:
    return f'<span class="fe-chip {tone}">{ICON[icon] if icon else ""}{escape(text)}</span>'


def _cell(col, markup: str) -> None:
    col.markdown(markup, unsafe_allow_html=True)


def _fmt_pct(p: float) -> str:
    return "—" if not np.isfinite(p) else ("0.00%" if p == 0 else f"{p:.2f}%" if p >= 0.01 else "<0.01%")


def _avail_chip(r) -> str:
    """When the value is known relative to the predicted date; features outside this mode say so instead."""
    if not r.available:
        return _chip("Competition only", "grey", "lock")
    from ..leakage import timing_class
    word, tone = KNOWN[timing_class(r.name)]
    return _chip(word, tone)


# ------------------------------------------------------------------ details

def _details(r, scope: str, mode: str, tr: pd.DataFrame) -> None:
    fs = CATALOG_BY_NAME[r.name]
    with st.container(key=f"fe_det_{r.name}"):
        tabs = st.tabs(["Definition", "Source & Calculation", "Missing Values", "Scenario Behaviour", "Assumptions & Notes"])
        src = "".join(_chip(s.strip(), "blue") for s in fs.source.split(","))
        n = r.missing_n
        miss_line = (f'<div class="fe-big">{_fmt_pct(r.missing_pct)}<small>({n:,} of {r.rows:,} training rows)</small></div>'
                     if n >= 0 else '<div class="fe-big">—</div>')
        handling = missing_treatment(r.name)
        scen_icon = "check" if r.scenario == "Yes" else "info"
        with tabs[0]:
            design.html(
                f'<div class="fe-dg"><div><h5>Description</h5><p>{escape(fs.definition)}</p><div class="fe-formula">{_formula_html(r.name)}</div>'
                f'<p style="margin-top:8px"><b>Unit:</b> {escape(_UNIT.get(r.name, _unit_of(r.name)))}</p></div>'
                f'<div><h5>Source columns</h5><div class="fe-src">{src}</div><p>{escape(_dataset(r.name))}</p></div>'
                f'<div><h5>Missing values (train)</h5>{miss_line}<div class="fe-box {"amber" if n > 0 else "green"}">{ICON["info"] if n > 0 else ICON["check"]}<span>{escape(_short_handling(r.name, n))}</span></div></div>'
                f'<div><h5>Scenario behaviour</h5><p>{_chip(r.scenario, SCEN_TONE[r.scenario], scen_icon)}</p>'
                f'<div class="fe-box blue">{ICON["info"]}<span>{escape(r.scenario_why)}</span></div></div></div>')
        with tabs[1]:
            design.html(
                f'<dl class="fe-kv"><dt>Dataset</dt><dd>{escape(_dataset(r.name))}</dd>'
                f'<dt>Source columns</dt><dd>{escape(fs.source)}</dd>'
                f'<dt>Aggregation</dt><dd>{escape(_aggregation(r.name, scope))}</dd>'
                f'<dt>Time window</dt><dd>{escape(_time_window(r.name))}</dd>'
                f'<dt>Formula</dt><dd>{escape(_FORMULA.get(r.name, ""))}</dd>'
                f'<dt>Invalid / missing inputs</dt><dd>{escape(missing_reason(r.name, scope, mode))}</dd>'
                f'<dt>Built by</dt><dd>The same feature functions are used for training, evaluation, competition prediction and the What-if Simulator.</dd></dl>')
        with tabs[2]:
            col = tr[r.name] if r.name in tr else None
            zeros = zero_days(tr, r.name)
            extra = ""
            if r.name == "same_day_guests":
                e = _empty_sdg(scope, mode)
                if e is not None:
                    extra = (f'<dt>Suppressed (“*”) read as 0</dt><dd>{e:,} training rows include an empty Same-Day Guests cell that is read as 0, '
                             'so the value is a lower bound. Empty cells cannot be told apart from genuine zeros.</dd>')
            dist = ""
            if col is not None and fs.kind == "numeric" and col.notna().any():
                q = col.quantile([0, .5, 1])
                dist = f'<dt>Training values</dt><dd>min {_num(q.iloc[0])} · median {_num(q.iloc[1])} · max {_num(q.iloc[2])}</dd>'
            design.html(
                f'<dl class="fe-kv"><dt>Missing in training rows</dt><dd>{(f"{n:,} of {r.rows:,} ({_fmt_pct(r.missing_pct)})") if n >= 0 else "—"}</dd>'
                f'<dt>Why values are missing</dt><dd>{escape(missing_reason(r.name, scope, mode))}</dd>'
                f'<dt>Genuine zeros</dt><dd>{zeros:,} training rows are exactly 0{" — days without an operated flight, which are real zeros, not gaps" if zeros and _is_aviation(r.name) else ""}.</dd>'
                f'{extra}{dist}<dt>Preprocessing</dt><dd>{escape(handling)}</dd></dl>')
        with tabs[3]:
            design.html(
                f'<dl class="fe-kv"><dt>Changes in scenarios</dt><dd>{_chip(r.scenario, SCEN_TONE[r.scenario])}</dd>'
                f'<dt>How it is recomputed</dt><dd>{escape(r.scenario_why)}</dd>'
                f'<dt>Scenario inputs that move it</dt><dd>{escape(_scenario_inputs(r.name, _moved()))}</dd>'
                f'<dt>Monotone constraint</dt><dd>{escape(_mono_text(r))}</dd></dl>')
        with tabs[4]:
            lv_tone = LEAK_TONE[r.leakage]
            items = "".join(f"<li>{escape(x)}</li>" for x in assumptions(r.name, scope)) or "<li>No additional assumptions.</li>"
            avail = ("Available in this mode." if r.available else r.reason)
            design.html(
                f'<dl class="fe-kv"><dt>Availability</dt><dd>{escape(avail)} Modes: {escape(", ".join("What-if" if m == "whatif" else "Competition" for m in fs.modes))}.</dd>'
                f'<dt>Leakage risk</dt><dd>{_chip(r.leakage, lv_tone)} {escape(r.leakage_why)}</dd></dl>'
                f'<div class="fe-dg" style="grid-template-columns:1fr"><div><h5>Assumptions & limitations</h5><ul class="fe-list">{items}</ul></div></div>')


def _formula_html(name: str) -> str:
    """Render 'lhs = num ÷ den' as a stacked fraction; anything else as plain text."""
    f = _FORMULA.get(name, "")
    if f.count(" ÷ ") == 1 and " = " in f:
        lhs, rhs = f.split(" = ", 1)
        num, den = rhs.split(" ÷ ")
        tail = ""
        if den.endswith(" − 1"):
            den, tail = den[:-4], " − 1"
        return f'{escape(lhs)} = <span class="fe-frac"><span>{escape(num)}</span><span>{escape(den)}</span></span>{escape(tail)}'
    return escape(f)


def _num(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.3f}" if abs(v) < 10 else f"{v:,.1f}"


def _is_aviation(name: str) -> bool:
    return CATALOG_BY_NAME[name].group in ("aviation", "derived") or name.endswith("_prev7d")


def _unit_of(name: str) -> str:
    from ..feature_facts import _QTY, _base
    q = _base(name)
    return _QTY[q][0] if q else ""


def _dataset(name: str) -> str:
    g = CATALOG_BY_NAME[name].group
    if g == "time":
        return "Derived from the Date column of the hotel files."
    if g == "market":
        return "Nationality (international hotel files) matched by exact name to Departure Country Name (flight_data.xlsx)."
    if name.startswith("guests_lag364"):
        return "Guests from the training files (data_international_train / data_domestic_train), 2022-01-01 → 2025-07-31."
    if g == "competition":
        return "Hotel files, train and official test (both contain New Arrivals and Same-Day Guests)."
    return "flight_data.xlsx — daily route-day rows from 2023-01-01 (the 2022 monthly rows are not used)."


def _aggregation(name: str, scope: str) -> str:
    g = CATALOG_BY_NAME[name].group
    if g in ("time", "market"):
        return "None — one value per row."
    if name.startswith("guests_lag364"):
        return ("Per nationality (market scope) on a complete calendar; missing nationality-days stay missing."
                if scope == "international_market" else "Daily hotel total, shifted by 364 days.")
    if g == "competition":
        return ("Per nationality-day, as recorded." if scope == "international_market"
                else "Summed over the nationality rows of each day that have New Arrivals ≥ 10, matching the official test extract.")
    who = "per departure country (the market)" if scope == "international_market" else "over every arriving route (all departure countries)"
    ratio = " Ratios are recomputed from the summed numerator and denominator, never averaged." if name in _FORMULA and "÷" in _FORMULA[name] else ""
    return f"Route-day rows summed {who} on a complete daily calendar; a day without a row counts as zero flights.{ratio}"


def _time_window(name: str) -> str:
    g = CATALOG_BY_NAME[name].group
    if g in ("time", "market"):
        return "None."
    if name == "guests_lag364":
        return "364 days earlier (same weekday)."
    if name == "guests_lag364_7d":
        return "Mean of t−370 … t−364 (needs at least 4 observed days)."
    if name == "new_arrivals_7d":
        return "Trailing 7 days (t−6 … t); train and test are concatenated so the window is complete on the first test days."
    return _window(name)


def _short_handling(name: str, n: int) -> str:
    if CATALOG_BY_NAME[name].kind == "categorical":
        return "Never missing; encoded inside the model pipeline."
    if name == "same_day_guests":
        return "Empty cells read as 0 (lower bound); remaining gaps imputed inside the pipeline."
    if n == 0:
        return "No missing values in these training rows. Any gap at prediction time is imputed with the training median."
    return "Imputed inside the pipeline with the training-fold median plus a missing-indicator; CatBoost handles NaN natively."


def _scenario_inputs(name: str, moved: dict) -> str:
    from ..feature_facts import SCENARIO_INPUTS
    keys = moved.get(name, [])
    return ", ".join(SCENARIO_INPUTS[k] for k in keys) + "." if keys else "None."


def _mono_text(r) -> str:
    if r.monotone:
        return ("Non-decreasing for tree models (Random Forest, Extra Trees, XGBoost, CatBoost) when the monotone option is on: holding "
                "every other input fixed, a higher value cannot lower predicted Guests. Not applied to Ridge, SVR or the baseline.")
    return "Unconstrained — ratios, growth rates, calendar terms and guest history may relate to Guests in either direction."


# ------------------------------------------------------------------ page

def render() -> None:
    design.inject()
    st.markdown(heroes.CSS + CSS, unsafe_allow_html=True)
    design.html(heroes.banner("features", "Section 3", "Feature <span>Engineering</span>", "Which predictors the models use, how they are built and when they are known."))
    st.write("")

    scopes, modes = list(config.SCOPES), list(config.MODES)
    with st.container(key="fe_cards"):
        c1, c2, c3 = st.columns([1.6, 1.4, 1.72], gap="small")
        with c1, st.container(key="fe_card_scope"):
            design.html(f'<div class="fe-ch"><div class="ds-ico blue">{ICON["globe"]}</div><b>Prediction scope</b></div>')
            scope = st.selectbox("Prediction scope", scopes, format_func=SCOPE_SHORT.get,
                                 key="fe_scope", label_visibility="collapsed", on_change=_first_page, persist_state="session")
            design.html(f'<div class="fe-cap">{escape(SCOPE_HINT[scope])}</div>')
        with c2, st.container(key="fe_card_mode"):
            design.html(f'<div class="fe-ch"><div class="ds-ico violet">{ICON["split"]}</div><b>Prediction mode</b></div>')
            mode = st.radio("Prediction mode", modes, key="fe_mode", label_visibility="collapsed", on_change=_first_page, persist_state="session",
                            format_func=lambda m: ("What-if :gray[— aviation, calendar, last year's guests]" if m == "whatif"
                                                   else "Competition :gray[— also same-day hotel fields]"),
                            help="Competition mode may use New Arrivals and Same-Day Guests, which the official test files contain. "
                                 "Such models are nowcasts: they cannot forecast or drive the What-if Simulator.")

    rows = _rows(scope, mode)
    tr = _matrix(scope, mode)
    avail = rows[rows["available"]]["name"].tolist()
    draft = [f for f in _draft(scope, mode) if f in avail]
    applied = selected_features(scope, mode)
    pending = set(draft) != set(applied)

    with c3, st.container(key="fe_card_sel"):
        a, b = st.columns([1.0, 1.2], vertical_alignment="center")
        with a:
            design.html(f'<div class="fe-ch"><div class="ds-ico teal">{ICON["layers"]}</div><b>Selected features</b></div>'
                        f'<div class="fe-count"><b>{len(draft)}</b><span>/ {len(avail)}</span></div><div class="fe-cap">usable in {"What-if" if mode == "whatif" else "Competition"} mode</div>'
                        + ('<span class="fe-state bad">' + ICON["warning"] + 'Select at least one</span>' if not draft else
                           f'<span class="fe-state warn">{ICON["info"]}Not applied yet</span>' if pending else
                           f'<span class="fe-state ok">{ICON["check"]}Ready to use</span>'))
        with b:
            st.button("Apply selection", type="primary", icon=":material/arrow_forward:", icon_position="right", key="fe_apply",
                      on_click=_apply, args=(scope, mode), width="stretch", disabled=not draft)
            st.button("Reset to defaults", icon=":material/settings_backup_restore:", key="fe_reset", on_click=_reset, args=(scope, mode), width="stretch")

    flash = st.session_state.pop("fe_flash", None)
    if flash:
        (st.toast if flash[0] == "ok" else st.warning)(flash[1], icon=":material/check_circle:" if flash[0] == "ok" else ":material/warning:")

    # ---------------------------------------------------------------- catalogue
    st.write("")
    with st.container(key="fe_catalog"):
        counts = rows["group"].value_counts()
        groups = ["all"] + [g for g in GROUP_ORDER if g in counts]
        with st.container(key="fe_toolbar"):
            t1, t2, t3, t4 = st.columns([3.7, 1.1, 0.95, 0.8], vertical_alignment="center")
            with t1, st.container(key="fe_group"):
                grp = st.radio("Feature group", groups, horizontal=True, label_visibility="collapsed", key="fe_group_pick", on_change=_first_page,
                               format_func=lambda g: f"All ({len(rows)})" if g == "all" else f"{TAB_LABEL[g]} ({counts[g]})")
            with t2, st.container(key="fe_q"):
                q = st.text_input("Search", placeholder="Search features", label_visibility="collapsed",
                                  icon=":material/search:", key="fe_search", on_change=_first_page)
            only = t3.toggle("Selected only", key="fe_only", on_change=_first_page)
            if t4.button("Full table", icon=":material/open_in_full:", key="fe_cat_full", help="Every feature with all its attributes", width="stretch"):
                _catalogue_dialog(rows, draft, scope, mode)

        view = rows.copy()
        view["selected"] = view["name"].isin(draft)
        if grp != "all":
            view = view[view["group"] == grp]
        if q:
            ql = q.lower()
            hay = view["name"].str.lower() + " " + view["label"].str.lower() + " " + view["name"].map(
                lambda n: (CATALOG_BY_NAME[n].source + " " + CATALOG_BY_NAME[n].definition).lower())
            view = view[hay.str.contains(ql, regex=False)]
        if only:
            view = view[view["selected"]]
        col, direction = st.session_state.get("fe_sort", (None, None))
        view["_g"] = view["group"].map(GROUP_ORDER.index)
        view["_o"] = view["name"].map(list(CATALOG_BY_NAME).index)
        view["_l"] = view["leakage"].map({"None": 0, "Low": 1, "High": 2, "Not allowed": 3})
        if col:
            key = {"feature": "name", "group": "_g", "missing": "missing_pct", "leakage": "_l"}[col]
            view = view.sort_values([key, "_o"], ascending=[direction == "asc", True])
        else:
            view = view.sort_values(["_g", "_o"])

        size = st.session_state.get("fe_size", 10)
        n_pages = max(1, math.ceil(len(view) / size)) if size else 1
        page = min(st.session_state.get("fe_page", 1), n_pages)
        page_rows = view.iloc[(page - 1) * size: page * size] if size else view

        # header
        with st.container(key="fe_head"):
            h = st.columns(SPEC, vertical_alignment="center")
            vis_avail = [n for n in page_rows["name"] if n in avail]
            all_on = bool(vis_avail) and all(n in draft for n in vis_avail)
            k_all = f"fe_all_{scope}_{mode}"
            h[0].checkbox("Select all visible", value=all_on, key=k_all, label_visibility="collapsed", disabled=not vis_avail,
                          on_change=_toggle_all, args=(scope, mode, vis_avail, k_all), help="Select or clear every available feature on this page")
            arrow = {"asc": ":material/arrow_upward:", "desc": ":material/arrow_downward:"}

            def sort_btn(c, label, name):
                ic = arrow[direction] if col == name else ":material/unfold_more:"
                c.button(label, key=f"fe_sort_{name}", type="tertiary", icon=ic, icon_position="right", on_click=_sort, args=(name,))
            sort_btn(h[1], "Feature", "feature")
            _cell(h[2], '<span class="fe-hd">Label</span>')
            sort_btn(h[3], "Group", "group")
            h[4].markdown('<span class="fe-hd">Known</span>', unsafe_allow_html=True,
                          help="When the value is known relative to the date being predicted. “On the day” values are realised flight figures: "
                               "fine for scenarios, where you set them, but an advance forecast would need airline schedules.")
            sort_btn(h[5], "Missing (train)", "missing")
            sort_btn(h[6], "Leakage risk", "leakage")
            _cell(h[7], '<span class="fe-hd" style="white-space:normal;line-height:1.25">Moves in scenarios</span>')
            _cell(h[8], '<span class="fe-hd"></span>')

        opened = st.session_state.setdefault("fe_open", set())
        if page_rows.empty:
            design.html('<div class="fe-box blue" style="margin:10px 0">' + ICON["search"] + '<span>No feature matches these filters.</span></div>')
        for r in page_rows.itertuples():
            with st.container(key=f"fe_row_{r.name}"):
                c = st.columns(SPEC, vertical_alignment="center")
                k = f"fe_cb_{scope}_{mode}_{r.name}"
                c[0].checkbox(f"Use {r.name}", value=r.name in draft, key=k, label_visibility="collapsed", disabled=not r.available,
                              on_change=_toggle, args=(scope, mode, r.name, k),
                              help=None if r.available else r.reason)
                _cell(c[1], f'<span class="fe-name">{escape(r.name).replace("_", "_<wbr>")}</span>')
                _cell(c[2], f'<span class="fe-lab">{escape(r.label)}</span>')
                _cell(c[3], _chip(GROUP_LABEL[r.group].split(" (")[0], GROUP_TONE[r.group]))
                _cell(c[4], _avail_chip(r))
                pct = r.missing_pct
                _cell(c[5], f'<span class="fe-num{" hi" if np.isfinite(pct) and pct >= 1 else ""}">{_fmt_pct(pct)}</span>')
                _cell(c[6], _chip(r.leakage, LEAK_TONE[r.leakage]))
                _cell(c[7], _chip(r.scenario, SCEN_TONE[r.scenario]))
                c[8].button("", key=f"fe_x_{r.name}", type="tertiary", on_click=_open, args=(r.name,),
                            icon=":material/expand_more:" if r.name in opened else ":material/chevron_right:",
                            help=f"{'Hide' if r.name in opened else 'Show'} details for {r.name}")
            if r.name in opened:
                _details(r, scope, mode, tr)

        with st.container(key="fe_foot"):
            f1, f2, f3 = st.columns([0.9, 2.5, 1.9], vertical_alignment="center")
            f1.selectbox("Rows per page", PAGE_SIZES, index=PAGE_SIZES.index(size), key="fe_size", on_change=_first_page,
                         format_func=lambda v: f"Show {v}" if v else "Show all", label_visibility="collapsed")
            shown = f"{(page - 1) * size + 1 if len(view) else 0}–{min(page * size, len(view)) if size else len(view)}"
            _cell(f2, f'<span class="fe-cap">{shown} of {len(view)} features · '
                      f'{len(draft)} selected · '
                      + " · ".join(f"{GROUP_LABEL[g].split(' (')[0]} {n}" for g, n in pd.Series([CATALOG_BY_NAME[x].group for x in draft]).value_counts().items())
                      + "</span>")
            with f3, st.container(key="fe_pager", horizontal=True, horizontal_alignment="right", gap="small"):
                st.button("", icon=":material/chevron_left:", key="fe_prev", disabled=page <= 1, on_click=_goto, args=(page - 1,), help="Previous page")
                for p in range(1, n_pages + 1):
                    st.button(str(p), key=f"fe_p{p}", type="primary" if p == page else "secondary", on_click=_goto, args=(p,))
                st.button("", icon=":material/chevron_right:", key="fe_next", disabled=page >= n_pages, on_click=_goto, args=(page + 1,), help="Next page")

    # ---------------------------------------------------------------- matrix preview
    st.write("")
    feats = draft
    cols = matrix_columns(scope, feats)
    mx = tr[cols].rename(columns={"y": "Guests (target)", "market_key": "Market"})
    fp = matrix_fingerprint(tr[cols])
    with design.panel("matrix", "Preview of the model matrix",
                      f"{len(tr):,} training rows · {tr['Date'].min():%Y-%m-%d} → {tr['Date'].max():%Y-%m-%d} · {len(feats)} selected features · "
                      "training windows only"):
        m1, m2 = st.columns([2.2, 1.8], vertical_alignment="center")
        with m1:
            design.html(f'<div class="fe-meta"><span>Target <b>Guests</b> ({"per nationality-day" if scope == "international_market" else "per day"})</span>'
                        f'<span>Unique dates <b>{tr["Date"].nunique():,}</b></span>'
                        f'<span>Holdout from <b>{config.DEFAULT_HOLDOUT_START:%Y-%m-%d}</b> by default</span></div>')
        with m2, st.container(horizontal=True, horizontal_alignment="right", gap="small", key="fe_mx_actions"):
            if st.button("Expand full table", icon=":material/open_in_full:", key="fe_expand"):
                _full_table(mx, scope, mode, fp)
            st.download_button("Download CSV", mx.to_csv(index=False).encode(), file_name=f"model_matrix_{scope}_{mode}.csv", mime="text/csv",
                               icon=":material/download:", key="fe_dl", on_click="ignore")
        design.html(_matrix_html(mx.head(8)))
        design.html('<div class="ds-note" style="margin-top:8px">Built by the same function the ML Training Studio uses (model frame → training rows → '
                    f'selected columns), so a model trained with this selection fits exactly this matrix. Values are rounded for display; the CSV keeps full precision. '
                    f'Matrix fingerprint {fp}.</div>')

    # ---------------------------------------------------------------- not offered
    st.write("")
    with st.expander("Not offered — and why", icon=":material/block:"):
        design.html('<div class="fe-rej">' + "".join(f'<div><b>{escape(x["name"])}</b><p>{escape(x["reason"])}</p></div>' for x in REJECTED_FEATURES) + "</div>")
    next_step("features")



def _matrix_html(df: pd.DataFrame) -> str:
    head = "".join(f'<th class="{"tgt" if c == "Guests (target)" else ""}">{escape(c)}</th>' for c in df.columns) + "<th>⋯</th>"
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if c == "Date":
                s = f"{v:%Y-%m-%d}"
            elif c == "Market":
                s = str(v).title()
            elif isinstance(v, (int, np.integer)):
                s = f"{v:,}"
            elif isinstance(v, (float, np.floating)):
                s = "—" if not np.isfinite(v) else (f"{v:,.0f}" if abs(v) >= 1000 or (float(v).is_integer() and abs(v) >= 10) else f"{v:.4f}")
            else:
                s = str(v)
            cells.append(f'<td class="{"tgt" if c == "Guests (target)" else ""}">{escape(s)}</td>')
        body.append("<tr>" + "".join(cells) + "<td>⋯</td></tr>")
    return f'<div class="fe-mx-wrap"><table class="fe-mx"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _catalogue_frame(rows: pd.DataFrame, draft: list[str]) -> pd.DataFrame:
    df = rows.copy()
    df.insert(0, "selected", df["name"].isin(draft))
    df["group"] = df["group"].map(lambda g: GROUP_LABEL[g])
    df["definition"] = df["name"].map(lambda n: CATALOG_BY_NAME[n].definition)
    df["source_columns"] = df["name"].map(lambda n: CATALOG_BY_NAME[n].source)
    df["formula"] = df["name"].map(lambda n: _FORMULA.get(n, ""))
    df["monotone"] = df["monotone"].map({True: "+ (non-decreasing, trees)", False: "—"})
    df = df.rename(columns={"name": "feature", "missing_n": "missing_rows", "missing_pct": "missing_pct", "rows": "training_rows",
                            "leakage_why": "leakage_justification", "scenario": "changes_in_scenarios", "scenario_why": "scenario_explanation",
                            "reason": "unavailable_reason"})
    return df[["selected", "feature", "label", "group", "available", "unavailable_reason", "missing_rows", "training_rows", "missing_pct",
               "leakage", "leakage_justification", "changes_in_scenarios", "scenario_explanation", "monotone", "default",
               "definition", "formula", "source_columns"]]


@st.dialog("Feature catalogue — full table", width="large")
def _catalogue_dialog(rows: pd.DataFrame, draft: list[str], scope: str, mode: str) -> None:
    df = _catalogue_frame(rows, draft)
    st.caption(f"{len(df)} features · {config.SCOPES[scope]} · {config.MODES[mode]} · missing values counted on the {int(rows['rows'].iloc[0]):,} training rows")
    st.dataframe(df, hide_index=True, width="stretch", height=560, column_config={
        "selected": st.column_config.CheckboxColumn("Selected"), "available": st.column_config.CheckboxColumn("Available"),
        "default": st.column_config.CheckboxColumn("Default"), "missing_pct": st.column_config.NumberColumn("Missing %", format="%.2f"),
        "leakage_justification": st.column_config.TextColumn(width="large"), "scenario_explanation": st.column_config.TextColumn(width="large"),
        "definition": st.column_config.TextColumn(width="large")})
    st.download_button("Download catalogue (CSV)", df.to_csv(index=False).encode(), file_name=f"feature_catalogue_{scope}_{mode}.csv",
                       mime="text/csv", icon=":material/download:", key="fe_cat_dl", on_click="ignore")


@st.dialog("Model matrix — full table", width="large")
def _full_table(mx: pd.DataFrame, scope: str, mode: str, fp: str) -> None:
    st.caption(f"{len(mx):,} rows × {mx.shape[1]} columns · {config.SCOPES[scope]} · {config.MODES[mode]} · fingerprint {fp}")
    fmt = {c: st.column_config.NumberColumn(format="%.4f") for c in mx.columns if mx[c].dtype.kind == "f" and mx[c].abs().max() < 10}
    st.dataframe(mx, hide_index=True, width="stretch", height=560, column_config={"Date": st.column_config.DateColumn(format="YYYY-MM-DD"), **fmt})
