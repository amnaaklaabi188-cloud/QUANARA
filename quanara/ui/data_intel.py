"""Section 2 — Data Intelligence (same design system as the Welcome & Overview page).

Layout: banner · five aligned dataset cards · tabs (coverage timeline, passenger composition,
data quality, dataset explorer, distributions, audit & dictionary). Every number and chart is
computed from the five workbooks (cleaned, read-only) or from reports generated from them.
"""
from __future__ import annotations

import json
from html import escape as esc

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import config
from ..dictionary import field_profiles, field_table
from ..data_loading import read_raw
from .. import registry
from .. import leakage
from .. import pdf_dictionary
from ..intel_facts import (STATUS, audit_findings, checksum_status, dataset_cards, dictionary_reconciliation, missing_profile,
                           passenger_composition, quality_stats, technical_checks, value_kinds)
from . import design as ds
from . import heroes
from .common import analytic, audit, cleaned, md, next_step
from .design import ICON, SERIES

_LABEL = {"intl_train": "International · train", "intl_test": "International · test", "dom_train": "Domestic · train",
          "dom_test": "Domestic · test", "flight": "Flight operations"}
_HELPER_COLS = {"same_day_suppressed", "grain", "flights", "flights_decode_error", "market_key"}
PLOT_CFG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}

CSS = """
<style>
/* quality overview cards */
.dq-top { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; margin-bottom:16px; }
.dq-btn { display:inline-flex; gap:8px; align-items:center; white-space:nowrap; padding:8px 14px; border-radius:10px; border:1px solid #97f3d2; background:#e8fcf5;
          color:#059669 !important; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:13.5px; font-weight:650; text-decoration:none !important; }
.dq-btn:hover { background:#d0faea; } .dq-btn svg { width:16px; height:16px; }
.dq-cards { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:14px; }
.dq-card { display:grid; grid-row:span 4; grid-template-rows:subgrid; row-gap:0; background:#ffffff; border:1px solid #e7e8e7; border-top:3px solid; border-radius:14px;
           padding:16px 18px 16px; min-width:0; transition:transform .2s ease, box-shadow .2s ease; }
.dq-card:hover { transform:translateY(-2px); box-shadow:0 12px 26px rgba(15,23,20,.08); }
@supports not (grid-template-rows: subgrid) { .dq-card { display:flex; flex-direction:column; } .dq-foot { margin-top:auto; } }
.dq-card.green { border-top-color:#047857; } .dq-card.blue { border-top-color:#059669; } .dq-card.orange { border-top-color:#1f2937; } .dq-card.violet { border-top-color:#047857; }
.dq-head { display:flex; gap:12px; align-items:center; }
.dq-ico { width:44px; height:44px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex:none; }
.dq-ico svg { width:22px; height:22px; }
.dq-ico.green { background:#e3f6f3; color:#0f8f83; } .dq-ico.blue { background:#ecfdf5; color:#059669; } .dq-ico.orange { background:#fdeee6; color:#d65a20; } .dq-ico.violet { background:#ecfdf5; color:#065f46; }
.dq-label { font-size:12.8px; font-weight:700; } .dq-label.green { color:#0f8f83; } .dq-label.blue { color:#059669; } .dq-label.orange { color:#c4541c; } .dq-label.violet { color:#065f46; }
.dq-title { margin-top:12px; font-size:17.5px; font-weight:780; color:#101614; line-height:1.3; letter-spacing:-.01em; }
.dq-desc { margin-top:8px; font-size:13.4px; color:#515553; line-height:1.55; }
.dq-foot { margin-top:14px; align-self:end; }
.dq-chip { display:inline-flex; align-items:center; gap:6px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.8px; font-weight:700;
           border-radius:999px; padding:5px 11px; white-space:nowrap; }
.dq-chip svg { width:14px; height:14px; }
.dq-chip.good { color:#047857; background:#d1fae5; } .dq-chip.info { color:#047857; background:#c6f9e6; } .dq-chip.warn { color:#8a5a00; background:#fff2d6; }
.dq-chip.violet { color:#065f46; background:#ecfdf5; }
.dq-chiplink { text-decoration:none !important; } .dq-chiplink:hover .dq-chip { filter:brightness(.97); }
.dq-points { display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:14px 28px; margin-top:6px; }
.dq-point { display:flex; gap:12px; align-items:flex-start; } .dq-point b { font-size:14.5px; color:#101614; }
.dq-point p { margin:3px 0 0; font-size:13.5px; color:#515553; line-height:1.55; }
/* audit findings — full-width accordion */
[class*="st-key-di_find_"]:not(.st-key-di_findings_head):not(.st-key-di_find_expand):not(.st-key-di_find_collapse) details {
    background:#ffffff; border:1px solid #e7e8e7 !important; border-radius:14px !important; box-shadow:0 1px 2px rgba(15,23,20,.04), 0 6px 16px rgba(15,23,20,.04);
    transition:box-shadow .2s ease; }
[class*="st-key-di_find_"] details:hover { box-shadow:0 2px 4px rgba(15,23,20,.05), 0 10px 24px rgba(15,23,20,.07); }
[class*="st-key-di_find_"] summary { padding:14px 18px !important; }
[class*="st-key-di_find_"] summary p { font-family:'Plus Jakarta Sans',-apple-system,'Segoe UI',sans-serif; font-size:14px; color:#515553;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
[class*="st-key-di_find_"] summary p strong { color:#101614; font-weight:750; }
[class*="st-key-di_find_"] details[open] summary p { white-space:normal; }
[class*="st-key-di_find_"] summary p::before { display:inline-block; min-width:74px; text-align:center; margin-right:14px; font-size:11px; font-weight:800;
    letter-spacing:.07em; border-radius:999px; padding:3px 10px; vertical-align:1px; }
.st-key-di_find_high_0 summary p::before, [class*="st-key-di_find_high_"] summary p::before { content:"HIGH"; color:#9b1c1c; background:#fdeeee; }
[class*="st-key-di_find_medium_"] summary p::before { content:"MEDIUM"; color:#8a5a00; background:#fff2d6; }
[class*="st-key-di_find_low_"] summary p::before { content:"LOW"; color:#0d734e; background:#dcf7ed; }
[class*="st-key-di_find_info_"] summary p::before { content:"INFO"; color:#047857; background:#c6f9e6; }
[class*="st-key-di_find_"] [data-testid="stExpanderDetails"] { padding:4px 22px 20px !important; }
.fd-sec { margin-top:10px; } .fd-sec p { margin:6px 0 0; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:14px; color:#313432; line-height:1.65; }
.fd-k { display:flex; gap:8px; align-items:center; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12px; font-weight:800; letter-spacing:.08em;
        text-transform:uppercase; color:#059669; } .fd-k svg { width:15px; height:15px; }
.fd-tab { width:100%; border-collapse:collapse; margin-top:8px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:13.5px; }
.fd-tab td { padding:8px 10px; border:none !important; border-bottom:1px solid #f0f1f0 !important; vertical-align:top; color:#515553; }
.fd-tab, .fd-tab tr { border:none !important; background:transparent !important; }
.fd-tab td:last-child { text-align:right; font-weight:700; color:#101614; white-space:nowrap; }
.fd-tab tr:last-child td { border-bottom:none; }
.fd-impact { margin-top:16px; padding:12px 16px; border-radius:12px; background:linear-gradient(90deg,#e1fcf2,#f0fdf6); border:1px solid #b0f5dc;
             font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:14px; color:#313432; line-height:1.55; display:flex; gap:12px; flex-wrap:wrap; }
.fd-impact b { color:#059669; white-space:nowrap; }
/* field dictionary list */
.fd-list { border:1px solid #e7e8e7; border-radius:14px; overflow:hidden; }
.fd-row { border-bottom:1px solid #f0f1f0; background:#fff; } .fd-row:last-child { border-bottom:none; }
.fd-row summary { list-style:none; cursor:pointer; display:grid; grid-template-columns:minmax(180px,1.2fr) minmax(140px,1fr) minmax(160px,1.4fr) 24px;
    gap:14px; align-items:center; padding:12px 16px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.fd-row summary::-webkit-details-marker { display:none; }
.fd-row summary:hover { background:#f3fbf8; }
.fd-col { font-weight:750; color:#101614; font-size:14px; } .fd-src { font-size:13px; color:#6f7572; }
.fd-role { font-size:12.5px; font-weight:650; color:#313432; background:#ebf6f2; border-radius:999px; padding:3px 10px; justify-self:start; }
.fd-chev { color:#a0a4a2; transition:transform .2s ease; display:flex; } .fd-chev svg { width:18px; height:18px; }
.fd-row[open] .fd-chev { transform:rotate(90deg); }
.fd-row[open] summary { background:#f3fbf8; }
.fd-body { padding:6px 18px 18px; background:#f8fdfb; border-top:1px solid #f0f1f0; }
.fd-grid { display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:22px; margin-top:8px; }
.fd-grid p { margin:6px 0 0; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:13.8px; color:#313432; line-height:1.6; }
.fd-attrs { display:flex; flex-wrap:wrap; gap:8px 22px; margin-top:14px; padding-top:12px; border-top:1px dashed #e7e8e7;
            font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:13px; color:#313432; }
.fd-attrs b { display:block; font-size:11.5px; color:#6f7572; font-weight:700; text-transform:uppercase; letter-spacing:.05em; }
.fd-yes { color:#047857; font-weight:700; } .fd-no { color:#8f9592; font-weight:700; }
.fd-disc div { display:flex; gap:10px; align-items:flex-start; padding:9px 0; border-bottom:1px solid #f0f1f0; font-family:'Plus Jakarta Sans',-apple-system,sans-serif;
               font-size:13.8px; color:#313432; line-height:1.55; }
.fd-disc div:last-child { border-bottom:none; } .fd-disc svg { width:16px; height:16px; color:#059669; flex:none; margin-top:3px; }
.fd-wide { width:100%; border-collapse:collapse; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:13.4px; }
.fd-wide th { text-align:left; font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:#6f7572; padding:8px 10px; border-bottom:1px solid #e7e8e7; background:#f3fbf8; }
.fd-wide td { padding:9px 10px; border:none !important; border-bottom:1px solid #f0f1f0 !important; color:#313432; vertical-align:top; line-height:1.5; }
.fd-wide th { border:none !important; border-bottom:1px solid #e7e8e7 !important; }
.fd-wide, .fd-wide tr { border:none !important; }
.fd-wide td.num { text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
/* equal-height chart panels side by side */
[data-testid="stHorizontalBlock"]:has(.st-key-ds_panel_missing) { align-items:stretch; }
[data-testid="stHorizontalBlock"]:has(.st-key-ds_panel_missing) > [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { height:100%; }
[data-testid="stLayoutWrapper"]:has(> .st-key-ds_panel_missing), [data-testid="stLayoutWrapper"]:has(> .st-key-ds_panel_missingcombos),
.st-key-ds_panel_missing, .st-key-ds_panel_missingcombos { height:100%; }
[class*="st-key-di_x_"] details { background:#ffffff; border:1px solid #e7e8e7 !important; border-radius:14px !important;
    box-shadow:0 1px 2px rgba(15,23,20,.04), 0 6px 16px rgba(15,23,20,.04); }
[class*="st-key-di_x_"] summary p { font-family:'Plus Jakarta Sans',-apple-system,'Segoe UI',sans-serif; font-weight:650; color:#101614; }
/* readiness checks */
.rc-wrap { margin-top:16px; padding-top:14px; border-top:1px solid #f0f1f0; }
.rc-title { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:14px; font-weight:750; color:#101614; margin-bottom:10px; }
.rc-title span { font-weight:500; color:#6f7572; font-size:13px; }
.rc-grid { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:12px; }
.rc-item { background:#f3fbf8; border:1px solid #e8f2ee; border-radius:12px; padding:10px 12px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.rc-top { display:flex; flex-direction:column; align-items:flex-start; gap:6px; } .rc-top b { font-size:13.5px; color:#101614; }
.rc-item p { margin:6px 0 0; font-size:12.4px; color:#515553; line-height:1.5; }
.dq-chip.bad { color:#9b1c1c; background:#fdeeee; } .dq-chip.info { color:#3a3f3c; background:#eef0ef; }
/* value kinds */
.vk-grid { display:grid; grid-template-columns:repeat(5, minmax(0,1fr)); gap:12px; margin-top:6px; }
.vk-item { border:1px solid #e8f2ee; border-radius:12px; padding:12px 14px; background:#f8fdfb; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; display:flex; flex-direction:column; }
.vk-head { display:flex; gap:10px; align-items:center; } .vk-head b { font-size:14px; color:#101614; line-height:1.3; }
.vk-item p { margin:8px 0 10px; font-size:13px; color:#515553; line-height:1.5; }
.vk-ex { margin-top:auto; font-size:12.5px; font-weight:650; color:#313432; background:#ebf6f2; border-radius:8px; padding:6px 9px; line-height:1.45; }
/* dictionary vs delivered */
.dd-table { width:100%; border-collapse:collapse; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:13.4px; table-layout:fixed; }
.dd-table th { text-align:left; font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:#6f7572; padding:9px 12px; background:#f3fbf8;
               border:none !important; border-bottom:1px solid #e7e8e7 !important; }
.dd-table td { padding:11px 12px; border:none !important; border-bottom:1px solid #f0f1f0 !important; vertical-align:top; color:#313432; line-height:1.55; }
.dd-table td:first-child { font-weight:750; color:#101614; }
.dd-table, .dd-table tr { border:none !important; }
.dd-ok { display:inline-block; margin-left:6px; font-size:11px; font-weight:700; color:#047857; } .dd-miss { display:inline-block; margin-left:6px; font-size:11px; font-weight:700; color:#9b1c1c; }
@media (max-width: 1180px) { .rc-grid { grid-template-columns:repeat(3, minmax(0,1fr)); } .vk-grid { grid-template-columns:repeat(2, minmax(0,1fr)); } }
@media (max-width: 860px) {
  .dd-table, .dd-table tbody, .dd-table tr, .dd-table td { display:block; width:100%; } .dd-table tr:first-child { display:none; }
  .dd-table tr { border-bottom:1px solid #e7e8e7 !important; padding:8px 0; } .dd-table td { border:none !important; padding:4px 4px; }
  .dd-table td::before { content:attr(data-label); display:block; font-size:11px; font-weight:800; letter-spacing:.06em; text-transform:uppercase; color:#6f7572; }
}
@media (max-width: 640px) { .rc-grid, .vk-grid { grid-template-columns:1fr; } }
/* findings show up to two lines, full text when open */
[class*="st-key-di_find_"] summary p { white-space:normal !important; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
[class*="st-key-di_find_"] details[open] summary p { -webkit-line-clamp:unset; display:block; }
/* misc */
.di-chip { display:inline-flex; align-items:center; gap:7px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.5px; color:#515553; margin-right:16px; }
.di-stats { display:grid; grid-template-columns:repeat(var(--n,3), minmax(0,1fr)); gap:14px; }
.di-stat { padding:16px 18px; display:grid; grid-template-rows:auto auto 1fr; }
.di-stat .v { font-size:26px; font-weight:800; color:#101614; letter-spacing:-.01em; line-height:1.1; }
.di-stat .l { font-size:13.3px; font-weight:700; color:#313432; margin-top:6px; }
.di-stat .n { font-size:12.3px; color:#6f7572; margin-top:4px; line-height:1.45; }
.di-stat .top { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
@media (max-width: 1180px) { .dq-cards { grid-template-columns:repeat(2, minmax(0,1fr)); } .di-stats { --n:2 !important; } }
@media (max-width: 860px) { .dq-points, .fd-grid { grid-template-columns:1fr; } [class*="st-key-di_find_"] summary p { white-space:normal; }
  .fd-row summary { grid-template-columns:1fr 24px; } .fd-src, .fd-role { grid-column:1; } .fd-chev { grid-row:1; grid-column:2; } }
@media (max-width: 640px) { .dq-cards { grid-template-columns:1fr; } .di-stats { --n:1 !important; } .dq-top { flex-direction:column; } }
</style>
"""


@st.cache_data(show_spinner=False)
def _cards() -> list[dict]:
    return dataset_cards({k: cleaned(k) for k in _LABEL})


@st.cache_data(show_spinner=False)
def _composition() -> dict:
    return passenger_composition(cleaned("flight"))


@st.cache_data(show_spinner=False)
def _qstats() -> dict:
    stars = sum(int((r[c].astype(str).str.strip() == "*").sum()) for k in _LABEL for r in [read_raw(k)] for c in r.columns)
    return quality_stats({k: cleaned(k) for k in _LABEL}, stars)


@st.cache_data(show_spinner=False)
def _checksums() -> tuple[bool, int]:
    return checksum_status()


def _d(iso: str) -> str:
    t = pd.Timestamp(iso)
    return f"{t.day} {t:%b %Y}"


def _plot(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CFG, key=key)


def _csv(df: pd.DataFrame, name: str, key: str, label: str = "Download data (CSV)") -> None:
    st.download_button(label, df.to_csv(index=False).encode(), file_name=name, mime="text/csv", key=key, icon=":material/download:", on_click="ignore")


def _stat(value: str, label: str, note: str, tone: str, icon: str) -> str:
    return (f'<div class="ds-card di-stat ds-anim"><div class="top"><div class="ds-ico {tone}" style="width:36px;height:36px">{ICON[icon]}</div></div>'
            f'<div class="v">{value}</div><div class="l">{label}</div><div class="n">{note}</div></div>')


# ------------------------------------------------------------------ tabs

def _tab_coverage(a: dict) -> None:
    h, f = a["hotel"], a["flight"]
    segs = [
        ("International guests", h["intl_train"]["dates"]["start"], h["intl_train"]["dates"]["end"], "#059669", "Guests observed (train)"),
        ("International guests", h["intl_test"]["dates"]["start"], h["intl_test"]["dates"]["end"], "#c7cac9", "Official test period — no Guests"),
        ("Domestic guests", h["dom_train"]["dates"]["start"], h["dom_train"]["dates"]["end"], "#059669", "Guests observed (train)"),
        ("Domestic guests", h["dom_test"]["dates"]["start"], h["dom_test"]["dates"]["end"], "#c7cac9", "Official test period — no Guests"),
        ("Flights", f["grain"]["monthly"]["start"], "2022-12-31", "#9fdcc1", "Flights reported monthly"),
        ("Flights", f["grain"]["daily"]["start"], f["grain"]["daily"]["end"], "#047857", "Flights reported daily"),
        ("Modelling window", str(config.MODEL_START.date()), str(config.MODEL_END.date()), "#1f2937", "Train + validate (daily flights ∩ observed Guests)"),
    ]
    rows = ["International guests", "Domestic guests", "Flights", "Modelling window"]
    fig = go.Figure()
    for row, s, e, col, lab in segs:
        span = ((pd.Timestamp(e) - pd.Timestamp(s)).days + 1) * 86_400_000
        fig.add_trace(go.Bar(y=[row], x=[span], base=[pd.Timestamp(s).isoformat()], orientation="h", marker=dict(color=col, cornerradius=6),
                             width=0.42, showlegend=False, hovertemplate=f"{row}<br>{s} → {e}<br>{lab}<extra></extra>"))
    for x, txt in [(config.FLIGHT_DAILY_START, "Daily flights begin"), (config.HOTEL_TEST_START, "Official test period")]:
        fig.add_vline(x=x.timestamp() * 1000, line=dict(color="#a0a4a2", width=1))
        fig.add_annotation(x=x.timestamp() * 1000, y=1.06, yref="paper", text=txt, showarrow=False, font=dict(size=12, color="#515553"), xanchor="left", xshift=4)
    fig.update_xaxes(type="date", range=["2021-12-15", "2026-03-20"])
    fig.update_yaxes(categoryorder="array", categoryarray=rows[::-1], showgrid=False)
    ds.style_fig(fig, height=300, legend=False)
    fig.update_layout(margin=dict(l=8, r=8, t=34, b=8), barmode="overlay")
    with ds.panel("timeline", "Data coverage timeline", "Which dates each file covers, where Guests are observed, and the window used to train and validate models."):
        _plot(fig, "di_timeline")
        chips = [("#059669", "Guests observed"), ("#c7cac9", "Official test period (no Guests)"), ("#9fdcc1", "Flights monthly (2022)"),
                 ("#047857", "Flights daily"), ("#1f2937", "Modelling window")]
        ds.html("".join(f'<span class="di-chip"><span class="ds-dot" style="background:{c}"></span>{t}</span>' for c, t in chips))
    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    ds.html('<div class="ds di-stats" style="--n:3">'
            + _stat(f"{(config.MODEL_END - config.MODEL_START).days + 1:,} days", "Train & validate",
                    f"{config.MODEL_START.day} {config.MODEL_START:%b %Y} → {config.MODEL_END.day} {config.MODEL_END:%b %Y}: daily flights and observed Guests overlap (after a {config.FEATURE_WARMUP_DAYS}-day feature warm-up).", "orange", "flask")
            + _stat(f"{h['intl_test']['dates']['distinct_dates']:,} days", "Forecast only",
                    f"{_d(h['intl_test']['dates']['start'])} → {_d(h['intl_test']['dates']['end'])}: flights are known, Guests are not provided in either test file.", "slate", "lock")
            + _stat("Monthly → daily", "Flight grain changes",
                    f"{f['grain']['monthly']['rows']:,} monthly rows in 2022, {f['grain']['daily']['rows']:,} daily rows from {_d(f['grain']['daily']['start'])}. 2022 flight rows appear only in monthly views; the last-year guest feature uses 2022 hotel records.", "blue", "plane")
            + "</div>")


def _tab_passengers() -> None:
    c = _composition()
    t = c["daily_totals"]
    ds.html('<div class="ds ds-defs">'
            f'<div class="ds-def"><b><span class="ds-dot" style="background:{SERIES[0]}"></span>Point-to-point (P2P) · {t["Total P2P"] / 1e6:.2f}M · {c["daily_p2p_share"]:.1%}</b>'
            '<p>Journey originates at the departure city and terminates in Abu Dhabi — the arrivals who can become hotel guests (not all do).</p></div>'
            f'<div class="ds-def"><b><span class="ds-dot" style="background:{SERIES[1]}"></span>Transfer · {t["Total Transfer"] / 1e6:.2f}M · {c["daily_transfer_share"]:.1%}</b>'
            '<p>Arrived on this route and changed to another flight at AUH. Connecting passengers; kept separate from P2P.</p></div>'
            f'<div class="ds-def"><b><span class="ds-dot" style="background:{SERIES[2]}"></span>Transit · {t["Total Transit"]:,.0f} passengers</b>'
            f'<p>Stayed on the same aircraft or a simpler connection. Non-zero on only {c["transit_rows_daily"]} daily rows.</p></div></div>')
    st.caption(f"Totals over daily route rows ({config.FLIGHT_DAILY_START:%b %Y} → {config.FLIGHT_END:%b %Y}). "
               f"PAX = P2P + Transfer + Transit holds on every row: {'yes' if c['identity_holds'] else 'NO — review'}.")
    m = c["monthly"]
    fig = go.Figure()
    for col, name, colr in [("Total P2P", "P2P", SERIES[0]), ("Total Transfer", "Transfer", SERIES[1]), ("Total Transit", "Transit", SERIES[2])]:
        fig.add_trace(go.Bar(x=m["month"], y=m[col], name=name, marker=dict(color=colr, line=dict(color="#ffffff", width=1)),
                             hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra>" + name + "</extra>"))
    fig.update_layout(barmode="stack", bargap=0.25, legend_traceorder="normal")
    fig.update_yaxes(title_text="Arriving passengers per month", separatethousands=True)
    ds.style_fig(fig, height=360)
    bc = c["by_country"].sort_values("p2p_share")
    avg = c["daily_p2p_share"]
    fig2 = go.Figure(go.Bar(y=bc["Departure Country Name"], x=bc["p2p_share"] * 100, orientation="h", marker=dict(color=SERIES[0], cornerradius=4),
                            text=[f"{v:.0%}" for v in bc["p2p_share"]], textposition="outside", cliponaxis=False, textfont=dict(color="#515553"),
                            hovertemplate="%{y}: %{x:.1f}% P2P<extra></extra>"))
    fig2.add_vline(x=avg * 100, line=dict(color="#a0a4a2", width=1))
    fig2.add_annotation(x=avg * 100, y=1.0, yref="paper", yanchor="bottom", text=f"All routes {avg:.0%}", showarrow=False,
                        font=dict(size=12, color="#515553"), yshift=4)
    fig2.update_xaxes(range=[0, 108], title_text="P2P share (%)")
    ds.style_fig(fig2, height=360, legend=False)
    fig2.update_layout(bargap=0.35, margin=dict(l=8, r=28, t=34, b=8))
    l, r = st.columns([1.45, 1], gap="medium")
    with l:
        with ds.panel("pax_month", "Arriving passengers by journey type", "Monthly totals, all routes. 2022 comes from monthly rows; later months sum daily rows."):
            _plot(fig, "di_pax_month")
            _csv(m, "passenger_composition_monthly.csv", "di_dl_paxm")
    with r:
        with ds.panel("pax_country", "Share of arrivals ending in Abu Dhabi", f"P2P share by departure country — the {len(bc)} largest by passengers (daily rows)."):
            _plot(fig2, "di_pax_country")
            _csv(c["by_country"], "p2p_share_by_country.csv", "di_dl_paxc")


@st.cache_data(show_spinner="Running the automated leakage tests …")
def _leak(model_id: str | None) -> list[dict]:
    return leakage.run_checks(analytic(), registry.load_metadata(model_id) if model_id else None)


def _tab_quality() -> None:
    s = _qstats()
    findings = audit_findings(s)
    mid = registry.active_model_id()
    md = registry.load_metadata(mid) if mid else None
    hp = registry.load_table(mid, "predictions_holdout") if mid else None
    # ---- 1 · key takeaways (plain language)
    pts = [
        ("shield", "teal", "A clean foundation",
         "No duplicate records, and passenger totals add up on every flight row, so aviation figures can be combined safely."),
        ("plane", "teal", "Learning from 2023 to 2025",
         f"Daily flights start in 2023, so models learn from {(config.MODEL_END - config.MODEL_START).days + 1:,} days and are scored on recent months they never saw."),
        ("users", "violet", "Market results are approximations",
         "Guests are counted by nationality and flights by departure country, so source-market scenarios are indicative."),
        ("lock", "slate", "Blank and unobserved values add uncertainty",
         f"Same-Day Guests blanks cannot be interpreted uniquely and {s['missing_nat_days']} nationality-days are unobserved; totals are barely affected."),
    ]
    with ds.panel("interpret", "Key takeaways", "What the audit means for predicting hotel demand."):
        ds.html('<div class="ds dq-points">' + "".join(
            f'<div class="dq-point"><div class="ds-ico {tone}" style="width:38px;height:38px">{ICON[ico]}</div><div><b>{esc(t)}</b><p>{esc(x)}</p></div></div>'
            for ico, tone, t, x in pts) + "</div>")
    st.write("")
    # ---- 2 · automated technical checks (Verified only when code ran and passed)
    checks = technical_checks(s, md, hp, _leak(mid), bool(_composition()["identity_holds"]))
    with st.container(key="ds_panel_checks"):
        n_ok = sum(c["status"] == "verified" for c in checks)
        ds.html(f'<div class="ds-ph">Automated technical checks</div><div class="ds-psub">{n_ok} of {len(checks)} verified by executed code · '
                '“Attention” marks an expected limitation that is handled · see the detailed evidence below.</div>')
        items = []
        for c in checks:
            word, tone, ico = STATUS[c["status"]]
            items.append(f'<div class="rc-item"><div class="rc-top"><span class="dq-chip {tone}">{ICON[ico]}{word}</span><b>{esc(c["check"])}</b></div>'
                         f'<p>{esc(c["evidence"])}</p></div>')
        ds.html(f'<div class="ds rc-grid" style="margin-top:12px">{"".join(items)}</div>')
        with st.expander("See detailed evidence", icon=":material/fact_check:", key="di_x_checks"):
            ds.html('<table class="fd-wide"><tr><th>Check</th><th>Status</th><th>What was executed</th><th>Result</th></tr>'
                    + "".join(f'<tr><td>{esc(c["check"])}</td><td><span class="dq-chip {STATUS[c["status"]][1]}">{STATUS[c["status"]][0]}</span></td>'
                              f'<td>{esc(c["method"])}</td><td>{esc(c["evidence"])}</td></tr>' for c in checks) + "</table>")
            leak = _leak(mid)
            ds.html('<div class="ds-note" style="margin:12px 0 6px"><b style="color:#101614">Leakage tests run on this page</b></div>'
                    '<table class="fd-wide"><tr><th>Test</th><th>Result</th><th>How</th></tr>'
                    + "".join(f'<tr><td>{esc(c["check"])}</td><td>{"Passed" if c["ok"] else ("Not run" if c["ok"] is None else "Failed")}</td><td>{esc(c["evidence"])}</td></tr>'
                              for c in leak) + "</table>")
    # ---- 3 · detailed findings
    keys = [f"di_find_{f['severity']}_{i}" for i, f in enumerate(findings)]

    def _set_all(value: bool) -> None:
        for k_ in keys:
            st.session_state[k_] = value

    ds.html('<div id="di-findings" style="height:22px"></div>')
    with st.container(key="di_findings_head"):
        c1, c2, c3 = st.columns([5, 1, 1], vertical_alignment="bottom")
        n_by = pd.Series([f["severity"] for f in findings]).value_counts()
        counts = " · ".join(f"{n_by.get(sv, 0)} {lab}" for sv, lab in [("high", "high"), ("medium", "medium"), ("low", "low"), ("info", "informational")] if n_by.get(sv, 0))
        c1.markdown(f'<div class="ds-h" style="margin:0">Audit findings</div><div class="ds-hsub" style="margin:4px 0 0">'
                    f'Ranked by impact on the models · {counts}. Open a finding for its evidence and how it is handled.</div>', unsafe_allow_html=True)
        c2.button("Expand all", key="di_find_expand", icon=":material/unfold_more:", width="stretch", on_click=_set_all, args=(True,))
        c3.button("Collapse all", key="di_find_collapse", icon=":material/unfold_less:", width="stretch", on_click=_set_all, args=(False,))
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    for f, k_ in zip(findings, keys):
        with st.expander(f"**{f['title']}** — {f['summary']}", key=k_, on_change="rerun"):
            ev_rows = "".join(f'<tr><td>{esc(e["label"])}</td><td>{esc(e["display"])}</td></tr>' for e in f["evidence"])
            a_, b_ = st.columns(2, gap="large")
            with a_:
                ds.html(f'<div class="fd-sec"><div class="fd-k">{ICON["search"]}What we found</div><p>{esc(f["found"])}</p></div>'
                        f'<div class="fd-sec"><div class="fd-k">{ICON["bulb"]}Why it matters</div><p>{esc(f["why"])}</p></div>')
            with b_:
                ds.html(f'<div class="fd-sec"><div class="fd-k">{ICON["chart"]}Evidence</div><table class="fd-tab">{ev_rows}</table></div>'
                        f'<div class="fd-sec"><div class="fd-k">{ICON["sliders"]}How it is handled</div><p>{esc(f["handling"])}</p></div>')
            ds.html(f'<div class="fd-impact"><b>Impact on the models</b><span>{esc(f["impact"])}</span></div>')
    # ---- 4 · blank, unobserved and zero values (progressive disclosure)
    st.write("")
    with st.expander("Blank, unobserved and zero values — where they occur", icon=":material/data_table:", key="di_x_values"):
        ds.html('<div class="ds-note" style="margin-bottom:8px">Five different things can look like “no value”. They are counted separately and handled differently.</div>'
                '<div class="ds vk-grid">' + "".join(
                    f'<div class="vk-item"><div class="vk-head"><div class="ds-ico {k["tone"]}" style="width:34px;height:34px">{ICON[k["icon"]]}</div><b>{esc(k["kind"])}</b></div>'
                    f'<p>{esc(k["text"])}</p><div class="vk-ex">{esc(k["example"])}</div></div>' for k in value_kinds(s)) + "</div>")
        st.write("")
        l, r = st.columns([1.12, 1], gap="medium")
        with l:
            h1, h2 = st.columns([2.2, 1])
            h1.markdown('<div class="ds-ph">Blank cells by column</div><div class="ds-psub">Share of empty cells per column in the selected file.</div>',
                        unsafe_allow_html=True)
            top = h2.selectbox("Columns", ["Top 10 columns", "All columns"], key="di_miss_top", label_visibility="collapsed")
            k = st.segmented_control("Dataset", list(_LABEL), format_func=_LABEL.get, default="intl_train", key="di_miss_ds", label_visibility="collapsed") or "intl_train"
            df = cleaned(k)
            if k == "flight":
                g = st.segmented_control("Rows", ["daily", "monthly"], default="daily", key="di_miss_grain", format_func=lambda x: f"{x} rows") or "daily"
                df = df[df["grain"] == g]
            prof = missing_profile(df)
            if top.startswith("Top"):
                prof = prof.head(10)
            prof = prof.sort_values("pct")
            fig = go.Figure(go.Bar(y=prof["column"], x=prof["pct"], orientation="h", marker=dict(color="#059669", cornerradius=4),
                                   text=[f"{v:.1f}%" for v in prof["pct"]], textposition="outside", cliponaxis=False, textfont=dict(color="#4b4f4d"),
                                   customdata=np.stack([prof["empty"], prof["rows"]], axis=1),
                                   hovertemplate="%{y}: %{customdata[0]:,} of %{customdata[1]:,} rows empty (%{x:.2f}%)<extra></extra>"))
            fig.update_xaxes(range=[0, 100], title_text="% of rows empty", dtick=20)
            ds.style_fig(fig, height=max(260, 30 * len(prof) + 80), legend=False)
            fig.update_layout(bargap=0.38, margin=dict(l=8, r=40, t=6, b=8))
            _plot(fig, f"di_miss_{k}_{top[:3]}")
        with r:
            wn = s["missing_by_nationality"]
            ds.html(f'<div class="ds-ph">Unobserved nationality-days</div><div class="ds-psub">{s["missing_nat_days"]} of {s["possible_nat_days"]:,} possible combinations '
                    f'({s["missing_nat_days"] / s["possible_nat_days"]:.2%}). The files do not say why.</div>')
            w = pd.DataFrame({"nat": [n.title() for n in wn], "days": list(wn.values())}).sort_values("days")
            fig2 = go.Figure(go.Bar(y=w["nat"], x=w["days"], orientation="h", marker=dict(color="#047857", cornerradius=4),
                                    text=[f"{v}" for v in w["days"]], textposition="outside", cliponaxis=False, textfont=dict(color="#4b4f4d"),
                                    hovertemplate="%{y}: %{x} days without a row (of " + f"{s['possible_nat_days'] // s['nationalities']:,}" + ")<extra></extra>"))
            fig2.update_xaxes(title_text="Days without a row", range=[0, max(w["days"]) * 1.2 if len(w) else 1])
            ds.style_fig(fig2, height=max(260, 30 * len(w) + 80), legend=False)
            fig2.update_layout(bargap=0.38, margin=dict(l=8, r=30, t=6, b=8))
            _plot(fig2, "di_missingcombos")
            ds.html(f'<div class="ds-note">The other {s["nationalities"] - len(wn)} nationalities have a row on every one of the {s["possible_nat_days"] // s["nationalities"]:,} days.</div>')


@st.cache_data(show_spinner=False)
def _dictionary_text() -> str:
    import pdfplumber
    from ..data_loading import discover_files
    with pdfplumber.open(discover_files()["dictionary"]) as pdf:
        return " ".join(" ".join((pg.extract_text() or "") for pg in pdf.pages).split())


def _dictionary_vs_delivered(s: dict, panel: bool = True) -> None:
    rows = dictionary_reconciliation(s)
    text = _dictionary_text()
    body = ('<table class="dd-table"><tr><th style="width:17%">Topic</th><th style="width:25%">Data_Dictionary.pdf says</th>'
            '<th style="width:31%">Delivered workbooks show</th><th style="width:27%">Modelling implication</th></tr>'
            + "".join(f'<tr><td data-label="Topic">{esc(r["topic"])}</td>'
                      f'<td data-label="Data_Dictionary.pdf says">{esc(r["documented"])}'
                      f'{"<span class=dd-ok>✓ in PDF</span>" if r["phrase"] in text else "<span class=dd-miss>wording not found in PDF</span>"}</td>'
                      f'<td data-label="Delivered workbooks show">{esc(r["observed"])}</td><td data-label="Modelling implication">{esc(r["implication"])}</td></tr>'
                      for r in rows) + "</table>")
    note = ('<div class="ds-note" style="margin-top:10px">Where the two disagree, QuanAra follows the delivered workbooks. '
            'The original files are never edited; “✓ in PDF” means the documented wording was found in the dictionary text.</div>')
    if panel:
        with ds.panel("dictvsdata", "Dictionary vs. delivered data", "What the official dictionary documents, what the workbooks actually contain, and what it means for the models."):
            ds.html(body + note)
    else:
        ds.html(body + note)


def _tab_explorer() -> None:
    with ds.panel("explorer", "Dataset explorer", "Filter any file and inspect or download the rows (cleaned view: parsed dates, numbers, suppressed cells empty)."):
        c1, c2, c3, c5, c4 = st.columns([1.05, 1.3, 1.4, 1.3, 0.65])
        k = c1.selectbox("File", list(_LABEL), format_func=_LABEL.get, key="di_ex_ds")
        df = cleaned(k)
        lo, hi = df["Date"].min().date(), df["Date"].max().date()
        rng = c2.date_input("Dates", (lo, hi), min_value=lo, max_value=hi, key=f"di_ex_dates_{k}")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            df = df[(df["Date"] >= pd.Timestamp(rng[0])) & (df["Date"] <= pd.Timestamp(rng[1]))]
        if "Nationality" in df.columns:
            pick = c3.multiselect("Nationality", sorted(df["Nationality"].unique()), key=f"di_ex_nat_{k}", placeholder="All nationalities")
            if pick:
                df = df[df["Nationality"].isin(pick)]
        elif k == "flight":
            pick = c3.multiselect("Departure country", sorted(df["Departure Country Name"].unique()), key="di_ex_cty", placeholder="All countries")
            if pick:
                df = df[df["Departure Country Name"].isin(pick)]
        query = c5.text_input("Search text", key=f"di_ex_q_{k}", placeholder="Search any text column")
        if query.strip():
            txt = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith(("str", "string"))]
            if txt:
                hit = np.zeros(len(df), dtype=bool)
                for c in txt:
                    hit |= df[c].astype(str).str.contains(query.strip(), case=False, regex=False).to_numpy()
                df = df[hit]
        n = c4.selectbox("Rows", [25, 100, 500], index=1, key="di_ex_n")
        view = df[[c for c in df.columns if c not in _HELPER_COLS]]
        def _n(n: int, one: str, many: str) -> str:
            return f"{n} {one if n == 1 else many}"
        ent = (_n(view['Nationality'].nunique(), "nationality", "nationalities") if "Nationality" in view else
               f"{_n(view['Departure Country Name'].nunique(), 'country', 'countries')} · {_n(view['Airline Name'].nunique(), 'airline', 'airlines')}"
               if k == "flight" else "one row per date")
        span = f"{view['Date'].min().day} {view['Date'].min():%b %Y} → {view['Date'].max().day} {view['Date'].max():%b %Y}" if len(view) else "—"
        ds.html(f'<div class="ds-note" style="margin:4px 0 10px"><b style="color:#101614">{len(view):,} rows</b> · {span} · {ent} · '
                'grey “None” = an empty cell (the dictionary’s “*” suppression marker)</div>')
        st.dataframe(view.head(n), hide_index=True, width="stretch", height=380,
                     column_config={"Date": st.column_config.DateColumn(format="YYYY-MM-DD")})
        _csv(view, f"{k}_filtered.csv", "di_ex_dl", "Download filtered rows (CSV)")


def _tab_distributions() -> None:
    pairs = {"International guests": ("intl_train", "intl_test"), "Domestic guests": ("dom_train", "dom_test"), "Flights (daily route rows)": ("flight", None)}
    with ds.panel("dist", "Feature distributions", "Where a field exists in both the training and the official test file, both are shown so shifts are visible."):
        c1, c2, c3, _ = st.columns([1, 1, 0.55, 1.2])
        grp = c1.selectbox("Data", list(pairs), key="di_dist_grp")
        tr_k, te_k = pairs[grp]
        tr = cleaned(tr_k)
        if tr_k == "flight":
            tr = tr[tr["grain"] == "daily"]
        num = [c for c in tr.select_dtypes("number").columns if c not in _HELPER_COLS and tr[c].nunique() > 1]
        default = num.index("Total P2P") if "Total P2P" in num else (num.index("Guests") if "Guests" in num else 0)
        col = c2.selectbox("Field", num, index=default, key=f"di_dist_col_{tr_k}")
        log = c3.toggle("Log scale", value=tr_k != "dom_train", key="di_dist_log")
        te = cleaned(te_k) if te_k else None
        series = [("Train", tr[col].dropna(), SERIES[0])]
        if te is not None and col in te.columns:
            series.append(("Official test", te[col].dropna(), SERIES[1]))
        fig = go.Figure()
        allv = pd.concat([s for _, s, _ in series])
        lo_v, hi_v = float(allv.min()), float(allv.max())
        use_log = log and lo_v > 0
        edges = np.logspace(np.log10(lo_v), np.log10(hi_v), 41) if use_log else np.linspace(lo_v, hi_v, 41)
        for name, s, colr in series:
            cnt, _ = np.histogram(s, bins=edges)
            share = cnt / max(cnt.sum(), 1) * 100
            lo_e, hi_e = edges[:-1], edges[1:]
            # log scale: bars are drawn on log10(value) so every bin has the same visual width
            x = (np.log10(lo_e) + np.log10(hi_e)) / 2 if use_log else (lo_e + hi_e) / 2
            labels = [f"{a:,.0f}–{b:,.0f}" if b >= 10 else f"{a:,.2f}–{b:,.2f}" for a, b in zip(lo_e, hi_e)]
            fig.add_trace(go.Bar(x=x, y=share, name=f"{name} (n={len(s):,})", marker=dict(color=colr), customdata=labels,
                                 opacity=0.75 if len(series) > 1 else 1, hovertemplate="%{customdata}: %{y:.2f}% of rows<extra>" + name + "</extra>"))
        fig.update_layout(barmode="overlay", bargap=0.06)
        if use_log:
            ticks = [v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5) if lo_v <= v <= hi_v * 1.0001] or [lo_v, hi_v]
            fig.update_xaxes(tickvals=[np.log10(v) for v in ticks], ticktext=[f"{v:,.0f}" for v in ticks])
        fig.update_xaxes(title_text=col + (" (log scale)" if use_log else ""))
        fig.update_yaxes(title_text="% of rows")
        ds.style_fig(fig, height=360, legend=True)
        _plot(fig, f"di_dist_{tr_k}_{col}_{use_log}")
        s0 = series[0][1]
        stats = [(f"{len(s0):,}", "rows (train)"), (f"{s0.mean():,.1f}", "mean"), (f"{s0.median():,.1f}", "median"),
                 (f"{s0.quantile(.05):,.1f} – {s0.quantile(.95):,.1f}", "P5 – P95"), (f"{tr[col].isna().mean():.1%}", "empty")]
        ds.html('<div class="ds" style="margin-top:4px">' + ds.tiles(stats) + "</div>")
        if col == "New Arrivals" and te is not None:
            ds.html(f'<div class="ds-note" style="margin-top:10px">Train rows with New Arrivals below 10: {(tr[col] < 10).mean():.1%}; test rows: {(te[col] < 10).mean():.1%}. '
                    'The official test extract excludes them, so models that use New Arrivals are trained on rows with ≥ 10 to match.</div>')


def _yesno(v: bool) -> str:
    return '<span class="fd-yes">Yes</span>' if v else '<span class="fd-no">No</span>'


@st.cache_data(show_spinner=False)
def _pdf_entry(column: str, source: str) -> dict | None:
    """The verbatim Data_Dictionary.pdf entry for a field (hotel fields: international section first)."""
    sections = ["flight"] if source == "flight" else ["international", "domestic"]
    for sec in sections:
        for name in (column, f"{column} TARGET VARIABLE"):
            e = pdf_dictionary.lookup(name, sec)
            if e:
                return e
    return None


@st.cache_data(show_spinner=False)
def _profiles() -> dict:
    return field_profiles({k: cleaned(k) for k in _LABEL})


def _tab_audit(a: dict) -> None:
    ft = field_table()
    prof = _profiles()
    with ds.panel("dictionary", "Data Dictionary — field by field",
                  f"All {len(ft)} column entries: the definition quoted from Data_Dictionary.pdf next to what the delivered files contain. Open a field for details."):
        c1, c2, c3, _ = st.columns([1.6, 1, 1.1, 0.6])
        q = c1.text_input("Search fields", key="di_ft_q", placeholder="e.g. P2P, transit, Guests", label_visibility="collapsed")
        src = c2.selectbox("Source", ["All files", "Hotel files", "Flight file"], key="di_ft_src", label_visibility="collapsed")
        role = c3.selectbox("Role", ["All roles", "Target & predictors", "Keys & identifiers", "Excluded or constant"], key="di_ft_role", label_visibility="collapsed")
        view = ft
        if src != "All files":
            view = view[view["source"].str.contains("flight" if src == "Flight file" else "intl|dom")]
        if role != "All roles":
            pat = {"Target & predictors": "target|predictor", "Keys & identifiers": "key|identifier", "Excluded or constant": "excluded|constant|component"}[role]
            view = view[view["role"].str.contains(pat, case=False)]
        if q.strip():
            ql = q.strip().lower()
            view = view[view.apply(lambda r: ql in " ".join(map(str, r.values)).lower(), axis=1)]
        rows = []
        for r in view.itertuples():
            e = _pdf_entry(r.column, "flight" if r.source == "flight" else "hotel")
            pdf_def = e["description"] if e else r.definition
            pdf_src = f"Data_Dictionary.pdf, p. {e['page']} (quoted)" if e else "Paraphrased — no exact PDF entry found"
            avail = (f"What-if: {_yesno(r.known_whatif)} · Test period: {_yesno(r.known_competition)} · Scenario lever: {_yesno(r.scenario_editable)}")
            rows.append(
                f'<details class="fd-row"><summary><span class="fd-col">{esc(r.column)}</span><span class="fd-src">{esc(r.source)}</span>'
                f'<span class="fd-role">{esc(r.role)}</span><span class="fd-chev">{ICON["chev"]}</span></summary>'
                f'<div class="fd-body"><div class="fd-grid"><div><div class="fd-k">{ICON["book"]}1 · Definition</div><p>{esc(pdf_def)}</p>'
                f'<p class="ds-note">{esc(pdf_src)}</p></div>'
                f'<div><div class="fd-k">{ICON["search"]}2 · What the delivered files contain</div><p>{esc(r.audit_finding)}</p>'
                f'<p class="ds-note">Computed now: {esc(prof.get(f"{r.column}|{r.source}", "—"))}</p></div></div>'
                f'<div class="fd-attrs"><span><b>3 · Documented type</b>{esc(e["type"]) if e else "—"}</span>'
                f'<span><b>4 · Documented examples</b>{esc(e["examples"]) if e and e.get("examples") else "—"}</span>'
                f'<span><b>5 · Type in the files</b>{esc(r.dtype_units)}</span><span><b>6 · Blank or missing means</b>{esc(r.missing_meaning)}</span>'
                f'<span><b>7 · Role in QuanAra</b>{esc(r.role)}</span><span><b>8 · Availability</b>{avail}</span></div></div></details>')
        ds.html(f'<div class="ds-note" style="margin:2px 0 10px">{len(view)} of {len(ft)} fields</div><div class="ds fd-list">{"".join(rows) or "<p class=ds-note>No field matches.</p>"}</div>')
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        _csv(ft, "quanara_field_dictionary.csv", "di_ft_dl", "Download field dictionary (CSV)")
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    with st.expander("Dictionary vs. delivered data", key="di_x_disc"):
        _dictionary_vs_delivered(_qstats(), panel=False)
    with st.expander("Market matching — nationality ↔ departure country", key="di_x_market"):
        s = _qstats()
        ds.html(f'<div class="ds-note" style="margin-bottom:10px">{s["matched_markets"]} of {s["nationalities"]} nationalities share a name with a flight departure country '
                f'({s["matched_guest_share"]:.1%} of international guest-days). The match is a modelling proxy: passengers departing a country are not all its nationals, '
                'and nationals also arrive through other hubs.</div>')
        it, fl = cleaned("intl_train"), cleaned("flight")
        g = it.groupby("Nationality")["Guests"].sum().rename("guest_days")
        p = fl[fl["grain"] == "daily"].groupby("market_key").agg(p2p=("Total P2P", "sum"), pax=("Total PAX", "sum"))
        tab = g.to_frame().join(p, how="left").reset_index().rename(columns={"index": "Nationality"})
        tab["matched"] = tab["Nationality"].isin(set(fl["market_key"]))
        tab["share_of_guest_days"] = tab["guest_days"] / tab["guest_days"].sum() * 100
        tab["p2p_share"] = tab["p2p"] / tab["pax"] * 100
        tab["Nationality"] = tab["Nationality"].str.title()
        st.dataframe(tab.sort_values("guest_days", ascending=False), hide_index=True, width="stretch", height=420, column_config={
            "Nationality": st.column_config.TextColumn(width="medium"),
            "guest_days": st.column_config.NumberColumn("Guest-days (train)", format="%,.0f"),
            "share_of_guest_days": st.column_config.NumberColumn("Share of guest-days", format="%.2f%%"),
            "matched": st.column_config.CheckboxColumn("Departure country match"),
            "p2p": st.column_config.NumberColumn("P2P arrivals (daily rows)", format="%,.0f"), "pax": st.column_config.NumberColumn("All arrivals", format="%,.0f"),
            "p2p_share": st.column_config.NumberColumn("P2P share", format="%.1f%%")},
            column_order=["Nationality", "matched", "guest_days", "share_of_guest_days", "p2p", "pax", "p2p_share"])
    ver = config.REPORTS_DIR / "verification_pages_1_2.json"
    if ver.exists():
        v = json.loads(ver.read_text())
        with st.expander(f"Independent recomputation — {v['passed']} of {v['total']} displayed figures match the raw files", key="di_x_verify"):
            ds.html('<div class="ds-note" style="margin-bottom:10px">Every figure on the Overview and Data Intelligence pages was recomputed directly from the raw workbooks '
                    f'(without the app’s own loading code) and from the active model’s saved predictions on {v["generated_at"][:10]}.</div>'
                    '<table class="fd-wide"><tr><th>Check</th><th>Shown in the app</th><th>Recomputed from source</th><th>Result</th></tr>'
                    + "".join(f'<tr><td>{esc(c["check"])}</td><td>{esc(c["displayed"])}</td><td>{esc(c["independent"])}</td>'
                              f'<td><span class="dq-chip {"good" if c["status"] == "PASS" else "warn"}">{"Pass" if c["status"] == "PASS" else "Fail"}</span></td></tr>'
                              for c in v["checks"]) + "</table>")
    with st.expander("Integration build log — rows before and after each step", key="di_x_log"):
        log = pd.DataFrame(analytic().log.steps)
        fmt = lambda x: "—" if pd.isna(x) else f"{int(x):,}"
        ds.html('<table class="fd-wide"><tr><th>Step</th><th>Rows before</th><th>Rows after</th><th>Note</th></tr>'
                + "".join(f'<tr><td>{esc(str(r.step))}</td><td class="num">{fmt(r.rows_before)}</td><td class="num">{fmt(r.rows_after)}</td><td>{esc(str(r.note or ""))}</td></tr>'
                          for r in log.itertuples()) + "</table>")


# ------------------------------------------------------------------ page

def _tab_overview() -> None:
    ok, n = _checksums()
    ds.html(ds.section("The five datasets", "Row counts, coverage and grain recomputed from each workbook.").replace('class="ds-h"', 'class="ds-h" style="margin-top:4px"'))
    cards = [ds.kpi_card(c["icon"], c["tone"], c["title"], c["value"], c["unit"], c["explanation"], c["secondary"],
                         ds.spark_line(c["series"], c["color"], f"dis_{c['key']}"), c["caption"], badge=c["badge"]) for c in _cards()]
    # five cards do not fit side by side at a readable width, so they sit in a carousel the judge moves (never auto-advancing)
    ds.html(ds.carousel(cards, "The five competition datasets", item="dataset"))
    ds.html_js(ds.CAROUSEL_JS)
    chip = (f'<span class="dq-chip good">{ICON["check"]}Originals verified · {n} files match their SHA-256 checksums</span>' if ok
            else f'<span class="dq-chip warn">{ICON["warning"]}Checksum mismatch — the copies differ from the recorded originals</span>')
    ds.html(f'<div class="ds" style="margin-top:16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap">{chip}'
            '<span class="ds-note">The app reads byte-identical, read-only copies of the six competition files; the originals are never modified.</span></div>')


def render() -> None:
    a = audit()
    ds.inject()
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(heroes.CSS, unsafe_allow_html=True)
    ds.html(heroes.banner("data", "Section 2", "Data <span>Intelligence</span>", "Explore, understand and validate the original competition datasets."))
    st.write("")
    sections = [("Overview", _tab_overview), ("Dataset explorer", _tab_explorer), ("Data coverage", lambda: _tab_coverage(a)),
                ("Passenger composition", _tab_passengers), ("Distributions", _tab_distributions), ("Data quality", _tab_quality),
                ("Audit & dictionary", lambda: _tab_audit(a))]
    with st.container(key="di_tabs"):
        # lazy tabs: only the selected section runs, so its charts are laid out at their visible width
        tabs = st.tabs([name for name, _ in sections], key="di_tabnav", on_change="rerun")
        for tab, (_, fn) in zip(tabs, sections):
            if tab.open:
                with tab:
                    fn()
    next_step("data")

