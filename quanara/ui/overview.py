"""Section 1 — Welcome & Overview: the challenge, the evidence base and the three-stage solution.

Every number is read from the competition files or from the active model's saved artefacts at run
time (see quanara/overview_facts.py); nothing on this page is typed in.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import config, registry
from .. import model_identity as MI
from ..overview_facts import data_facts, model_facts
from . import brand, heroes
from . import design as ds
from .common import go_to, next_step
from .design import ICON

CSS = """
<style>
.ov-q { display:flex; gap:14px; align-items:center; padding:13px 18px; margin-top:14px; border-radius:16px; background:#ffffff;
        border:1px solid #cfeedd; box-shadow:0 1px 2px rgba(15,23,20,.03); font-family:var(--q-font); }
.ov-q b { display:block; font-size:12px; color:#3a3f3c; letter-spacing:.08em; text-transform:uppercase; }
.ov-q q { font-size:16.5px; color:#101614; font-weight:600; quotes:"\\201C" "\\201D"; }
.ov-head { display:flex; justify-content:space-between; align-items:flex-end; gap:16px; flex-wrap:wrap; margin:30px 0 14px; font-family:var(--q-font); }
.ov-head .ds-h { margin:0 0 4px; } .ov-head .ds-hsub { margin:0; }
.ov-chips { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
.ov-chip { display:inline-flex; gap:8px; align-items:center; font-size:12.8px; color:#3a3f3c; padding:7px 12px; border-radius:10px; }
.ov-chip.live i { width:8px; height:8px; border-radius:50%; background:#10b981; box-shadow:0 0 0 3px rgba(16,185,129,.18); display:inline-block; }
.ov-chip.model { background:#ffffff; border:1px solid #d6e6dd; font-weight:650; color:#101614; }
.ov-chip.model svg { width:16px; height:16px; color:#059669; }
.ov-flow { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }
.ov-stage { padding:18px 20px 16px; display:flex; gap:14px; align-items:flex-start; font-family:var(--q-font); }
.ov-stage > div:last-child { display:flex; flex-direction:column; gap:5px; min-width:0; }
.ov-stage .n { font-size:11.5px; font-weight:800; letter-spacing:.12em; color:#047857; }
.ov-stage .t { font-size:16.5px; font-weight:750; color:#101614; line-height:1.3; }
.ov-stage .d { font-size:13.4px; color:#4b4f4d; line-height:1.5; }
.ov-stage .p { font-size:12px; color:#6f7572; margin-top:2px; }
.st-key-ov_cta { background:radial-gradient(90% 140% at 100% 0%, rgba(16,185,129,.28) 0%, rgba(16,185,129,0) 55%), linear-gradient(100deg,#03110b 0%,#062619 60%,#0a3a27 100%);
    border-radius:18px; padding:16px 22px; box-shadow:0 10px 26px rgba(4,16,11,.22); animation:dsFadeUp .5s ease both; }
.st-key-ov_cta [data-testid="stMarkdownContainer"] { margin-bottom:0 !important; }
.ov-cta { display:flex; gap:16px; align-items:center; font-family:var(--q-font); }
.ov-cta svg { width:30px; height:30px; color:#34d399; flex:none; }
.ov-cta-t { font-size:18px; font-weight:800; color:#ffffff; }
.ov-cta-s { font-size:13.4px; color:#cfe9dc; margin-top:3px; }
.st-key-ov_cta [data-testid="stBaseButton-primary"] { background:#34d399 !important; border-color:#34d399 !important; }
.st-key-ov_cta [data-testid="stBaseButton-primary"]:hover { background:#6ee7b7 !important; border-color:#6ee7b7 !important; }
.st-key-ov_cta [data-testid="stBaseButton-primary"] p, .st-key-ov_cta [data-testid="stBaseButton-primary"] span { color:#03110b !important; font-weight:750; }
.st-key-ov_cta [data-testid="stBaseButton-secondary"] { background:transparent; border:1px solid rgba(110,231,183,.55); }
.st-key-ov_cta [data-testid="stBaseButton-secondary"] p, .st-key-ov_cta [data-testid="stBaseButton-secondary"] span { color:#ffffff !important; }
.st-key-ov_cta [data-testid="stBaseButton-secondary"]:hover { border-color:#6ee7b7 !important; }
.hz-title .q-ink { color:#101614; }
@media (max-width: 1000px) { .ov-flow { grid-template-columns:1fr; } }
@media (max-width: 900px) { .st-key-ov_cta [data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
  .st-key-ov_cta [data-testid="stColumn"] { flex:1 1 100% !important; min-width:100% !important; } }
</style>
"""


def _m(ts) -> str:
    return pd.Timestamp(ts).strftime("%b %Y")


@st.cache_data(show_spinner="Reading the competition files …")
def _facts() -> dict:
    return data_facts()


def _cards(f: dict, m: dict | None, md: dict | None) -> str:
    flight = ds.kpi_card("plane", "teal", "Flight operations", f"{f['flight_rows']:,}", "route records",
                         "Arrivals into Abu Dhabi by departure city and airline.",
                         f"<b>{_m(f['flight_start'])} – {_m(f['flight_end'])}</b> · monthly in 2022, daily from {_m(f['flight_daily_start'])}",
                         ds.spark_line(f["flight_monthly_pax"], "#059669", "ovs1"), "Arriving passengers per month")
    hotel = ds.kpi_card("users", "teal", "Hotel guests", f"{f['labelled_records']:,}", "labelled records",
                        f"Daily Guests for {f['intl_nationalities']} nationalities plus domestic guests.",
                        f"<b>{_m(f['hotel_start'])} – {_m(f['hotel_end'])}</b> · {f['test_intl_rows'] + f['test_dom_rows']:,} unlabelled test records to {_m(f['test_end'])}",
                        ds.spark_line(f["intl_monthly_guests"], "#059669", "ovs2"), "International guests per month")
    target = ds.kpi_card("target", "teal", "Prediction target", "Guests", "per day",
                         "Hotel guests recorded on each date, international and domestic.",
                         "<b>Scopes:</b> international total, by nationality, domestic, overall",
                         ds.tiles([(f"{f['intl_avg_daily_total']:,.0f}", "International"), (f"{f['dom_avg_daily']:,.0f}", "Domestic")]),
                         f"Average guests per day, {_m(f['hotel_start'])} – {_m(f['hotel_end'])}")
    if m is None or md is None:
        model = ds.kpi_card("cpu", "slate", "Active model", "None", "", "Activate a What-if model in the ML Training Studio.", "No validation results yet.",
                            ds.tiles([("—", "R²"), ("—", "MAE"), ("—", "WMAPE")]), "", badge=("Inactive", "grey"))
    else:
        model = ds.kpi_card("cpu", "teal", "Active model", MI.algorithm(md), "", f"{MI.target(md)} · {MI.scope(md).lower()}",
                            f"<b>Validated</b> {_m(m['holdout'][0])} – {_m(m['holdout'][1])} on {m['n_holdout']:,} unseen {MI.row_unit(md)}",
                            ds.tiles([(f"{m['r2']:.3f}", "R²"), (f"{m['mae']:,.0f}", "MAE"), (f"{m['wmape']:.1%}", "WMAPE")]),
                            "Recomputed from the saved holdout predictions" if m["consistent"] else "Saved and recomputed metrics differ",
                            badge=("Active", "green") if m["consistent"] else ("Check", "amber"))
    return ds.kpi_grid([flight, hotel, target, model], cols=4)


STAGES = [
    ("01", "database", "Understand the data", "Audit the five competition workbooks, build trailing aviation features and explore how they move with hotel demand.",
     "Data Intelligence · Feature Engineering · Correlation Lab"),
    ("02", "bars", "Train and evaluate", "Fit models on earlier months and score them on six unseen months — no random splits.",
     "ML Training Studio · Model Evaluation"),
    ("03", "trend", "Simulate aviation changes", "Change seats, frequency, load factor or passenger mix and re-predict with a validated model.",
     "What-if Simulator · AI Copilot"),
]


def render() -> None:
    f = dict(_facts())
    m = model_facts()
    md = registry.load_metadata(m["model_id"]) if m else None
    ds.inject()
    st.markdown(heroes.CSS + brand.MARK_CSS + CSS, unsafe_allow_html=True)
    title = f'<i class="hz-mark">{brand.mark_svg(46, "ovm", animated=True)}</i><b class="q-ink">Quan</b><span>Ara</span>'
    ds.html(heroes.banner("overview", brand.TAGLINE, title,
                          "Predict how changes in flight operations shift hotel demand in Abu Dhabi, and test capacity scenarios before they happen.",
                          title_px=46))
    ds.html(f'<div class="ov-q ds-anim"><div class="ds-ico teal">{ICON["bulb"]}</div><div><b>Challenge question</b>'
            '<q>How might we build a tool that predicts how flight changes affect hotel visits in Abu Dhabi?</q></div></div>')
    chips = (f'<span class="ov-chip live"><i></i>Live data from {f["n_datasets"]} workbooks</span>'
             + (f'<span class="ov-chip model">{ICON["cpu"]}Active model: {MI.algorithm(md)}</span>' if md else ""))
    ds.html('<div class="ds">'
            f'<div class="ov-head"><div>{ds.section("At a glance", "Read live from the competition files and the active model.")}</div>'
            f'<div class="ov-chips">{chips}</div></div>' + _cards(f, m, md)
            + ds.section("How it works", "Three stages, each built on the one before.")
            + '<div class="ov-flow">' + "".join(
                f'<div class="ds-card ov-stage ds-anim"><div class="ds-ico teal">{ICON[ic]}</div><div><span class="n">STAGE {n}</span>'
                f'<span class="t">{t}</span><span class="d">{d}</span><span class="p">{p}</span></div></div>'
                for n, ic, t, d, p in STAGES) + "</div></div>")
    st.write("")
    with st.container(key="ov_cta"):
        c1, c2, c3 = st.columns([2.3, 1, 1], vertical_alignment="center")
        c1.markdown(f'<div class="ov-cta">{ICON["plane"]}<div><div class="ov-cta-t">Run a what-if scenario</div><div class="ov-cta-s">'
                    'Change seat capacity for a source market and see the predicted effect on hotel guests.</div></div></div>', unsafe_allow_html=True)
        if c2.button("Open the Simulator", type="primary", icon=":material/flight_takeoff:", width="stretch", key="ov_btn_sim"):
            go_to("sim")
        demo_ready = (config.ARTIFACTS_DIR / "judges_demo.json").exists()
        if c3.button("Judges' Demo", icon=":material/verified:", width="stretch", key="ov_btn_demo", disabled=not demo_ready):
            go_to("demo")
    with st.expander(f"About {brand.NAME}", icon=":material/info:"):
        r = f.get("p2p7_guests_r")
        st.markdown(brand.DESCRIPTION)
        st.markdown(f"**The approach.** Point-to-point passengers end their journey in Abu Dhabi, so they are the arrivals most likely to need a hotel room — though not all do. "
                    f"International guests track trailing 7-day P2P arrivals closely (r = {r:.2f}, {_m(config.MODEL_START)} – {_m(config.MODEL_END)}), "
                    "but both also follow seasons and long-term growth. The models learn this association; scenarios are model-based estimates, not causal effects. "
                    "Source markets pair a guest nationality with flights from the country of the same name — a proxy, not passenger attribution.")
    next_step("home")
