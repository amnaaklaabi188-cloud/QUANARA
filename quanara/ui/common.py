"""Shared UI helpers: cached data access, theme, cards and export widgets."""
from __future__ import annotations

import json
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .. import config, registry
from .. import model_identity as MI
from ..audit import run_audit, to_jsonable
from ..integration import get_analytic

PLOTLY_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
/* ---------- QuanAra design tokens (emerald · black · white) ---------- */
:root { --q-green:#059669; --q-green-ink:#047857; --q-green-deep:#065f46; --q-green-dark:#064e3b; --q-bright:#10b981; --q-mint:#34d399;
        --q-mint-l:#6ee7b7; --q-mint-bg:#ecfdf5; --q-mint-bd:#c6f0dc; --q-ink:#101614; --q-ink2:#3a3f3c; --q-muted:#6f7572; --q-line:#e5e8e6;
        --q-bg:#f5f7f6; --q-radius:16px; --q-shadow:0 1px 2px rgba(15,23,20,.04),0 8px 24px rgba(15,23,20,.06);
        --q-font:'Plus Jakarta Sans','Inter',-apple-system,'Segoe UI',Roboto,sans-serif;
        --navy:#072419; --sky:#059669; --amber:#1f2937; --ink:#101614; --ink2:#3a3f3c; --muted:#6f7572; --line:#e5e8e6; --card:#ffffff; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family:var(--q-font); }
.block-container { padding-top:2rem; max-width:1320px; }
h1, h2, h3 { letter-spacing:-0.01em; color:var(--q-ink); }
.stApp a { color:var(--q-green-ink); }
[data-testid="stHeader"] { background:transparent; }
/* ---------- sidebar: identical on every page ---------- */
section[data-testid="stSidebar"] { background:radial-gradient(130% 55% at 0% 100%, rgba(16,185,129,.16) 0%, rgba(16,185,129,0) 62%),
    linear-gradient(180deg,#040806 0%,#06100c 52%,#08170f 100%); border-right:1px solid #0c1c14; }
section[data-testid="stSidebar"] * { color:#e6ede9 !important; font-family:var(--q-font); }
section[data-testid="stSidebar"] [data-testid="stIconMaterial"] { font-family:'Material Symbols Rounded' !important; font-size:20px; color:#e6ede9 !important; }
[data-testid="stSidebarContent"] { display:flex; flex-direction:column; min-height:100%; }
[data-testid="stSidebarUserContent"] { margin-top:auto; padding-bottom:18px; }
[data-testid="stSidebarHeader"] { padding:22px 16px 8px 16px; height:auto; }
img[data-testid="stSidebarLogo"] { height:66px !important; max-height:66px !important; width:262px !important; max-width:none !important; object-fit:contain; }
[data-testid="stSidebarNav"] a { border-radius:10px; padding:9px 12px; margin:3px 10px; border:1px solid transparent; transition:background .15s, border-color .15s; }
[data-testid="stSidebarNav"] a:hover { background:rgba(52,211,153,.08) !important; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background:linear-gradient(90deg, rgba(16,185,129,.30) 0%, rgba(16,185,129,.12) 100%) !important;
    border-color:rgba(52,211,153,.80); box-shadow:0 0 0 1px rgba(16,185,129,.12), 0 6px 18px rgba(16,185,129,.16); }
[data-testid="stSidebarNav"] a[aria-current="page"] * { color:#ffffff !important; }
[data-testid="stSidebarNav"] a span { font-size:14.5px; }
[data-testid="stSidebarNavLinkContainer"] { width:100%; } [data-testid="stSidebarNav"] a { display:flex; box-sizing:border-box; flex:1 1 auto; width:auto; }
[data-testid="stSidebarNav"] a span:not([data-testid="stIconMaterial"]), [data-testid="stSidebarNav"] a [data-testid="stMarkdownContainer"],
[data-testid="stSidebarNav"] a p { white-space:normal !important; overflow:visible !important; text-overflow:clip !important; line-height:1.3; }
[data-testid="stNavSectionHeader"], [data-testid="stNavSectionHeader"] * { color:#9aa39e !important; font-size:11.5px !important; letter-spacing:.13em;
    text-transform:uppercase; font-weight:700 !important; }
section[data-testid="stSidebar"] hr { border-color:#16261e; }
.q-side-foot { position:relative; margin-top:-74px; padding:26px 4px 2px; }
.q-side-foot b, .q-side-foot span { text-shadow:0 2px 14px rgba(0,0,0,.95), 0 0 4px rgba(0,0,0,.9); }
.q-side-foot b { display:block; font-size:30px; line-height:1.05; letter-spacing:-.01em; color:#ffffff !important; font-weight:800; }
.q-side-foot b i { font-style:normal; color:#34d399 !important; }
.q-side-foot span { display:block; font-size:12.5px; color:#a9b3ae !important; line-height:1.4; margin-top:7px; letter-spacing:.04em; font-weight:600; }
.q-globe { margin:0 -16px 0 -24px; height:200px; overflow:hidden; }
.q-globe svg { width:100%; height:100%; display:block; }
.q-globe .qg-flow { stroke-dasharray:4 7; animation:qgFlow 3.5s linear infinite; }
.q-globe .qg-static { display:none; }
@keyframes qgFlow { to { stroke-dashoffset:-44; } }
@media (prefers-reduced-motion: reduce) { .q-globe .qg-flow { animation:none; } .q-globe .qg-motion { display:none; } .q-globe .qg-static { display:inline; } }
@media (max-height: 1000px) { .q-globe { height:140px; } [data-testid="stSidebarNav"] a { padding:7px 12px; margin:2px 10px; } }
@media (max-height: 820px) { .q-globe { height:86px; } .q-side-foot { margin-top:-40px; padding-top:14px; } .q-side-foot b { font-size:26px; }
  [data-testid="stSidebarNav"] a { padding:5px 12px; } [data-testid="stSidebarHeader"] { padding:14px 16px 4px 16px; } }
/* ---------- buttons ---------- */
[data-testid="stBaseButton-primary"] { background:#047857 !important; border-color:#047857 !important; color:#ffffff !important; }
[data-testid="stBaseButton-primary"]:hover { background:#065f46 !important; border-color:#065f46 !important; }
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primary"] span { color:#ffffff !important; }
[data-testid="stBaseButton-secondary"]:hover { border-color:var(--q-green) !important; color:var(--q-green-ink) !important; }
[data-testid="stBaseButton-secondary"]:hover p { color:var(--q-green-ink) !important; }
.stApp button:focus-visible, .stApp a:focus-visible, .stApp [role="tab"]:focus-visible { outline:2px solid var(--q-bright) !important; outline-offset:2px; }
/* ---------- tabs: one pill style for every page ---------- */
.stApp [role="tablist"] { gap:4px; background:#edf1ef; border-radius:12px; padding:4px; flex-wrap:wrap; overflow:visible !important; }
.stApp [data-baseweb="tab-highlight"], .stApp [data-baseweb="tab-border"] { display:none !important; }
.stApp [data-testid="stTab"] { height:38px; padding:0 15px; border-radius:9px; border:none !important; background:transparent; }
.stApp [data-testid="stTab"] p { font-weight:650; font-size:13.6px; color:#2b302d; }
.stApp [data-testid="stTab"]:hover p { color:var(--q-green-ink); }
.stApp [data-testid="stTab"][aria-selected="true"] { background:#047857; box-shadow:0 4px 12px rgba(4,120,87,.22); }
.stApp [data-testid="stTab"][aria-selected="true"] p { color:#ffffff; }
.stApp [role="tabpanel"] { animation:dsFadeUp .35s ease both; padding-top:6px; }
/* ---------- expanders, pills, page links ---------- */
.stApp [data-testid="stExpander"] details { border-radius:12px; border-color:var(--q-line); background:#ffffff; }
.stApp [data-testid="stExpander"] summary:hover p, .stApp [data-testid="stExpander"] summary:hover span { color:var(--q-green-ink); }
.stApp [data-testid="stPageLink-NavLink"] { border-radius:10px; }
@keyframes dsFadeUp { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:none; } }
@media (prefers-reduced-motion: reduce) { .stApp [role="tabpanel"] { animation:none; } }
/* ---------- legacy helpers still used by a few pages ---------- */
.as-card { background:#fff; border:1px solid var(--q-line); border-radius:14px; padding:16px 18px; height:100%; box-shadow:var(--q-shadow); }
.as-card .lbl { color:var(--q-ink2); font-size:13px; }
.as-card .val { color:var(--q-ink); font-size:28px; font-weight:750; line-height:1.15; margin-top:2px; }
.as-card .sub { color:var(--q-muted); font-size:12.5px; margin-top:4px; }
.as-card.accent { border-top:3px solid var(--q-green); } .as-card.amber { border-top:3px solid #1f2937; }
.as-badge { display:inline-block; padding:2px 9px; border-radius:999px; font-size:12px; font-weight:600; border:1px solid; }
.as-badge.good { color:#065f46; border-color:#a7f3d0; background:#ecfdf5; }
.as-badge.warn { color:#7a5200; border-color:#f3d27a; background:#fff8e6; }
.as-badge.crit { color:#9b1c1c; border-color:#efb1b1; background:#fdeeee; }
.as-badge.info { color:#3a3f3c; border-color:#d9dedb; background:#f3f5f4; }
.as-note { background:var(--q-mint-bg); border:1px solid var(--q-mint-bd); border-radius:10px; padding:10px 14px; color:#065f46; font-size:14px; }
.as-panel-title { font-size:13px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:var(--q-green-ink); margin:6px 0 2px; }
</style>
"""


def md(text: str) -> str:
    """Escape a literal '*' (the dictionary's suppression marker) so Markdown does not italicise it."""
    return text.replace("*", "\\*")


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(num: int | str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="as-page-head"><div class="num">Section {num}</div><h2>{escape(title)}</h2><p>{escape(subtitle)}</p></div>',
                unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "", kind: str = "accent") -> str:
    return (f'<div class="as-card {kind}"><div class="lbl">{escape(label)}</div><div class="val">{escape(value)}</div>'
            f'<div class="sub">{escape(sub)}</div></div>')


def cards(items: list[tuple], kind: str = "accent") -> None:
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        c.markdown(card(*it[:3], kind=it[3] if len(it) > 3 else kind), unsafe_allow_html=True)


def badge(text: str, level: str = "info") -> str:
    icon = {"good": "✓", "warn": "▲", "crit": "■", "info": "●"}[level]
    return f'<span class="as-badge {level}">{icon} {escape(text)}</span>'


def panel_title(text: str) -> None:
    st.markdown(f'<div class="as-panel-title">{escape(text)}</div>', unsafe_allow_html=True)


def plot(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOTLY_CONFIG, key=key)


def csv_button(df: pd.DataFrame, name: str, label: str = "Download data (CSV)", key: str | None = None) -> None:
    st.download_button(label, df.to_csv(index=False).encode(), file_name=name, mime="text/csv", key=key or f"dl_{name}", icon=":material/download:", on_click="ignore")


def png_button(fig: go.Figure, name: str, label: str = "Download chart (PNG)", key: str | None = None, width: int = 1100) -> None:
    """The PNG is rendered only when the button is clicked (deferred), so charts never slow the page."""
    h = fig.layout.height or 420

    def render() -> bytes:
        return fig.to_image(format="png", scale=2, width=width, height=h)
    st.download_button(label, render, file_name=name, mime="image/png", key=key or f"png_{name}", icon=":material/image:", on_click="ignore")


# ------------------------------------------------------------------ cached data

@st.cache_resource(show_spinner="Loading the analytical dataset …")
def analytic():
    return get_analytic()


@st.cache_data(show_spinner="Auditing the five workbooks …")
def audit() -> dict:
    p = config.REPORTS_DIR / "audit.json"
    if p.exists():
        return json.loads(p.read_text())
    from ..data_loading import file_inventory, load_all
    a = run_audit(load_all(), file_inventory())
    config.ensure_dirs()
    p.write_text(json.dumps(to_jsonable(a), indent=2))
    return json.loads(p.read_text())


@st.cache_data(show_spinner=False)
def cleaned(dataset: str) -> pd.DataFrame:
    from ..data_loading import load_dataset
    return load_dataset(dataset)


def models_df() -> pd.DataFrame:
    return registry.models_frame()


def active_id() -> str | None:
    return registry.active_model_id()


def model_label(mid: str) -> str:
    """The shared display name (quanara.model_identity); the active model is starred."""
    try:
        md = registry.load_metadata(mid)
    except Exception:
        return "Unknown model"
    lab = MI.labels(registry.list_models()).get(mid, MI.name(md))
    return f"{'★ ' if mid == active_id() else ''}{lab}"


def go_to(key: str) -> None:
    from .pages import PAGES
    st.switch_page(PAGES[key])


# ------------------------------------------------------------------ workflow continuity

NEXT = {
    "home": ("data", "Data Intelligence", "Check the quality and coverage of the five datasets."),
    "data": ("features", "Feature Engineering", "See which predictors are built from these files."),
    "features": ("eda", "Correlation & Visualization Lab", "Explore how the predictors move with hotel demand."),
    "eda": ("train", "ML Training Studio", "Train models on the selected features."),
    "train": ("eval", "Model Evaluation", "Check how trained models perform on unseen months."),
    "eval": ("sim", "What-if Simulator", "Use a validated model to test aviation scenarios."),
    "sim": ("copilot", "AI Copilot", "Ask for an evidence-based explanation of your scenario."),
    "copilot": ("method", "Methodology & Limitations", "Review the assumptions and limits behind every result."),
    "method": ("demo", "Judges' Demo", "See the whole solution in five short stages."),
    "demo": ("home", "Welcome & Overview", "Return to the project overview."),
}

_NEXT_CSS = """<style>
.st-key-next_step { margin-top:22px; background:#ffffff; border:1px solid #e7e8e7; border-radius:16px; padding:12px 18px;
    box-shadow:0 1px 2px rgba(15,23,20,.04),0 6px 18px rgba(15,23,20,.05); }
.st-key-next_step [data-testid="stMarkdownContainer"] { margin-bottom:0 !important; }
.nx-k { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:11.5px; font-weight:800; letter-spacing:.12em; color:#6f7572; text-transform:uppercase; }
.nx-d { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:13.2px; color:#515553; }
.st-key-next_step a[data-testid="stPageLink-NavLink"] { border:1px solid #c6f0dc; background:#f0fdf6; border-radius:10px; }
.st-key-next_step a[data-testid="stPageLink-NavLink"] p { font-weight:700; color:#047857; }
</style>"""


def next_step(current: str) -> None:
    """One slim card linking each section to the next stage of the analytical workflow."""
    from .pages import PAGES
    key, title, desc = NEXT[current]
    st.markdown(_NEXT_CSS, unsafe_allow_html=True)
    with st.container(key="next_step"):
        c1, c2 = st.columns([2.4, 1.2], vertical_alignment="center")
        c1.markdown(f'<div class="nx-k">Next step</div><div class="nx-d">{escape(desc)}</div>', unsafe_allow_html=True)
        c2.page_link(PAGES[key], label=title, icon=PAGES[key].icon, width="stretch")
