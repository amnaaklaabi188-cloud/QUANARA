"""Shared design system for the redesigned pages (Welcome & Overview, Data Intelligence).

One look everywhere: Plus Jakarta Sans, silver page, white 18px-radius cards with soft shadows,
emerald / deep-green accents on white, charcoal for comparison series, gentle fade-in and hover.

KPI cards use CSS Grid *subgrid*: every card spans the same six row tracks of its parent grid
(head · KPI · explanation · secondary · visual · caption), so headings, numbers, mini-charts and
captions line up across a row whatever the text length — no fixed positioning.
"""
from __future__ import annotations

from contextlib import contextmanager
from html import escape

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .hero import hero_svg

# chart series colours: the validated categorical slots (blue, orange, aqua) of the data-viz palette
SERIES = ["#059669", "#1f2937", "#9ca3af"]
INK, INK2, MUTED, LINE = "#101614", "#515553", "#6f7572", "#e7e8e7"
FONT = "'Plus Jakarta Sans', 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif"

_I = 'xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"'
ICON = {
    "plane": f'<svg {_I}><path d="M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z"/></svg>',
    "users": f'<svg {_I}><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "home": f'<svg {_I}><path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h14V9.5"/><path d="M9.5 21v-6h5v6"/></svg>',
    "target": f'<svg {_I}><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "cpu": f'<svg {_I}><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/></svg>',
    "database": f'<svg {_I}><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/></svg>',
    "sliders": f'<svg {_I}><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>',
    "chart": f'<svg {_I}><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>',
    "pin": f'<svg {_I}><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>',
    "bulb": f'<svg {_I}><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg>',
    "orbit": f'<svg {_I}><circle cx="12" cy="12" r="3"/><path d="M12 2a10 10 0 1 0 10 10"/><path d="M12 6a6 6 0 1 0 6 6"/></svg>',
    "play": f'<svg {_I}><polygon points="6 3 20 12 6 21 6 3"/></svg>',
    "arrow": f'<svg {_I}><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>',
    "flask": f'<svg {_I}><path d="M9 3h6"/><path d="M10 3v6L4.5 19a1.5 1.5 0 0 0 1.3 2h12.4a1.5 1.5 0 0 0 1.3-2L14 9V3"/><path d="M7.5 15h9"/></svg>',
    "shield": f'<svg {_I}><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>',
    "lock": f'<svg {_I}><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>',
    "check": f'<svg {_I}><path d="M20 6 9 17l-5-5"/></svg>',
    "warning": f'<svg {_I}><path d="m21.7 18-8-14a2 2 0 0 0-3.4 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>',
    "info": f'<svg {_I}><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>',
    "file": f'<svg {_I}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/></svg>',
    "search": f'<svg {_I}><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>',
    "chev": f'<svg {_I}><path d="m9 18 6-6-6-6"/></svg>',
    "book": f'<svg {_I}><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5z"/><path d="M4 19.5A2.5 2.5 0 0 0 6.5 22H20v-5"/></svg>',
    "globe": f'<svg {_I}><circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
    "layers": f'<svg {_I}><path d="m12 2 10 5-10 5L2 7l10-5z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/></svg>',
    "split": f'<svg {_I}><path d="M16 3h5v5"/><path d="M8 3H3v5"/><path d="M12 22v-8.3a4 4 0 0 0-1.17-2.83L3 3"/><path d="m15 9 6-6"/></svg>',
    "calendar": f'<svg {_I}><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M16 2v4M8 2v4M3 10h18"/></svg>',
    "eye": f'<svg {_I}><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>',
    "scatter": f'<svg {_I}><circle cx="7.5" cy="16.5" r="1.5"/><circle cx="12" cy="11" r="1.5"/><circle cx="17" cy="7" r="1.5"/><circle cx="16" cy="15" r="1.5"/><path d="M3 3v18h18"/></svg>',
    "trend": f'<svg {_I}><path d="m22 7-8.5 8.5-5-5L2 17"/><path d="M16 7h6v6"/></svg>',
    "clock": f'<svg {_I}><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>',
    "bars": f'<svg {_I}><path d="M3 3v18h18"/><rect x="7" y="12" width="3" height="6"/><rect x="12" y="8" width="3" height="10"/><rect x="17" y="5" width="3" height="13"/></svg>',
    "brain": f'<svg {_I}><circle cx="5" cy="6" r="2"/><circle cx="5" cy="18" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="19" cy="18" r="2"/><path d="M7 6.8 10.3 11M7 17.2 10.3 13M13.7 11 17 6.8M13.7 13 17 17.2"/></svg>',
    "hotel": f'<svg {_I}><path d="M4 21V5a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v16"/><path d="M2 21h20"/><path d="M9 21v-4h6v4"/><path d="M9 7h.01M15 7h.01M9 11h.01M15 11h.01"/></svg>',
    "scale": f'<svg {_I}><path d="M12 3v18"/><path d="M5 7h14"/><path d="m5 7-3 7a3 3 0 0 0 6 0z"/><path d="m19 7-3 7a3 3 0 0 0 6 0z"/></svg>',
    "trophy": f'<svg {_I}><path d="M6 9H4a2 2 0 0 1 0-4h2"/><path d="M18 9h2a2 2 0 0 0 0-4h-2"/><path d="M6 3h12v6a6 6 0 0 1-12 0z"/><path d="M12 15v4"/><path d="M8 21h8"/></svg>',
    "gear": f'<svg {_I}><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>',
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
.ds, .ds * { box-sizing:border-box; }
.ds { --blue:#059669; --violet:#047857; --teal:#047857; --orange:#1f2937; --ink:#101614; --ink2:#4b4f4d; --muted:#6f7572; --line:#e5e8e6;
      font-family:'Plus Jakarta Sans','Inter',-apple-system,'Segoe UI',Roboto,sans-serif; color:var(--ink); }
.ds-card { background:#fff; border:1px solid #e5e8e6; border-radius:16px;
           box-shadow:0 1px 2px rgba(15,23,20,.04), 0 10px 28px rgba(15,23,20,.06); transition:transform .2s ease, box-shadow .2s ease; }
.ds-card:hover { transform:translateY(-2px); box-shadow:0 2px 4px rgba(15,23,20,.05), 0 16px 36px rgba(15,23,20,.09); }
@keyframes dsFadeUp { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:none; } }
.ds-anim { animation:dsFadeUp .55s ease both; }
.ds-anim:nth-child(2) { animation-delay:.06s; } .ds-anim:nth-child(3) { animation-delay:.12s; } .ds-anim:nth-child(4) { animation-delay:.18s; }
.ds-anim:nth-child(5) { animation-delay:.24s; } .ds-anim:nth-child(6) { animation-delay:.30s; }
/* banner (aviation hero) */
.ds-hero { position:relative; border-radius:22px; overflow:hidden; border:1px solid var(--line);
           background:linear-gradient(180deg,#f6fefb 0%,#dafbef 55%,#ccf3e4 100%); box-shadow:0 10px 30px rgba(15,23,20,.07); }
.ov-hero-art { position:absolute; inset:0; width:100%; height:100%; display:block; }
.ds-hero-fade { position:absolute; inset:0; background:linear-gradient(90deg, rgba(250,252,251,.97) 0%, rgba(250,252,251,.92) 40%, rgba(250,252,251,0) 64%); }
.ds-hero-text { position:absolute; left:40px; top:50%; transform:translateY(-50%); width:50%; }
.ds-kicker { font-size:12.5px; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:#515553; }
.ds-title { font-weight:800; letter-spacing:-.02em; line-height:1.05; margin-top:10px; color:var(--ink); }
.ds-title span { background:linear-gradient(90deg,#059669,#047857); -webkit-background-clip:text; background-clip:text; color:transparent; }
.ds-tag { font-size:24px; font-weight:600; color:#1f2937; margin-top:8px; }
.ds-lead { font-size:16px; font-weight:700; color:#1f2937; margin-top:8px; line-height:1.4; }
.ds-lead + .ds-sub { font-size:14px; margin-top:4px; max-width:none; }
.ds-sub { font-size:15.5px; color:var(--ink2); margin-top:8px; max-width:470px; line-height:1.5; }
.ds-pill { position:absolute; right:22px; top:20px; display:flex; gap:10px; align-items:center; padding:10px 14px; border-radius:14px;
           background:rgba(255,255,255,.8); border:1px solid rgba(255,255,255,.9); backdrop-filter:blur(6px); box-shadow:0 4px 14px rgba(15,23,20,.08); }
.ds-pill svg { width:20px; height:20px; color:var(--blue); }
.ds-pill b { display:block; font-size:13.5px; color:var(--ink); } .ds-pill small { font-size:12px; color:var(--muted); }
.ov-flight-static { display:none; }
@media (prefers-reduced-motion: reduce) { .ov-flight-anim { display:none; } .ov-flight-static { display:inline; } .ds-anim { animation:none; } .ds-card { transition:none; } }
/* icons */
.ds-ico { width:42px; height:42px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex:none; }
.ds-ico svg { width:21px; height:21px; }
.ds-ico.blue, .ds-ico.teal, .ds-ico.indigo, .ds-ico.green { background:#e6f7ef; color:#047857; }
.ds-ico.violet { background:#ddf3e8; color:#065f46; }
.ds-ico.orange, .ds-ico.dark { background:#eef0ef; color:#1f2937; } .ds-ico.slate { background:#f0f2f1; color:#4b4f4d; }
/* section headings */
.ds-h { font-size:23px; font-weight:750; letter-spacing:-.01em; margin:32px 0 4px; color:var(--ink); }
.ds-hsub { font-size:14.5px; color:var(--muted); margin-bottom:14px; }
/* KPI cards — subgrid rows: head · KPI · explanation · secondary · visual · caption */
.ds-kpis { display:grid; grid-template-columns:repeat(var(--cols,4), minmax(0,1fr)); gap:16px; }
.ds-kpi { display:grid; grid-row:span 6; grid-template-rows:subgrid; grid-template-columns:minmax(0,1fr); row-gap:0; padding:18px 18px 15px; min-width:0; overflow:hidden; }
.ds-kpi > * { min-width:0; }
@supports not (grid-template-rows: subgrid) { .ds-kpi { display:flex; flex-direction:column; } .ds-kpi-vis { margin-top:auto; } }
.ds-kpi-head { display:flex; gap:11px; align-items:center; min-height:42px; }
.ds-kpi-tw { display:flex; flex-direction:column; align-items:flex-start; gap:4px; min-width:0; }
.ds-kpi-title { font-size:13.8px; font-weight:700; color:#313432; line-height:1.25; }
.ds-badge { font-size:11px; font-weight:700; border-radius:999px; padding:2px 8px; white-space:nowrap; line-height:1.5; }
.ds-badge.green { color:#047857; background:#d1fae5; } .ds-badge.grey { color:#4b4f4d; background:#eef0ef; } .ds-badge.amber { color:#8a5a00; background:#fff2d6; }
.ds-kpi-value { margin-top:14px; font-size:28px; font-weight:800; letter-spacing:-.015em; line-height:1.1; color:var(--ink); }
.ds-kpi-value small { font-size:14px; font-weight:600; color:var(--muted); letter-spacing:0; margin-left:4px; white-space:nowrap; display:inline-block; }
.ds-kpi-expl { margin-top:6px; font-size:13.3px; color:var(--ink2); line-height:1.45; }
.ds-kpi-sec { margin-top:10px; padding-top:10px; border-top:1px solid #f0f1f0; font-size:12.4px; color:var(--muted); line-height:1.5; }
.ds-kpi-sec b { color:#313432; font-weight:650; }
.ds-kpi-vis { margin-top:12px; height:52px; align-self:end; }
.ds-kpi-vis svg { width:100%; height:52px; display:block; }
.ds-kpi-cap { margin-top:6px; font-size:11.5px; color:var(--muted); line-height:1.35; }
.ds-tiles { display:grid; grid-template-columns:repeat(var(--n,3), minmax(0,1fr)); gap:6px; height:52px; }
.ds-tile { background:#f0f9f6; border-radius:10px; padding:6px 9px; display:flex; flex-direction:column; justify-content:center; min-width:0; }
.ds-tile b { font-size:15.5px; color:var(--ink); line-height:1.15; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
/* tile values shrink with the tiles themselves (container queries), so a value such as "-0.048" never truncates */
.ds-tiles { container-type:inline-size; }
@container (max-width: 250px) { .ds-tile { padding:6px 6px; } .ds-tile b { font-size:13.6px; } .ds-tile span { font-size:10.4px; } }
@container (max-width: 215px) { .ds-tile { padding:5px 5px; } .ds-tile b { font-size:12.4px; letter-spacing:-.01em; } }
@container (max-width: 185px) { .ds-tile { padding:5px 4px; } .ds-tile b { font-size:11.4px; } }
.ds-tile span { font-size:11px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
/* chart / content panels built from st.container(key="ds_panel_…") */
[class*="st-key-ds_panel_"] { background:#fff; border:1px solid #e5e8e6; border-radius:16px; padding:20px 22px 14px;
    box-shadow:0 1px 2px rgba(15,23,20,.04), 0 10px 28px rgba(15,23,20,.06); animation:dsFadeUp .5s ease both; }
/* Streamlit pulls each markdown block up by 1rem (for a trailing <p>); a panel ending in an HTML block keeps its bottom padding */
[class*="st-key-ds_panel_"] > [data-testid="stElementContainer"]:last-child [data-testid="stMarkdownContainer"]:has(> div:last-child),
[data-testid="stExpanderDetails"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:last-child [data-testid="stMarkdownContainer"]:has(> div:last-child) { margin-bottom:0; }
.ds-ph { font-family:'Plus Jakarta Sans',-apple-system,'Segoe UI',sans-serif; font-size:16.5px; font-weight:750; color:#101614; }
.ds-psub { font-family:'Plus Jakarta Sans',-apple-system,'Segoe UI',sans-serif; font-size:13px; color:#6f7572; margin-top:2px; line-height:1.45; }
.ds-note { font-family:'Plus Jakarta Sans',-apple-system,'Segoe UI',sans-serif; font-size:12.5px; color:#6f7572; line-height:1.5; }
.ds-defs { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin-top:8px; }
.ds-def { background:#f3fbf8; border:1px solid #e8f2ee; border-radius:12px; padding:10px 12px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.ds-def b { display:flex; gap:7px; align-items:center; font-size:13px; color:#101614; } .ds-def p { margin:4px 0 0; font-size:12.4px; color:#515553; line-height:1.45; }
.ds-dot { width:10px; height:10px; border-radius:3px; display:inline-block; flex:none; }
/* responsive */
.ds-kq { container-type:inline-size; }
@container (max-width: 1120px) { .ds-kpis { --cols:2 !important; } }
@container (max-width: 560px) { .ds-kpis { --cols:1 !important; } }
@media (max-width: 860px) { .ds-hero-text { width:62% !important; } .ds-hero-fade { background:linear-gradient(90deg, rgba(250,252,251,.97) 0%, rgba(250,252,251,.9) 55%, rgba(250,252,251,.2) 82%); } .ds-defs { grid-template-columns:1fr; } }
@media (max-width: 1000px) { .ds-pill { display:none; } }
@media (max-width: 640px) {
  .ds-hero { height:auto !important; } .ov-hero-art { position:relative; height:160px; } .ds-hero-fade { display:none; }
  .ds-hero-text { position:relative; left:0; top:0; transform:none; width:auto !important; padding:22px 20px 6px; } .ds-title { font-size:38px !important; } .ds-tag { font-size:20px; }
  .ds-kpis { --cols:1 !important; }
}
/* QuanAra loading states: spinners use the brand ring colours */
[data-testid="stSpinnerIcon"], [data-testid="stSpinner"] i { border-color:rgba(110,231,183,.45) !important; border-top-color:#059669 !important; }
[data-testid="stSpinner"] [data-testid="stMarkdownContainer"] p { color:#313432; font-weight:600; }
</style>
"""


def inject() -> None:
    st.markdown(CSS + CAROUSEL_CSS, unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render trusted, app-generated HTML (newlines removed so Markdown never re-interprets it)."""
    st.markdown(markup.replace("\n", ""), unsafe_allow_html=True)


def hero(kicker: str, title_html: str, tagline: str, subtitle: str, pill: tuple[str, ...] | None, height: int = 300,
         title_px: int = 54, lead: str = "", text_width: int = 50) -> str:
    """pill = (title, subtitle[, icon name])."""
    pill_html = (f'<div class="ds-pill">{ICON[pill[2] if len(pill) > 2 else "pin"]}<div><b>{escape(pill[0])}</b><small>{escape(pill[1])}</small></div></div>'
                 if pill else "")
    tag = f'<div class="ds-tag">{escape(tagline)}</div>' if tagline else ""
    return (f'<div class="ds-hero ds-anim" style="height:{height}px">{hero_svg()}<div class="ds-hero-fade"></div>'
            f'<div class="ds-hero-text" style="width:{text_width}%"><div class="ds-kicker">{escape(kicker)}</div><div class="ds-title" style="font-size:{title_px}px">{title_html}</div>'
            f'{tag}{f"<div class=ds-lead>{escape(lead)}</div>" if lead else ""}<div class="ds-sub">{escape(subtitle)}</div></div>{pill_html}</div>')


def section(title: str, sub: str = "") -> str:
    return f'<div class="ds-h">{escape(title)}</div>' + (f'<div class="ds-hsub">{escape(sub)}</div>' if sub else "")


def spark_line(values, color: str, gid: str) -> str:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2 or v.max() == v.min():
        return ""
    w, h, pad = 300, 52, 4
    x = np.linspace(0, w, len(v))
    y = pad + (h - 2 * pad) * (1 - (v - v.min()) / (v.max() - v.min()))
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" aria-hidden="true"><defs><linearGradient id="{gid}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{color}" stop-opacity="0.26"/><stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
            f'<polygon points="0,{h} {pts} {w},{h}" fill="url(#{gid})"/>'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" vector-effect="non-scaling-stroke" stroke-linejoin="round"/></svg>')


def tiles(items: list[tuple[str, str]]) -> str:
    return f'<div class="ds-tiles" style="--n:{len(items)}">' + "".join(
        f'<div class="ds-tile"><b>{escape(v)}</b><span>{escape(lab)}</span></div>' for v, lab in items) + "</div>"


def kpi_card(icon: str, tone: str, title: str, value: str, unit: str, explanation: str, secondary: str, visual: str, caption: str,
             badge: tuple[str, str] | None = None) -> str:
    """secondary may contain <b> markup generated by the app; everything else is escaped."""
    b = f'<span class="ds-badge {badge[1]}">{escape(badge[0])}</span>' if badge else ""
    u = f"<small>{escape(unit)}</small>" if unit else ""
    return (f'<div class="ds-card ds-kpi ds-anim"><div class="ds-kpi-head"><div class="ds-ico {tone}">{ICON[icon]}</div>'
            f'<div class="ds-kpi-tw"><div class="ds-kpi-title">{escape(title)}</div>{b}</div></div>'
            f'<div class="ds-kpi-value">{escape(value)}{u}</div><div class="ds-kpi-expl">{escape(explanation)}</div>'
            f'<div class="ds-kpi-sec">{secondary}</div><div class="ds-kpi-vis">{visual}</div><div class="ds-kpi-cap">{escape(caption)}</div></div>')


def kpi_grid(cards: list[str], cols: int) -> str:
    """Cards in one row when the content column is wide enough; the wrapper is a size container, so the
    column count follows the space actually available (not the window, which also holds the sidebar)."""
    return f'<div class="ds-kq"><div class="ds ds-kpis" style="--cols:{cols}">{"".join(cards)}</div></div>'


_ARROW_L = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M19 12H5"/><path d="m11 18-6-6 6-6"/></svg>'
_ARROW_R = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h14"/><path d="m13 6 6 6-6 6"/></svg>'

CAROUSEL_CSS = """
<style>
/* Card carousel: user-driven only (arrows, dots, swipe, trackpad, keyboard); never moves by itself. */
.qc { container-type:inline-size; position:relative; }
.qc-bar { display:flex; justify-content:space-between; align-items:center; gap:12px; margin:0 0 10px; }
.qc-status { font-size:12.8px; color:#6f7572; font-variant-numeric:tabular-nums; }
.qc-nav { display:flex; gap:8px; }
.qc-btn { width:40px; height:40px; border-radius:50%; border:1px solid #bfe9d3; background:#fff; color:#047857; display:grid; place-items:center;
          cursor:pointer; box-shadow:0 1px 2px rgba(15,23,20,.05), 0 6px 16px rgba(4,120,87,.10); transition:background .2s ease, color .2s ease, border-color .2s ease, opacity .2s ease, transform .2s ease; }
.qc-btn svg { width:19px; height:19px; }
.qc-btn:hover:not(:disabled) { background:#047857; border-color:#047857; color:#fff; transform:translateY(-1px); }
.qc-btn:focus-visible, .qc-track:focus-visible, .qc-dot:focus-visible { outline:2px solid #34d399; outline-offset:3px; }
.qc-btn:disabled { opacity:.35; cursor:default; box-shadow:none; }
.qc-track { --qc-n:3; --qc-gap:18px; display:grid; grid-auto-flow:column; grid-template-rows:repeat(6, auto);
            grid-auto-columns:calc((100% - (var(--qc-n) - 1) * var(--qc-gap)) / var(--qc-n)); column-gap:var(--qc-gap);
            overflow-x:auto; overflow-y:hidden; overscroll-behavior-x:contain; scroll-snap-type:x mandatory; scrollbar-width:none;
            padding:6px 4px 16px; margin:-6px -4px -10px; scroll-padding-inline:4px; border-radius:18px; }
.qc-track::-webkit-scrollbar { display:none; }
.qc-track > * { scroll-snap-align:start; }
.qc-track .ds-kpi { overflow:visible; }
.qc.is-moving .qc-track { scroll-snap-type:none; }
.qc-dots { display:flex; justify-content:center; gap:7px; margin-top:8px; }
.qc-dot { width:8px; height:8px; padding:0; border-radius:999px; border:0; background:#cfe3d9; cursor:pointer; transition:width .3s cubic-bezier(.65,0,.35,1), background .3s ease; }
.qc-dot[aria-current="true"] { width:24px; background:#059669; }
@container (max-width: 860px) { .qc-track { --qc-n:2; } }
@container (max-width: 540px) { .qc-track { --qc-n:1.12; --qc-gap:14px; } }
@media (prefers-reduced-motion: reduce) { .qc-btn, .qc-dot { transition:none; } }
</style>
"""

CAROUSEL_JS = """
<script>
(function () {
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var ease = function (t) { return t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
  function initAll() { document.querySelectorAll("[data-qc]:not([data-qc-ready])").forEach(init); }
  function init(root) {
    root.setAttribute("data-qc-ready", "1");
    var track = root.querySelector(".qc-track"), slides = Array.prototype.slice.call(track.children);
    var prev = root.querySelector('[data-qc-dir="-1"]'), next = root.querySelector('[data-qc-dir="1"]');
    var status = root.querySelector(".qc-status"), dotsBox = root.querySelector(".qc-dots"), anim = 0;
    var step = function () { return slides.length > 1 ? slides[1].offsetLeft - slides[0].offsetLeft : track.clientWidth; };
    var maxLeft = function () { return Math.max(0, track.scrollWidth - track.clientWidth); };
    var visible = function () { return Math.max(1, Math.floor((track.clientWidth + 2) / step() + 0.08)); };
    var positions = function () { return Math.max(1, Math.ceil(maxLeft() / step() - 0.05) + 1); };
    var index = function () { return Math.min(positions() - 1, Math.round(track.scrollLeft / step())); };
    function buildDots() {
      var n = positions(); if (dotsBox.children.length === n) return;
      dotsBox.innerHTML = "";
      for (var i = 0; i < n; i++) { var b = document.createElement("button"); b.type = "button"; b.className = "qc-dot";
        b.setAttribute("aria-label", "Go to position " + (i + 1) + " of " + n); b.dataset.i = i; dotsBox.appendChild(b); }
      dotsBox.hidden = n < 2;
    }
    function update() {
      buildDots();
      var i = index(), v = visible(), n = slides.length;
      prev.disabled = track.scrollLeft <= 2; next.disabled = track.scrollLeft >= maxLeft() - 2;
      status.textContent = v >= n ? "All " + n + " shown" : v === 1 ? "Showing " + (i + 1) + " of " + n : "Showing " + (i + 1) + "–" + Math.min(n, i + v) + " of " + n;
      Array.prototype.forEach.call(dotsBox.children, function (d, k) { d.setAttribute("aria-current", k === i ? "true" : "false"); });
    }
    function goTo(i) {
      var target = Math.max(0, Math.min(maxLeft(), i * step())), from = track.scrollLeft, t0 = null, dur = 520;
      cancelAnimationFrame(anim);
      if (reduce || Math.abs(target - from) < 1) { track.scrollLeft = target; update(); return; }
      root.classList.add("is-moving");
      var tick = function (ts) { if (t0 === null) t0 = ts; var p = Math.min(1, (ts - t0) / dur);
        track.scrollLeft = from + (target - from) * ease(p);
        if (p < 1) anim = requestAnimationFrame(tick); else { root.classList.remove("is-moving"); update(); } };
      anim = requestAnimationFrame(tick);
    }
    prev.addEventListener("click", function () { goTo(index() - 1); });
    next.addEventListener("click", function () { goTo(index() + 1); });
    dotsBox.addEventListener("click", function (e) { var d = e.target.closest(".qc-dot"); if (d) goTo(+d.dataset.i); });
    track.addEventListener("keydown", function (e) {
      var k = { ArrowLeft: -1, ArrowRight: 1 }[e.key];
      if (k) { e.preventDefault(); goTo(index() + k); }
      else if (e.key === "Home") { e.preventDefault(); goTo(0); } else if (e.key === "End") { e.preventDefault(); goTo(positions() - 1); }
    });
    var queued = false;
    track.addEventListener("scroll", function () { if (!queued) { queued = true; requestAnimationFrame(function () { queued = false; update(); }); } }, { passive: true });
    if (window.ResizeObserver) new ResizeObserver(update).observe(track); else window.addEventListener("resize", update);
    update();
  }
  initAll();
  // the cards are rendered by a separate element and may mount after this script; Streamlit also re-mounts them on reruns
  if (!window.__qcObserver && window.MutationObserver) {
    window.__qcObserver = new MutationObserver(function () { if (document.querySelector("[data-qc]:not([data-qc-ready])")) initAll(); });
    window.__qcObserver.observe(document.body, { childList: true, subtree: true });
  }
})();
</script>
"""


def carousel(cards: list[str], label: str, item: str = "card") -> str:
    """A horizontal card carousel (about three cards per view on desktop) moved only by the user."""
    n = len(cards)
    slides = "".join(c.replace('<div class="ds-card', f'<div role="group" aria-roledescription="slide" aria-label="{i + 1} of {n}" class="ds-card', 1)
                     for i, c in enumerate(cards))
    return (f'<div class="qc ds" data-qc><div class="qc-bar"><span class="qc-status" aria-live="polite">{n} {escape(item)}s</span>'
            f'<div class="qc-nav"><button type="button" class="qc-btn" data-qc-dir="-1" aria-label="Previous {escape(item)}">{_ARROW_L}</button>'
            f'<button type="button" class="qc-btn" data-qc-dir="1" aria-label="Next {escape(item)}">{_ARROW_R}</button></div></div>'
            f'<div class="qc-track" tabindex="0" role="region" aria-roledescription="carousel" aria-label="{escape(label)}">{slides}</div>'
            f'<div class="qc-dots" role="group" aria-label="Carousel position"></div></div>')


def html_js(script: str) -> None:
    """Run a trusted, app-generated script (st.html strips SVG, so visible markup goes through html() instead)."""
    st.html(script, unsafe_allow_javascript=True)


@contextmanager
def panel(key: str, title: str, sub: str = ""):
    with st.container(key=f"ds_panel_{key}"):
        st.markdown(f'<div class="ds-ph">{escape(title)}</div>' + (f'<div class="ds-psub">{escape(sub)}</div>' if sub else ""),
                    unsafe_allow_html=True)
        yield


def style_fig(fig: go.Figure, height: int | None = None, legend: bool = True) -> go.Figure:
    """Restyle a Plotly figure for a white panel in the shared look (does not change its data)."""
    fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(family=FONT, size=12.5, color=INK2),
                      margin=dict(l=8, r=8, t=12, b=8), showlegend=legend, transition=dict(duration=350, easing="cubic-in-out"),
                      hoverlabel=dict(bgcolor="#ffffff", bordercolor=LINE, font=dict(family=FONT, color=INK)),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=INK2), bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#e9f3ef", linecolor="#e2e4e3", tickfont=dict(color=MUTED), title_font=dict(color=INK2), automargin=True, zeroline=False)
    fig.update_yaxes(gridcolor="#e9f3ef", linecolor="#e2e4e3", tickfont=dict(color=MUTED), title_font=dict(color=INK2), automargin=True, zeroline=False)
    if height:
        fig.update_layout(height=height)
    return fig
