"""Section 4 — Correlation & Visualization Lab.

One filtered frame drives the scatter, the metric cards, the distributions and the CSV download,
so the numbers, the chart and the exported rows always describe the same days.
"""
from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from .. import config
from .. import correlation as C
from ..features import available_features
from . import design, heroes, lab_charts as LC
from .common import analytic, csv_button, plot, png_button, next_step
from .design import ICON

TRANSFORMS = {"levels": "Levels", "diff1": "Day-over-day Δ", "diff7": "Week-over-week Δ"}
HIGHLIGHT = {"none": "None", "year": "Year", "season": "Season", "daytype": "Weekday vs weekend"}
METHODS = ["Pearson", "Spearman", "Kendall"]
CTRL_KEYS = ["eda_level", "eda_market", "eda_x", "eda_y", "eda_rng", "eda_method", "eda_trend", "eda_hl", "eda_out", "eda_tf"]

CSS = """
<style>
[data-testid="stLayoutWrapper"]:has(> [class*="st-key-lab_c_"]) { height:100%; }
.st-key-lab_cards [data-testid="stHorizontalBlock"], .st-key-lab_row2 [data-testid="stHorizontalBlock"] { align-items:stretch; }
.st-key-lab_cards [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { height:100%; }
[class*="st-key-lab_c_"] { height:100%; background:#fff; border:1px solid #e7e8e7; border-radius:16px; padding:13px 15px 10px; gap:.4rem;
   box-shadow:0 1px 2px rgba(15,23,20,.04),0 8px 22px rgba(15,23,20,.05); animation:dsFadeUp .5s ease both; }
.st-key-lab_c_x { animation-delay:.05s; } .st-key-lab_c_y { animation-delay:.1s; } .st-key-lab_c_date { animation-delay:.15s; }
[class*="st-key-lab_c_"] [data-testid="stMarkdownContainer"], .st-key-lab_tools [data-testid="stMarkdownContainer"] { margin-bottom:0 !important; }
.lab-ch { display:flex; gap:10px; align-items:center; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.lab-ch b { font-size:13.8px; color:#101614; font-weight:750; } .lab-ch .ds-ico { width:34px; height:34px; } .lab-ch .ds-ico svg { width:18px; height:18px; }
.lab-cap { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:11.8px; color:#6f7572; line-height:1.4; }
.st-key-lab_tools { background:#fff; border:1px solid #e7e8e7; border-radius:16px; padding:10px 16px; box-shadow:0 1px 2px rgba(15,23,20,.04),0 8px 22px rgba(15,23,20,.05);
   animation:dsFadeUp .5s ease .18s both; }
.st-key-lab_tools label p { font-size:12.3px !important; color:#515553 !important; white-space:nowrap; }
.st-key-lab_tools button p { white-space:nowrap; font-size:13px; }
.st-key-lab_hm_actions button p { font-size:13px; }
.lab-metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.lab-m { border:1px solid #eceeed; border-radius:12px; padding:9px 12px; background:#f8fdfb; min-width:0; }
.lab-m span { display:block; font-size:12px; color:#515553; font-weight:600; }
.lab-m b { display:block; font-size:24px; font-weight:800; color:#101614; letter-spacing:-.01em; line-height:1.2; animation:fePop .45s cubic-bezier(.3,1.4,.6,1); }
.lab-m small { display:block; font-size:11.3px; color:#6f7572; line-height:1.35; margin-top:1px; }
@keyframes fePop { from { transform:scale(.8); opacity:.2; } to { transform:none; opacity:1; } }
.lab-title { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:18px; font-weight:800; color:#101614; letter-spacing:-.01em; }
.lab-title em { font-style:normal; color:#059669; }
.lab-note { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.3px; color:#6f7572; line-height:1.5; }
.lab-note b { color:#313432; }
.lab-warn { font-family:'Plus Jakarta Sans',-apple-system,sans-serif; font-size:12.4px; line-height:1.45; border-radius:10px; padding:8px 11px; display:flex; gap:8px;
   background:#fff7e8; color:#8a5a00; border:1px solid #fbe3b4; margin:6px 0; } .lab-warn svg { width:15px; height:15px; flex:none; margin-top:1px; }
.lab-info { background:#ecfdf5; color:#047857; border-color:#c6f0dc; }
.lab-stats { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:3px 6px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; margin-top:2px; }
.lab-stats div { background:#f0f9f6; border-radius:7px; padding:3px 8px; min-width:0; display:flex; justify-content:space-between; gap:6px; align-items:baseline; } .lab-stats b { font-size:12.2px; color:#101614; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.lab-stats span { font-size:10.2px; color:#6f7572; white-space:nowrap; }
/* relationship findings: full-width cards, horizontal badges (words never break mid-word), details on demand */
.lk { container-type:inline-size; margin-top:18px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; display:flex; flex-direction:column; gap:12px; }
.lk-title { font-size:16px; font-weight:750; color:#101614; } .lk-sub { font-size:12.8px; color:#6f7572; margin-top:-8px; line-height:1.45; }
.lk-card { display:grid; grid-template-columns:1fr; gap:10px 28px; border:1px solid #e3ece7; background:#fbfdfc; border-radius:14px; padding:16px 18px; }
@container (min-width: 760px) { .lk-card { grid-template-columns:minmax(250px, 0.9fr) minmax(0, 1.6fr); align-items:start; } }
.lk-head { display:flex; flex-direction:column; align-items:flex-start; gap:8px; min-width:0; }
.lk-badge { display:inline-block; max-width:100%; white-space:normal; overflow-wrap:normal; word-break:normal; font-size:11.5px; font-weight:700;
            line-height:1.35; letter-spacing:.01em; border-radius:999px; padding:4px 11px; }
.lk-badge.green { color:#065f46; background:#dcf5e9; } .lk-badge.amber { color:#7a4f00; background:#fff1d1; } .lk-badge.grey { color:#3f4542; background:#eceeed; }
.lk-card h4 { margin:0; font-size:17px; font-weight:750; color:#101614; line-height:1.25; }
.lk-stats { display:flex; flex-wrap:wrap; gap:8px; }
.lk-stats span { border:1px solid #e3ece7; background:#fff; border-radius:10px; padding:5px 10px; min-width:0; }
.lk-stats small { display:block; font-size:10.8px; color:#6f7572; font-weight:650; white-space:nowrap; }
.lk-stats b { font-size:15px; color:#101614; font-variant-numeric:tabular-nums; white-space:nowrap; }
.lk-body p { margin:0; font-size:14px; color:#2f3431; line-height:1.55; max-width:68ch; }
.lk-body details { margin-top:10px; } .lk-body summary { cursor:pointer; font-size:13px; font-weight:700; color:#047857; }
.lk-body ul { margin:8px 0 0; padding-left:18px; font-size:13px; color:#444846; line-height:1.55; max-width:72ch; } .lk-body li + li { margin-top:4px; }
.lab-sci { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px 24px; font-family:'Plus Jakarta Sans',-apple-system,sans-serif; }
.lab-sci p { margin:0; font-size:12.6px; color:#444846; line-height:1.5; } .lab-sci b { color:#101614; }
[class*="st-key-ds_panel_lab"] .stPlotlyChart { animation:labReveal .6s ease both; }
@keyframes labReveal { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }
@media (max-width: 1500px) {
  .st-key-lab_cards [data-testid="stHorizontalBlock"], .st-key-lab_tools [data-testid="stHorizontalBlock"] { flex-wrap:wrap; row-gap:10px; }
  .st-key-lab_cards [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex:1 1 45% !important; min-width:45% !important; }
  .st-key-lab_tools [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex:1 1 22% !important; min-width:22% !important; }
}
@media (max-width: 1100px) {
  .st-key-lab_cards [data-testid="stHorizontalBlock"], .st-key-lab_tools [data-testid="stHorizontalBlock"], .st-key-lab_row2 > div > [data-testid="stHorizontalBlock"] { flex-wrap:wrap; row-gap:10px; }
  .st-key-lab_cards [data-testid="stColumn"], .st-key-lab_row2 > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex:1 1 100% !important; min-width:100% !important; }
  .st-key-lab_tools [data-testid="stColumn"] { flex:1 1 45% !important; min-width:45% !important; }
  .lab-metrics { grid-template-columns:repeat(2,minmax(0,1fr)); } .lab-sci { grid-template-columns:1fr; }
  [class*="st-key-ds_panel_lab"] [data-testid="stHorizontalBlock"] { flex-wrap:wrap; row-gap:8px; }
  [class*="st-key-ds_panel_lab"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex:1 1 45% !important; min-width:45% !important; }
}
</style>
"""


# ------------------------------------------------------------------ cached data

@st.cache_data(show_spinner="Building the daily frame …")
def _frame(level: str, market: str | None) -> pd.DataFrame:
    return C.eda_frame(analytic(), level, market)


@st.cache_data(show_spinner=False)
def _guests(level: str, market: str | None) -> pd.DataFrame:
    return C.guests_series(analytic(), level, market)


@st.cache_data(show_spinner=False)
def _monthly(scope: str) -> pd.DataFrame:
    return C.monthly_history(analytic(), scope)


@st.cache_data(show_spinner="Computing lagged correlations …")
def _lags(level: str, market: str | None, base: str, window: int, max_lag: int, method: str, start: str, end: str) -> pd.DataFrame:
    return C.lag_correlation(_frame(level, market), base, window, "Guests", max_lag, method, pd.Timestamp(start), pd.Timestamp(end))


def _reset() -> None:
    for k in CTRL_KEYS:
        st.session_state.pop(k, None)


def _season(m: pd.Series) -> pd.Series:
    return np.select([m.isin([6, 7, 8]), m.isin([12, 1, 2])], ["Summer (Jun–Aug)", "Winter (Dec–Feb)"], "Spring / autumn")


def _groups(d: pd.DataFrame, how: str) -> pd.Series | None:
    if how == "year":
        return d["Date"].dt.year.astype(str)
    if how == "season":
        return pd.Series(_season(d["Date"].dt.month), index=d.index)
    if how == "daytype":
        return pd.Series(np.where(d["Date"].dt.dayofweek >= 5, "Weekend (Sat–Sun)", "Weekday"), index=d.index)
    return None


def _card_head(icon: str, tone: str, title: str) -> None:
    design.html(f'<div class="lab-ch"><div class="ds-ico {tone}">{ICON[icon]}</div><b>{escape(title)}</b></div>')


def _singular(u: str) -> str:
    for a, b in (("passengers", "passenger"), ("seats per flight", "seat per flight"), ("seats", "seat"), ("flights", "flight"), ("guests", "guest"),
                 ("routes", "route"), ("cities", "city"), ("airlines", "airline")):
        if u.startswith(a):
            return b + u[len(a):]
    return f"unit ({u})" if u else "unit"


def _p(v: float) -> str:
    return "—" if not np.isfinite(v) else ("< 1e-300" if v == 0 else f"{v:.1e}")


def _num(v: float, d: int = 3) -> str:
    if not np.isfinite(v):
        return "—"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.{d}f}"


def _stats_html(v: pd.Series, total: int, out: int) -> str:
    v = pd.to_numeric(v, errors="coerce")
    ok = v.dropna()
    items = [("n", f"{len(ok):,}"), ("missing", f"{total - len(ok):,}"), ("mean", _num(ok.mean())), ("median", _num(ok.median())),
             ("std", _num(ok.std())), ("min", _num(ok.min())), ("max", _num(ok.max())), ("outliers*", f"{out:,}")]
    return '<div class="lab-stats">' + "".join(f"<div><span>{escape(a)}</span><b>{escape(b)}</b></div>" for a, b in items) + "</div>"


def _findings_html(items: list[dict]) -> str:
    """'What the strongest links mean': one full-width card per relationship — horizontal badge, title, verified
    coefficients, one-sentence reading and the longer explanation behind a disclosure."""
    cards = []
    for f in items:
        stats_ = "".join(f'<span><small>{escape(k)}</small><b>{escape(v)}</b></span>' for k, v in f["stats"])
        det = "".join(f"<li>{escape(x)}</li>" for x in f["details"])
        cards.append(f'<article class="lk-card"><div class="lk-head"><span class="lk-badge {f["tone"]}">{escape(f["badge"])}</span>'
                     f'<h4>{escape(f["title"])}</h4><div class="lk-stats">{stats_}</div></div>'
                     f'<div class="lk-body"><p>{escape(f["summary"])}</p><details><summary>Technical details</summary><ul>{det}</ul></details></div></article>')
    return ('<div class="lk"><div class="lk-title">What the strongest links mean</div><div class="lk-sub">Feature-to-feature links describe redundancy '
            'between inputs; feature-to-target links describe association with Guests. Neither is a causal effect.</div>' + "".join(cards) + "</div>")


# ------------------------------------------------------------------ page

def render() -> None:
    design.inject()
    st.markdown(heroes.CSS + CSS, unsafe_allow_html=True)
    design.html(heroes.banner("correlation", "Section 4", 'Correlation &amp; <span>Visualization Lab</span>', "Observed relationships between aviation activity and hotel demand."))
    st.write("")
    a = analytic()

    # ---------------------------------------------------------------- controls row 1
    with st.container(key="lab_cards"):
        c = st.columns([1.15, 1.25, 1.2, 1.35], gap="small")
        with c[0], st.container(key="lab_c_level"):
            _card_head("globe", "blue", "Level")
            level = st.selectbox("Level", list(C.LEVELS), key="eda_level", label_visibility="collapsed")
            market = None
            if C.LEVELS[level] == "international_market":
                market = st.selectbox("Source market", a.markets, index=a.markets.index("INDIA"), format_func=str.title, key="eda_market",
                                      label_visibility="collapsed")
                design.html('<div class="lab-cap">Departure country is a proxy for nationality; the guest series is the nationality’s own.</div>')
            else:
                design.html('<div class="lab-cap">Choose the analysis grain for the charts below.</div>')
        df = _frame(level, market)
        num = C.numeric_columns(df)
        feats = [x for x in num if x != "Guests"]
        with c[1], st.container(key="lab_c_x"):
            _card_head("scatter", "violet", "X-axis feature")
            xcol = st.selectbox("X-axis feature", feats, index=feats.index("p2p_7d") if "p2p_7d" in feats else 0, format_func=C.label,
                                key="eda_x", label_visibility="collapsed")
            design.html('<div class="lab-cap">An aviation, calendar or history feature (horizontal axis).</div>')
        with c[2], st.container(key="lab_c_y"):
            _card_head("target", "teal", "Y-axis variable")
            yopts = [y for y in num if y != xcol]
            ycol = st.selectbox("Y-axis variable", yopts, index=0, format_func=C.label, key="eda_y", label_visibility="collapsed")
            design.html('<div class="lab-cap">Guests and other observed hotel fields exist only for training dates.</div>')
        end_max = C.availability_end([xcol, ycol] + (["Guests"] if C.is_hotel(ycol) or C.is_hotel(xcol) else []))
        with c[3], st.container(key="lab_c_date"):
            _card_head("calendar", "orange", "Date range")
            lo, hi = config.FLIGHT_DAILY_START.date(), end_max.date()
            rk = f"eda_rng_{hi}"
            default = (max(config.MODEL_START.date(), lo), hi)
            rng = st.slider("Date range", min_value=lo, max_value=hi, value=default, format="YYYY-MM-DD", key=rk, label_visibility="collapsed")
            design.html(f'<div class="lab-cap">Daily flights start {lo:%d %b %Y}; '
                        + (f'observed Guests end {hi:%d %b %Y} (the test period is unlabelled).' if hi == config.HOTEL_TRAIN_END.date()
                           else f'aviation data runs to {hi:%d %b %Y}.') + "</div>")

    # ---------------------------------------------------------------- controls row 2
    st.write("")
    with st.container(key="lab_tools"):
        t = st.columns([1.0, 1.05, 1.05, 1.05, 1.15, 0.42, 0.95, 1.0], vertical_alignment="bottom", gap="small")
        method = t[0].selectbox("Correlation method", METHODS, key="eda_method")
        trend = t[1].toggle("OLS trendline", value=True, key="eda_trend")
        hl = t[2].selectbox("Highlight by", list(HIGHLIGHT), format_func=HIGHLIGHT.get, key="eda_hl")
        show_out = t[3].toggle("Show outliers", value=True, key="eda_out", help="Extreme values (beyond 3×IQR on X or Y). Hiding them removes them "
                               "from the scatter, the metric cards and the distributions — the count is always shown.")
        tf = t[4].selectbox("Series", list(TRANSFORMS), format_func=TRANSFORMS.get, key="eda_tf",
                            help="Differences remove shared trends and seasonality, a check on whether a correlation is driven by both series drifting together.")
        t[5].button("", icon=":material/restart_alt:", key="eda_reset", on_click=_reset, help="Reset every filter on this page")

    # ---------------------------------------------------------------- the filtered frame
    start, end = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    w = df.copy()
    w["_x"], w["_y"] = C.transform(w[xcol].astype(float), tf), C.transform(w[ycol].astype(float), tf)
    d = w[(w["Date"] >= start) & (w["Date"] <= end) & w["_x"].notna() & w["_y"].notna()].copy()
    out = C.outlier_mask(d, ["_x", "_y"])
    d["extreme_value"] = out
    d_used = d if show_out else d[~out]
    s = C.pair_stats(d_used["_x"], d_used["_y"])
    xl = C.label(xcol) + ("" if tf == "levels" else f" — {TRANSFORMS[tf].lower()}")
    yl = C.label(ycol) + ("" if tf == "levels" else f" — {TRANSFORMS[tf].lower()}")
    pdf = d.rename(columns={"_x": "x__", "_y": "y__"})
    if show_out:
        fig = LC.scatter(pdf, "x__", "y__", xl, yl, s, trend, _groups(pdf, hl), out)
    else:
        pu = pdf[~out]
        fig = LC.scatter(pu, "x__", "y__", xl, yl, s, trend, _groups(pu, hl), None)
    export = d[["Date", xcol, ycol]].copy()
    if tf != "levels":
        export[f"{xcol} ({tf})"], export[f"{ycol} ({tf})"] = d["_x"], d["_y"]
    export["extreme_value"] = out
    export["included_in_statistics"] = ~out if not show_out else True
    with t[6]:
        csv_button(export, f"lab_{xcol}_vs_{ycol}_{rng[0]}_{rng[1]}.csv", "Data (CSV)", key="eda_dl_csv")
    with t[7]:
        png_button(fig, f"lab_{xcol}_vs_{ycol}.png", "Chart (PNG)", key="eda_dl_png")

    # ---------------------------------------------------------------- 1 · scatter
    st.write("")
    with st.container(key="ds_panel_lab_scatter"):
        h1, h2 = st.columns([1.05, 1.25], vertical_alignment="center")
        unit_y, unit_x = C.unit(ycol) or "units", C.unit(xcol) or "unit"
        with h1:
            design.html(f'<div class="lab-title">{escape(xl)} <em>vs</em> {escape(yl)}</div>'
                        f'<div class="ds-psub">Each point is one day ({escape(level if not market else market.title())}). '
                        f'The trend shows the linear relationship, not causation.</div>')
        with h2:
            neff = f" · effective n ≈ {s['n_eff']:,.0f}" if np.isfinite(s["n_eff"]) else ""
            design.html(
                '<div class="lab-metrics">'
                f'<div class="lab-m"><span>Pearson r</span><b>{_num(s["pearson"])}</b><small>p = {_p(s["pearson_p"])}*</small></div>'
                f'<div class="lab-m"><span>Spearman ρ</span><b>{_num(s["spearman"])}</b><small>p = {_p(s["spearman_p"])}*</small></div>'
                f'<div class="lab-m"><span>Sample size</span><b>{s["n"]:,}</b><small>days{neff}</small></div>'
                f'<div class="lab-m"><span>OLS slope</span><b>{_num(s["slope"], 4)}</b><small>R² = {_num(s["r2"])}</small></div></div>')
        sel = {"Pearson": s["pearson"], "Spearman": s["spearman"], "Kendall": s["kendall"]}[method]
        design.html(
            f'<div class="lab-note" style="margin-top:8px"><b>Slope:</b> {_num(s["slope"], 4)} {escape(unit_y)} per +1 {escape(_singular(unit_x))} '
            f'(95% CI {_num(s["slope_ci"][0], 4)} … {_num(s["slope_ci"][1], 4)}) · {escape(method)} {_num(sel)} · Kendall τ {_num(s["kendall"])}.</div>')
        if not show_out and out.any():
            design.html(f'<div class="lab-warn">{ICON["warning"]}<span>{int(out.sum())} extreme day(s) beyond 3×IQR are hidden and excluded from these '
                        f'statistics; they remain in the CSV with included_in_statistics = False.</span></div>')
        elif out.any():
            design.html(f'<div class="lab-note" style="margin-top:6px">{int(out.sum())} extreme day(s) beyond 3×IQR are shown as open circles and included in the statistics.</div>')
        if "same_day_guests" in (xcol, ycol):
            design.html(f'<div class="lab-warn">{ICON["info"]}<span>Same-Day Guests sums count blank cells as 0, so the series is a lower bound.</span></div>')
        plot(fig, key="eda_scatter")
        design.html(
            f'<div class="lab-note">Correlation is co-movement, not causation: both series follow seasons and long-term growth. Neighbouring days are '
            f'autocorrelated (lag-1 {_num(s["rho_x"], 2)} and {_num(s["rho_y"], 2)}), so the effective sample is about '
            f'{_num(s["n_eff"], 0) if np.isfinite(s["n_eff"]) else "—"} independent days, not {s["n"]:,}, and the p-values (*) are far too optimistic. '
            f'Set “Series” to a day-over-day change to remove shared trends.</div>')

    # ---------------------------------------------------------------- supporting evidence (progressive disclosure)
    ws = w[(w["Date"] >= start) & (w["Date"] <= end)]
    st.write("")
    with st.container(key="lab_more"):
        more = st.tabs(["Correlations", "Lag analysis", "Trends over time", "Distributions"], key="lab_tabnav", on_change="rerun")
        if more[0].open:
            with more[0]:
                with st.container(key="ds_panel_lab_heatmap"):
                    design.html(f'<div class="lab-title">Correlation heatmap ({escape(method)})</div>'
                                '<div class="ds-psub">Every day in the selected range and series; coefficients in the cells, −1 dark grey … +1 green.</div>')
                    hm_default = [x for x in ["Guests", "p2p_7d", "seats_7d", "transfer_7d", "flights_7d", "load_factor_7d", "p2p_share_7d", "guests_lag364_7d", "doy_cos", "is_weekend"] if x in num]
                    hk = f"eda_hm_{C.LEVELS[level]}"
                    with st.popover(f"Variables ({len(st.session_state.get(hk, hm_default))})", icon=":material/tune:"):
                        hm_cols = st.multiselect("Variables", num, default=hm_default, format_func=C.label, key=hk, placeholder="Choose at least two variables")
                    if len(hm_cols) >= 2:
                        hm_frame = pd.DataFrame({c_: C.transform(ws[c_].astype(float), tf) for c_ in hm_cols})
                        cm = C.corr_matrix(hm_frame, hm_cols, method)
                        fig_hm = LC.heatmap(cm, [C.short_label(x) for x in hm_cols])
                        plot(fig_hm, key="eda_hm")
                        with st.container(horizontal=True, gap="small", key="lab_hm_actions"):
                            csv_button(cm.rename(index=C.label, columns=C.label).reset_index(names="variable"), f"correlation_{method.lower()}.csv", "CSV", key="eda_hm_csv")
                            png_button(fig_hm, f"correlation_{method.lower()}.png", "PNG", key="eda_hm_png", width=1000)
                            if st.button("Enlarge", icon=":material/open_in_full:", key="eda_hm_big"):
                                _big_heatmap(cm, [C.label(x) for x in hm_cols], method)
                        design.html('<div class="lab-note">Axis labels are abbreviated (LY = last year, 7d = trailing 7-day window); hover a cell for full names.</div>')
                    else:
                        design.html('<div class="lab-note">Choose at least two variables.</div>')
                st.write("")
                with st.container(key="ds_panel_lab_rank"):
                    design.html('<div class="lab-title">Feature correlation ranking</div>'
                                f'<div class="ds-psub">Every numeric feature against Guests over the selected days ({escape(TRANSFORMS[tf].lower())}), ranked by |{escape(method)}|. '
                                'High correlation between two predictors is flagged, never removed automatically.</div>')
                    if ws["Guests"].notna().sum() >= 10:
                        r1, r2 = st.columns([3, 1.1], vertical_alignment="bottom")
                        thr = r2.slider("Redundancy threshold |r|", 0.80, 0.99, 0.95, 0.01, key="eda_thr")
                        rk_frame = pd.DataFrame({c_: C.transform(ws[c_].astype(float), tf) for c_ in num})
                        rk_frame["Date"] = ws["Date"].to_numpy()
                        rank = C.ranking(rk_frame[rk_frame["Guests"].notna()], num, "Guests", method, C.LEVELS[level], thr)
                        r1.caption(f"{len(rank)} features · {int(rank['redundant'].sum())} flagged as redundant at |r| ≥ {thr:.2f} with another feature.")
                        st.dataframe(rank, hide_index=True, width="stretch", height=340, column_config={
                            "feature": None, "label": st.column_config.TextColumn("Feature", width="large"), "group": st.column_config.TextColumn("Group"),
                            "pearson": st.column_config.NumberColumn("Pearson r", format="%.3f"), "spearman": st.column_config.NumberColumn("Spearman ρ", format="%.3f"),
                            "kendall": st.column_config.NumberColumn("Kendall τ", format="%.3f"), "n": st.column_config.NumberColumn("n days", format="%d"),
                            "missing_pct": st.column_config.NumberColumn("Missing %", format="%.2f"), "whatif": st.column_config.CheckboxColumn("What-if"),
                            "strongest_partner": st.column_config.TextColumn("Most correlated feature"), "partner_r": st.column_config.NumberColumn("r with it", format="%.3f"),
                            "redundant": st.column_config.CheckboxColumn("Redundant")})
                        csv_button(rank, f"feature_ranking_{C.LEVELS[level]}_{method.lower()}.csv", "Download ranking (CSV)", key="eda_rank_csv")
                    else:
                        design.html('<div class="lab-note">Guests are not observed in this date range.</div>')
                    base_days = ws[ws["Guests"].notna()]
                    if len(base_days) > 30 and "flights" in base_days:
                        design.html(_findings_html(C.link_findings(base_days)))
        if more[1].open:
            with more[1]:
                with st.container(key="ds_panel_lab_lag"):
                    design.html('<div class="lab-title">Aviation–hospitality relationship explorer</div>'
                                '<div class="ds-psub">How is aviation activity in the days before a date associated with hotel guests on that date? '
                                'Exploratory associations — not causal estimates or validated scenario impacts.</div>')
                    e = st.columns([1.1, 1, 1, 1.25, 1.1], vertical_alignment="bottom")
                    base = e[0].selectbox("Aviation quantity", list(C.AVIATION_BASE), format_func=C.AVIATION_BASE.get, index=1, key="eda_lag_base")
                    window = e[1].selectbox("Window length", [1, 7, 14, 28], index=1, format_func=lambda v: "Same day" if v == 1 else f"{v}-day sum", key="eda_lag_w")
                    max_lag = e[2].slider("Max lag (days)", 7, 90, 42, 7, key="eda_lag_max")
                    years = ["Whole selected range"] + [str(y) for y in range(max(start.year, 2023), min(end.year, 2025) + 1)]
                    period = e[3].selectbox("Period", years, key="eda_lag_period")
                    compare = e[4].toggle("Compare scopes", value=False, key="eda_lag_cmp", help="International, domestic and overall totals on one chart",
                                          disabled=C.LEVELS[level] == "international_market")
                    p0, p1 = (start, min(end, config.HOTEL_TRAIN_END)) if period.startswith("Whole") else \
                        (max(start, pd.Timestamp(f"{period}-01-01")), min(end, pd.Timestamp(f"{period}-12-31"), config.HOTEL_TRAIN_END))
                    curves = {}
                    if compare and C.LEVELS[level] != "international_market":
                        for lv_ in ("International", "Domestic", "Overall"):
                            curves[lv_] = _lags(lv_, None, base, window, max_lag, method, str(p0.date()), str(p1.date()))
                    else:
                        curves[level if not market else market.title()] = _lags(level, market, base, window, max_lag, method, str(p0.date()), str(p1.date()))
                    valid = {k: v for k, v in curves.items() if v["r"].notna().any()}
                    if valid:
                        plot(LC.lag_plot(valid, method), key="eda_lag")
                        best = []
                        for k, v in valid.items():
                            row = v.loc[v["r"].abs().idxmax()]
                            best.append(f"<b>{escape(k)}</b>: strongest |r| at lag {int(row['lag'])} d (r = {row['r']:.3f}, n = {int(row['n']):,}); "
                                        f"same-day window r = {v.loc[0, 'r']:.3f}")
                        design.html('<div class="lab-note">' + "<br>".join(best) + "</div>")
                        design.html(f'<div class="lab-warn lab-info">{ICON["info"]}<span>Lag k pairs the {escape(C.AVIATION_BASE[base].lower())} '
                                    f'{"on day" if window == 1 else f"summed over {window} days ending on day"} t − k with Guests on day t. Pairs are computed only where '
                                    'the whole window lies on or after 1 Jan 2023 (daily flight data); n per lag is in the tooltip. Both series trend and follow the seasons, '
                                    'so neighbouring lags are highly correlated and the peak is not a travel-time estimate.</span></div>')
                        lag_out = pd.concat([v.assign(series=k) for k, v in valid.items()], ignore_index=True)
                        csv_button(lag_out, f"lag_correlation_{base}_{window}d.csv", "Download lag table (CSV)", key="eda_lag_csv")
                    else:
                        design.html('<div class="lab-note">Not enough observed days in this period for a lag analysis.</div>')
        if more[2].open:
            with more[2]:
                with st.container(key="ds_panel_lab_guests"):
                    design.html('<div class="lab-title">Historical hotel guests</div>'
                                '<div class="ds-psub">Observed Guests from the training files only (1 Jan 2022 → 31 Jul 2025). Smoothed lines are trailing means — '
                                'a summary of the observations, not additional data. The official test period has no Guests and is not shown.</div>')
                    gs = _guests(level, market)
                    g = st.columns([1.3, 0.8, 0.8, 1.6], vertical_alignment="bottom")
                    view = g[0].segmented_control("View", ["Timeline", "Year comparison"], default="Timeline", key="eda_g_view") or "Timeline"
                    s7 = g[1].toggle("7-day mean", value=True, key="eda_g7")
                    s28 = g[2].toggle("28-day mean", value=True, key="eda_g28")
                    gr = g[3].slider("Dates", min_value=config.HOTEL_TRAIN_START.date(), max_value=config.HOTEL_TRAIN_END.date(),
                                     value=(config.HOTEL_TRAIN_START.date(), config.HOTEL_TRAIN_END.date()), format="YYYY-MM-DD", key="eda_g_rng")
                    gsel = gs[(gs["Date"] >= pd.Timestamp(gr[0])) & (gs["Date"] <= pd.Timestamp(gr[1]))]
                    ylab = "Guests per day" + (f" — {market.title()}" if market else "")
                    gfig = LC.guests_ts(gsel, s7, s28, ylab) if view == "Timeline" else LC.year_overlay(gsel, 7 if s7 else (28 if s28 else 1), ylab)
                    plot(gfig, key="eda_gts")
                    missing_g = int(gsel["Guests"].isna().sum())
                    design.html(f'<div class="lab-note">{gsel["Guests"].notna().sum():,} observed days'
                                + (f' · {missing_g} day(s) without a row for this nationality (left as gaps, not filled)' if missing_g else "") + ".</div>")
                    with st.container(horizontal=True, gap="small"):
                        csv_button(gsel, f"guests_{C.LEVELS[level]}{'_' + market if market else ''}.csv", "Download data (CSV)", key="eda_g_csv")
                        png_button(gfig, "historical_guests.png", "Download chart (PNG)", key="eda_g_png")
                st.write("")
                with st.container(key="ds_panel_lab_av"):
                    who = f"flights departing {market.title()}" if market else "all arrivals into AUH"
                    design.html('<div class="lab-title">Aviation capacity & passenger trends</div>'
                                f'<div class="ds-psub">Daily flight data ({escape(who)}), 1 Jan 2023 → 28 Feb 2026. Windowed series are trailing sums on a complete '
                                'calendar; the first days of a window are left empty instead of showing partial sums.</div>')
                    v = st.columns([1.9, 0.95, 1.0, 1.6], vertical_alignment="bottom")
                    series = v[0].pills("Series", ["seats", "p2p", "transfer"], selection_mode="multi", default=["seats", "p2p", "transfer"],
                                        format_func={"seats": "Total Seats", "p2p": "Total P2P", "transfer": "Total Transfer"}.get, key="eda_av_series") or ["seats"]
                    win = v[1].selectbox("Window", [1, 7, 28], index=1, format_func=lambda x: "Daily" if x == 1 else f"{x}-day sum", key="eda_av_w")
                    norm = v[2].toggle("Normalise (index)", value=False, key="eda_av_norm", help="Each series divided by its own mean over the first 28 shown days × 100.")
                    ar = v[3].slider("Dates", min_value=config.FLIGHT_DAILY_START.date(), max_value=config.FLIGHT_END.date(),
                                     value=(config.FLIGHT_DAILY_START.date(), config.FLIGHT_END.date()), format="YYYY-MM-DD", key="eda_av_rng")
                    av = C.aviation_series(df, series, win)
                    av = av[(av["Date"] >= pd.Timestamp(ar[0])) & (av["Date"] <= pd.Timestamp(ar[1]))].reset_index(drop=True)
                    labels = {"seats": "Total Seats", "p2p": "Total P2P", "transfer": "Total Transfer"}
                    unit_lab = ("seats / passengers per day" if win == 1 else f"seats / passengers, {win}-day trailing sum")
                    if norm:
                        for c_ in series:
                            ref = av[c_].dropna().head(28).mean()
                            av[c_] = av[c_] / ref * 100 if ref else np.nan
                        unit_lab = "Index (mean of first 28 shown days = 100)"
                    afig = LC.aviation_ts(av, {c_: labels[c_] for c_ in series}, unit_lab, pd.Timestamp("2026-02-28") if win == 1 else None)
                    plot(afig, key="eda_ats")
                    design.html('<div class="lab-note">The flight file’s last day (28 Feb 2026) lists 9 airlines instead of about 20 and roughly half the usual '
                                'seats — probably an incomplete final day. It is shown as recorded, not removed; with windowed sums it lowers the final point.'
                                + (" Normalised values are an index, not counts." if norm else "") + "</div>")
                    with st.container(horizontal=True, gap="small"):
                        csv_button(av, f"aviation_{'_'.join(series)}_{win}d.csv", "Download data (CSV)", key="eda_av_csv")
                        png_button(afig, "aviation_trends.png", "Download chart (PNG)", key="eda_av_png")
                st.write("")
                with st.expander("Monthly history 2022 → 2026 — the only grain at which 2022 flights are comparable", icon=":material/calendar_month:"):
                    mh = _monthly(C.LEVELS[level])
                    m1, m2 = st.columns(2)
                    with m1:
                        design.html('<div class="lab-note"><b>Total P2P arrivals per month</b> — grey: 2022 monthly rows; green: daily rows summed (all markets).</div>')
                        plot(LC.monthly_bars(mh, "Total P2P", "Total P2P per month"), key="eda_mp")
                    with m2:
                        design.html(f'<div class="lab-note"><b>Guest-days per month</b> (sum of daily Guests, {"international" if "intl" in C.LEVELS[level] or C.LEVELS[level] == "international_market" else C.LEVELS[level]}); '
                                    'test months have no observed Guests.</div>')
                        plot(LC.monthly_bars(mh.dropna(subset=["Guest-days"]), "Guest-days", "Guest-days per month"), key="eda_mg")
                    csv_button(mh, "monthly_history.csv", key="eda_mh_csv")
        if more[3].open:
            with more[3]:
                with st.container(key="ds_panel_lab_dist"):
                    design.html('<div class="lab-title">Feature distributions</div>'
                                '<div class="ds-psub">Same days as the scatter (outlier setting applied).</div>')
                    d1, d2 = st.columns(2)
                    v1 = d1.selectbox("Variable A", num, index=num.index(xcol), format_func=C.label, key="eda_dist_a", label_visibility="collapsed")
                    v2 = d2.selectbox("Variable B", num, index=num.index(ycol) if ycol in num else 0, format_func=C.label, key="eda_dist_b", label_visibility="collapsed")
                    d3, d4 = st.columns([1, 1.3], vertical_alignment="center")
                    kind = d3.segmented_control("Chart", ["Histogram", "Box plot"], default="Histogram", key="eda_dist_kind", label_visibility="collapsed") or "Histogram"
                    bins = d4.slider("Bins", 10, 80, 30, 5, key="eda_bins", disabled=kind != "Histogram", label_visibility="collapsed")
                    design.html('<div class="lab-note">*outliers: values beyond 3×IQR from the quartiles (flagged, never deleted).</div>')
                    cols_ = st.columns(2, gap="small")
                    for colw, v_, col in ((cols_[0], v1, LC.SERIES[0]), (cols_[1], v2, LC.SERIES[1])):
                        vals = C.transform(w[v_].astype(float), tf).loc[d_used.index]
                        n_out = int(C.outlier_mask(pd.DataFrame({"v": vals}), ["v"]).sum())
                        with colw:
                            design.html(f'<div class="lab-note" style="margin-top:4px"><b>{escape(C.label(v_))}</b></div>' + _stats_html(vals, len(d_used), n_out))
                            plot(LC.distribution(vals, C.short_label(v_), kind, bins, col), key=f"eda_dist_{v_}_{col}")
    st.write("")
    with st.expander("How to read these results", icon=":material/menu_book:", key="lab_x_notes"):
        design.html(
            '<div class="lab-sci">'
            '<p><b>Grain.</b> One point = one calendar day. International totals sum the nationality rows that exist each day; a source market pairs one '
            'nationality’s Guests with flights from the country of the same name — a proxy, not passenger-to-guest attribution.</p>'
            '<p><b>Autocorrelation.</b> Consecutive days are not independent, so classical p-values and confidence bands overstate certainty. The effective '
            'sample size shown with each scatter corrects for this approximately.</p>'
            '<p><b>Shared trends.</b> Aviation and hotel demand both grew and both follow the seasons. Differenced series (day-over-day or week-over-week) '
            'remove most of that shared drift; a correlation that vanishes after differencing was driven by trend or season.</p>'
            '<p><b>Not causal.</b> Correlations and lagged correlations describe co-movement in historical data. They are neither causal effects nor the '
            'What-if Simulator’s scenario impacts, which come from trained models.</p>'
            '<p><b>Unlabelled test period.</b> Guests are not observed after 31 Jul 2025; no chart on this page shows or imputes them.</p>'
            '<p><b>2022 flights.</b> Monthly 2022 flight rows are never spread over days; they appear only in the monthly history.</p>'
            '</div>')
    next_step("eda")



@st.dialog("Correlation heatmap — enlarged", width="large")
def _big_heatmap(cm: pd.DataFrame, labels: list[str], method: str) -> None:
    st.caption(f"{method} correlation · {len(labels)} variables")
    plot(LC.heatmap(cm, labels, height=max(520, 42 * len(labels) + 160)), key="eda_hm_dialog")
