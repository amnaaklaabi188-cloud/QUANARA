"""Plotly figures for the Correlation & Visualization Lab (shared white-card look via design.style_fig).

Colour rules (validated reference palette): at most three categorical hues on a scatter
(blue · orange · aqua, fixed order); diverging blue (+) / grey (0) / red (−) for correlations;
text in ink colours, never in series colours.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..correlation import mean_ci_band
from .design import INK, INK2, MUTED, style_fig

SERIES = ["#059669", "#1f2937", "#9ca3af"]
LINE4 = ["#d1d5d3", "#9ca3af", "#1f2937", "#059669"]  # year overlays: oldest year muted
DIVERGING = [[0.0, "#44403c"], [0.25, "#a8a29e"], [0.5, "#f5f5f4"], [0.75, "#6ee7b7"], [1.0, "#047857"]]
NAVY = "#1f2937"


def _fmt(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.3f}" if abs(v) < 10 else f"{v:,.1f}"


def scatter(d: pd.DataFrame, x: str, y: str, xl: str, yl: str, s: dict, trend: bool, groups: pd.Series | None,
            outliers: pd.Series | None) -> go.Figure:
    fig = go.Figure()
    base = d if outliers is None else d[~outliers]
    date = base["Date"].dt.strftime("%Y-%m-%d")
    hover = f"%{{customdata}}<br>{xl}: %{{x:,.4~f}}<br>{yl}: %{{y:,.4~f}}<extra>%{{fullData.name}}</extra>"
    if groups is None:
        fig.add_trace(go.Scattergl(x=base[x], y=base[y], mode="markers", name="Daily observation", customdata=date,
                                   marker=dict(size=6.5, color=SERIES[0], opacity=0.6, line=dict(width=0.8, color="#ffffff")), hovertemplate=hover))
    else:
        g = groups.loc[base.index]
        for i, cat in enumerate(sorted(g.dropna().unique(), key=str)):
            m = g == cat
            fig.add_trace(go.Scattergl(x=base.loc[m, x], y=base.loc[m, y], mode="markers", name=str(cat), customdata=date[m],
                                       marker=dict(size=6.5, color=SERIES[i % 3], opacity=0.62, line=dict(width=0.8, color="#ffffff")), hovertemplate=hover))
    if outliers is not None and outliers.any():
        o = d[outliers]
        fig.add_trace(go.Scattergl(x=o[x], y=o[y], mode="markers", name="Extreme value (beyond 3×IQR)", customdata=o["Date"].dt.strftime("%Y-%m-%d"),
                                   marker=dict(size=10, symbol="circle-open", color=NAVY, line=dict(width=1.6)), hovertemplate=hover))
    if trend and np.isfinite(s.get("slope", np.nan)):
        xs = np.linspace(float(base[x].min()), float(base[x].max()), 80)
        lo, hi = mean_ci_band(s, xs)
        col = SERIES[1] if groups is None else NAVY
        fig.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", fillcolor="rgba(31,41,55,0.13)" if groups is None else "rgba(17,24,39,0.10)",
                                 line=dict(width=0), hoverinfo="skip", name="95% CI of the fitted mean", showlegend=True))
        fig.add_trace(go.Scatter(x=xs, y=s["intercept"] + s["slope"] * xs, mode="lines", name="OLS trendline", line=dict(color=col, width=2.2),
                                 hovertemplate=f"{xl}: %{{x:,.4~f}}<br>fitted {yl}: %{{y:,.4~f}}<extra>OLS</extra>"))
    style_fig(fig, height=440)
    fig.update_layout(legend=dict(orientation="h", y=1.04, x=1, xanchor="right"), hovermode="closest", margin=dict(l=8, r=12, t=30, b=8))
    fig.update_xaxes(title_text=xl, separatethousands=True)
    fig.update_yaxes(title_text=yl, separatethousands=True)
    return fig


def heatmap(cm: pd.DataFrame, labels: list[str], height: int | None = None) -> go.Figure:
    z = cm.to_numpy()
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, colorscale=DIVERGING, zmin=-1, zmax=1, xgap=2, ygap=2,
                               text=np.where(np.isfinite(z), np.round(z, 2), np.nan), texttemplate="%{text}" if len(labels) <= 16 else None,
                               textfont=dict(size=11, color=INK), hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
                               colorbar=dict(thickness=10, len=0.9, tickvals=[-1, -0.5, 0, 0.5, 1], outlinewidth=0)))
    style_fig(fig, height=height or max(360, 30 * len(labels) + 140), legend=False)
    fig.update_xaxes(tickangle=-38, showgrid=False, tickfont=dict(size=11))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=11))
    return fig


def distribution(v: pd.Series, lab: str, kind: str, bins: int, color: str, outliers: pd.Series | None = None) -> go.Figure:
    v = pd.to_numeric(v, errors="coerce").dropna()
    if kind == "Box plot":
        fig = go.Figure(go.Box(x=v, name=lab, marker=dict(color=color, size=4), line=dict(color=color), boxpoints="outliers",
                               fillcolor="rgba(5,150,105,0.12)" if color == SERIES[0] else "rgba(31,41,55,0.12)",
                               hovertemplate=f"{lab}: %{{x:,.4~f}}<extra></extra>"))
        style_fig(fig, height=230, legend=False)
        fig.update_yaxes(showticklabels=False)
    else:
        fig = go.Figure(go.Histogram(x=v, nbinsx=bins, marker=dict(color=color, line=dict(color="#ffffff", width=1)),
                                     hovertemplate=f"{lab}: %{{x}}<br>days: %{{y}}<extra></extra>"))
        style_fig(fig, height=230, legend=False)
        fig.update_yaxes(title_text="Days")
    fig.update_xaxes(title_text=lab, separatethousands=True)
    fig.update_layout(bargap=0.04, margin=dict(l=8, r=8, t=8, b=8))
    return fig


def lag_plot(curves: dict[str, pd.DataFrame], method: str) -> go.Figure:
    fig = go.Figure()
    for i, (name, c) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(x=c["lag"], y=c["r"], mode="lines+markers", name=name, line=dict(color=SERIES[i % 3], width=2.2),
                                 marker=dict(size=6, line=dict(width=1, color="#ffffff")), customdata=c["n"],
                                 hovertemplate="lag %{x} days<br>r = %{y:.3f}<br>n = %{customdata:,} days<extra>" + name + "</extra>"))
        if c["r"].notna().any():
            k = c.loc[c["r"].abs().idxmax()]
            fig.add_trace(go.Scatter(x=[k["lag"]], y=[k["r"]], mode="markers", showlegend=False, hoverinfo="skip",
                                     marker=dict(size=13, symbol="circle-open", color=SERIES[i % 3], line=dict(width=2))))
    fig.add_hline(y=0, line=dict(color="#cfd1d0", width=1))
    style_fig(fig, height=330)
    fig.update_xaxes(title_text="Lag k (days): aviation window ends k days before the guest date")
    fig.update_yaxes(title_text=f"{method} correlation", range=[-1, 1])
    fig.update_layout(hovermode="x unified")
    return fig


def guests_ts(gs: pd.DataFrame, show7: bool, show28: bool, ylab: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gs["Date"], y=gs["Guests"], mode="lines", name="Observed daily Guests", line=dict(color="#b0b4b2", width=1),
                             hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra>observed</extra>"))
    if show7:
        fig.add_trace(go.Scatter(x=gs["Date"], y=gs["mean_7d"], mode="lines", name="7-day trailing mean", line=dict(color=SERIES[0], width=2),
                                 hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra>7-day mean</extra>"))
    if show28:
        fig.add_trace(go.Scatter(x=gs["Date"], y=gs["mean_28d"], mode="lines", name="28-day trailing mean", line=dict(color=SERIES[1], width=2.2),
                                 hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra>28-day mean</extra>"))
    style_fig(fig, height=340)
    fig.update_yaxes(title_text=ylab, separatethousands=True)
    fig.update_layout(hovermode="x unified")
    return fig


def year_overlay(gs: pd.DataFrame, smooth: int, ylab: str) -> go.Figure:
    g = gs.copy()
    g["year"], g["doy"] = g["Date"].dt.year, g["Date"].dt.dayofyear
    val = g["Guests"].rolling(smooth, min_periods=smooth).mean() if smooth > 1 else g["Guests"]
    g["v"] = val
    fig = go.Figure()
    years = sorted(g["year"].unique())
    for i, yr in enumerate(years):
        s = g[g["year"] == yr]
        col = LINE4[max(0, len(LINE4) - len(years) + i)]
        fig.add_trace(go.Scatter(x=s["doy"], y=s["v"], mode="lines", name=str(yr), line=dict(color=col, width=2 if yr == years[-1] else 1.6),
                                 customdata=s["Date"].dt.strftime("%Y-%m-%d"), hovertemplate="%{customdata}: %{y:,.0f}<extra>" + str(yr) + "</extra>"))
    style_fig(fig, height=340)
    fig.update_xaxes(title_text="Day of year", tickvals=[1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
                     ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    fig.update_yaxes(title_text=ylab + (f" ({smooth}-day trailing mean)" if smooth > 1 else ""), separatethousands=True)
    fig.update_layout(hovermode="x unified")
    return fig


def aviation_ts(s: pd.DataFrame, cols: dict[str, str], ylab: str, anomaly: pd.Timestamp | None) -> go.Figure:
    fig = go.Figure()
    for i, (c, lab) in enumerate(cols.items()):
        fig.add_trace(go.Scatter(x=s["Date"], y=s[c], mode="lines", name=lab, line=dict(color=SERIES[i % 3], width=2),
                                 hovertemplate="%{x|%Y-%m-%d}: %{y:,.4~f}<extra>" + lab + "</extra>"))
    if anomaly is not None and anomaly in set(s["Date"]):
        fig.add_vline(x=anomaly, line=dict(color=MUTED, width=1, dash="dot"))
        fig.add_annotation(x=anomaly, y=1, yref="paper", text="28 Feb 2026: 9 airlines, ~½ usual seats", showarrow=False, xanchor="right",
                           font=dict(size=11, color=INK2), bgcolor="rgba(255,255,255,.85)")
    style_fig(fig, height=340)
    fig.update_yaxes(title_text=ylab, separatethousands=True)
    fig.update_layout(hovermode="x unified")
    return fig


def monthly_bars(df: pd.DataFrame, col: str, lab: str) -> go.Figure:
    colors = np.where(df["flight_grain"] == "monthly rows", "#b0b4b2", SERIES[0]) if col != "Guest-days" else SERIES[0]
    fig = go.Figure(go.Bar(x=df["month"], y=df[col], marker=dict(color=colors, cornerradius=4), name=lab,
                           hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra></extra>"))
    style_fig(fig, height=280, legend=False)
    fig.update_yaxes(title_text=lab, separatethousands=True)
    fig.update_layout(bargap=0.25)
    return fig
