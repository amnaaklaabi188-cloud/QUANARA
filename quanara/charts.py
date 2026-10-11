"""Plotly figure builders. Palette = the reference data-viz palette (light mode):
categorical slots 1–3 (validated all-pairs), blue↔red diverging with a gray midpoint,
hairline grids, 2px lines, ≥8px markers, one y-axis per chart."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#059669", "#1f2937", "#9ca3af"   # blue, orange, aqua
NEUTRAL = "#b9b8b2"
DIVERGING = [[0.0, "#44403c"], [0.5, "#f5f5f4"], [1.0, "#059669"]]  # red (−) → gray → blue (+)
SEQUENTIAL = [[0, "#d1fae5"], [0.5, "#10b981"], [1, "#064e3b"]]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _layout(fig: go.Figure, title: str | None = None, height: int = 380, xtitle: str | None = None, ytitle: str | None = None,
            legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=INK)) if title else None,
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(family=FONT, size=12, color=INK2),
        margin=dict(l=10, r=10, t=48 if title else 16, b=10), hovermode="closest",
        hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(family=FONT, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(color=INK2)),
        showlegend=legend,
    )
    fig.update_xaxes(title_text=xtitle, gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED), title_font=dict(color=INK2), automargin=True)
    fig.update_yaxes(title_text=ytitle, gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED), title_font=dict(color=INK2),
                     separatethousands=True, automargin=True)
    return fig


def fmt(v: float, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.1%}" if pct else f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.3f}"


# ------------------------------------------------------------------ EDA

def scatter_trend(df: pd.DataFrame, x: str, y: str, xlab: str, ylab: str, stats: dict, trend: bool = True,
                  color_by_split: bool = False) -> go.Figure:
    fig = go.Figure()
    d = df[[x, y, "Date"]].dropna()
    fig.add_trace(go.Scattergl(x=d[x], y=d[y], mode="markers", name="Daily observation",
                               marker=dict(size=7, color=S1, opacity=0.55, line=dict(width=1, color=SURFACE)),
                               customdata=d["Date"].dt.strftime("%Y-%m-%d"),
                               hovertemplate=f"%{{customdata}}<br>{xlab}: %{{x:,.2f}}<br>{ylab}: %{{y:,.0f}}<extra></extra>"))
    if trend and np.isfinite(stats.get("slope", np.nan)):
        xs = np.linspace(d[x].min(), d[x].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=stats["intercept"] + stats["slope"] * xs, mode="lines", name="OLS trendline",
                                 line=dict(color=S2, width=2), hovertemplate="trend: %{y:,.0f}<extra></extra>"))
    return _layout(fig, None, 420, xlab, ylab, legend=trend)


def heatmap(corr: pd.DataFrame, labels: list[str]) -> go.Figure:
    z = corr.to_numpy()
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=labels, colorscale=DIVERGING, zmin=-1, zmax=1, xgap=2, ygap=2,
                               text=np.round(z, 2), texttemplate="%{text}" if len(labels) <= 14 else None,
                               hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>", colorbar=dict(title="r", thickness=12)))
    fig = _layout(fig, None, max(380, 34 * len(labels) + 120), legend=False)
    fig.update_xaxes(tickangle=-40, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def histogram(s: pd.Series, lab: str, bins: int = 40) -> go.Figure:
    fig = go.Figure(go.Histogram(x=s.dropna(), nbinsx=bins, marker=dict(color=S1, line=dict(color=SURFACE, width=2)),
                                 hovertemplate=f"{lab}: %{{x}}<br>days: %{{y}}<extra></extra>"))
    return _layout(fig, None, 300, lab, "Days", legend=False)


def guests_timeseries(df: pd.DataFrame, ycol: str = "Guests", title: str | None = None) -> go.Figure:
    d = df[["Date", ycol]].dropna().sort_values("Date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=d[ycol], mode="lines", name="Daily", line=dict(color=NEUTRAL, width=1),
                             hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=d["Date"], y=d[ycol].rolling(28, min_periods=14).mean(), mode="lines", name="28-day mean",
                             line=dict(color=S1, width=2), hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra>28-day mean</extra>"))
    fig = _layout(fig, title, 360, None, "Guests in-house per day")
    fig.update_layout(hovermode="x unified")
    return fig


def aviation_timeseries(df: pd.DataFrame, cols: dict[str, str], title: str | None = None, smooth: int = 7) -> go.Figure:
    colors = [S1, S2, S3]
    fig = go.Figure()
    for (c, lab), col in zip(cols.items(), colors):
        y = df[c].rolling(smooth, min_periods=1).sum() if smooth > 1 else df[c]
        fig.add_trace(go.Scatter(x=df["Date"], y=y, mode="lines", name=lab, line=dict(color=col, width=2),
                                 hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra>" + lab + "</extra>"))
    fig = _layout(fig, title, 360, None, f"Count ({smooth}-day trailing sum)" if smooth > 1 else "Count per day")
    fig.update_layout(hovermode="x unified")
    return fig


def monthly_bars(df: pd.DataFrame, col: str, lab: str) -> go.Figure:
    colors = np.where(df["flight_grain"] == "monthly rows", NEUTRAL, S1) if "flight_grain" in df and col not in ("Guest-nights", "Guest-days") else S1
    fig = go.Figure(go.Bar(x=df["month"], y=df[col], marker=dict(color=colors, cornerradius=4), name=lab,
                           hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra></extra>"))
    fig = _layout(fig, None, 300, None, lab, legend=False)
    fig.update_layout(bargap=0.25)
    return fig


def missingness_bar(miss: pd.DataFrame) -> go.Figure:
    m = miss.sort_values("missing_pct")
    fig = go.Figure(go.Bar(y=m["column"], x=m["missing_pct"], orientation="h", marker=dict(color=S1, cornerradius=4),
                           text=[f"{v:.1f}%" for v in m["missing_pct"]], textposition="outside", cliponaxis=False, textfont=dict(color=INK2),
                           hovertemplate="%{y}: %{x:.2f}% missing<extra></extra>"))
    fig = _layout(fig, None, max(260, 26 * len(m) + 60), "% of rows missing / suppressed", None, legend=False)
    fig.update_xaxes(range=[0, max(5, float(m["missing_pct"].max()) * 1.18)])
    fig.update_layout(bargap=0.35)
    return fig


def coverage_timeline(spans: list[dict]) -> go.Figure:
    color = {"hotel (target observed)": S1, "hotel (no target)": NEUTRAL, "flights daily": S3, "flights monthly": "#1bcb8a",
             "modelling window": S2}
    fig = go.Figure()
    for s in spans:
        span_ms = ((pd.Timestamp(s["end"]) - pd.Timestamp(s["start"])).days + 1) * 86_400_000  # bar length on a date axis is in ms
        fig.add_trace(go.Bar(y=[s["dataset"]], x=[span_ms], base=[pd.Timestamp(s["start"]).isoformat()],
                             orientation="h", marker=dict(color=color[s["kind"]], cornerradius=4), name=s["kind"], showlegend=False,
                             hovertemplate=f"{s['dataset']}<br>{s['start']} → {s['end']}<br>{s['kind']}<extra></extra>"))
    fig = _layout(fig, None, 330, None, None, legend=False)
    fig.update_xaxes(type="date")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(bargap=0.45, barmode="overlay")
    return fig


# ------------------------------------------------------------------ evaluation

def actual_vs_pred(pred: pd.DataFrame, title: str | None = None) -> go.Figure:
    lo = float(min(pred["y_true"].min(), pred["y_pred"].min()))
    hi = float(max(pred["y_true"].max(), pred["y_pred"].max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x (perfect)", line=dict(color=MUTED, width=1.5, dash="dot")))
    cd = pred["Date"].dt.strftime("%Y-%m-%d") + (" · " + pred["market_key"].str.title() if "market_key" in pred else "")
    fig.add_trace(go.Scattergl(x=pred["y_true"], y=pred["y_pred"], mode="markers", name="Holdout prediction",
                               marker=dict(size=7, color=S1, opacity=0.55, line=dict(width=1, color=SURFACE)), customdata=cd,
                               hovertemplate="%{customdata}<br>actual %{x:,.0f}<br>predicted %{y:,.0f}<extra></extra>"))
    fig = _layout(fig, title, 400, "Actual Guests", "Predicted Guests")
    return fig


def actual_vs_pred_ts(daily: pd.DataFrame, title: str | None = None, extra: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["Date"], y=daily["y_true"], mode="lines", name="Actual", line=dict(color=S1, width=2),
                             hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra>actual</extra>"))
    fig.add_trace(go.Scatter(x=daily["Date"], y=daily["y_pred"], mode="lines", name="Predicted", line=dict(color=S2, width=2),
                             hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra>predicted</extra>"))
    if extra is not None and len(extra):
        fig.add_trace(go.Scatter(x=extra["Date"], y=extra["y_pred"], mode="lines", name="Forecast (test dates, no actuals)",
                                 line=dict(color=S2, width=2, dash="dot"), hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra>forecast</extra>"))
    fig = _layout(fig, title, 380, None, "Guests per day")
    fig.update_layout(hovermode="x unified")
    return fig


def residuals_vs_pred(pred: pd.DataFrame) -> go.Figure:
    r = pred["y_true"] - pred["y_pred"]
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=MUTED, width=1.5, dash="dot"))
    fig.add_trace(go.Scattergl(x=pred["y_pred"], y=r, mode="markers", name="Residual", marker=dict(size=7, color=S1, opacity=0.55, line=dict(width=1, color=SURFACE)),
                               hovertemplate="predicted %{x:,.0f}<br>residual %{y:,.0f}<extra></extra>"))
    return _layout(fig, None, 340, "Predicted Guests", "Residual (actual − predicted)", legend=False)


def residual_hist(pred: pd.DataFrame) -> go.Figure:
    r = pred["y_true"] - pred["y_pred"]
    fig = go.Figure(go.Histogram(x=r, nbinsx=40, marker=dict(color=S1, line=dict(color=SURFACE, width=2)),
                                 hovertemplate="residual %{x}<br>count %{y}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1.5, dash="dot"))
    return _layout(fig, None, 340, "Residual (actual − predicted)", "Count", legend=False)


def metric_bars(df: pd.DataFrame, metric: str, label_col: str = "algorithm", highlight: str | None = None, pct: bool = False) -> go.Figure:
    d = df.sort_values(metric, ascending=metric not in ("R2", "Pearson", "Spearman"))
    colors = [S2 if (highlight and r == highlight) else S1 for r in d["model_id"]] if "model_id" in d else S1
    fig = go.Figure(go.Bar(y=d[label_col], x=d[metric], orientation="h", marker=dict(color=colors, cornerradius=4),
                           text=[(f"{v:.1%}" if pct else f"{v:,.3f}" if abs(v) < 10 else f"{v:,.0f}") for v in d[metric]],
                           textposition="outside", cliponaxis=False, textfont=dict(color=INK2), hovertemplate="%{y}: %{x:,.4f}<extra></extra>"))
    fig = _layout(fig, None, max(240, 40 * len(d) + 60), metric, None, legend=False)
    xmax = float(np.nanmax(d[metric])) if len(d) else 1
    xmin = float(np.nanmin(d[metric])) if len(d) else 0
    fig.update_xaxes(range=[min(0, xmin * 1.15), xmax * 1.22 if xmax > 0 else 1])
    fig.update_layout(bargap=0.35)
    return fig


def importance_bar(imp: pd.DataFrame, col: str) -> go.Figure:
    d = imp.dropna(subset=[col]).sort_values(col).tail(20)
    fig = go.Figure(go.Bar(y=d["label"], x=d[col], orientation="h", marker=dict(color=S1, cornerradius=4),
                           hovertemplate="%{y}: %{x:,.3f}<extra></extra>"))
    fig = _layout(fig, None, max(260, 26 * len(d) + 60), col.replace("_", " "), None, legend=False)
    fig.update_layout(bargap=0.35)
    return fig


# ------------------------------------------------------------------ simulator

def scenario_daily(daily: pd.DataFrame, observed: bool = True) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["Date"], y=daily["baseline"], mode="lines+markers", name="Baseline prediction",
                             line=dict(color=S1, width=2), marker=dict(size=8, line=dict(width=2, color=SURFACE)),
                             hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra>baseline</extra>"))
    fig.add_trace(go.Scatter(x=daily["Date"], y=daily["scenario"], mode="lines+markers", name="Scenario prediction",
                             line=dict(color=S2, width=2), marker=dict(size=8, line=dict(width=2, color=SURFACE)),
                             hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra>scenario</extra>"))
    if observed and "observed" in daily and daily["observed"].notna().any():
        fig.add_trace(go.Scatter(x=daily["Date"], y=daily["observed"], mode="lines", name="Observed (training data)",
                                 line=dict(color=MUTED, width=1.5, dash="dot"), hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra>observed</extra>"))
    fig = _layout(fig, None, 380, None, "Predicted guests in-house per day")
    fig.update_layout(hovermode="x unified")
    return fig


def baseline_vs_scenario_bars(b: float, s: float) -> go.Figure:
    fig = go.Figure(go.Bar(x=["Baseline", "Scenario"], y=[b, s], marker=dict(color=[S1, S2], cornerradius=4), width=0.4,
                           text=[f"{b:,.0f}", f"{s:,.0f}"], textposition="outside", cliponaxis=False, textfont=dict(color=INK),
                           hovertemplate="%{x}: %{y:,.0f} guests/day<extra></extra>"))
    fig = _layout(fig, None, 320, None, "Average predicted guests per day", legend=False)
    fig.update_yaxes(range=[0, max(b, s) * 1.18])
    return fig


def market_contrib_bar(mc: pd.DataFrame, highlight: str | None, top: int = 15) -> go.Figure:
    d = mc.copy()
    d["abs"] = d["change"].abs()
    d = d.sort_values("abs", ascending=False).head(top).sort_values("change")
    colors = [S2 if m == highlight else NEUTRAL for m in d["market_key"]]
    fig = go.Figure(go.Bar(y=d["market_key"].str.title(), x=d["change"], orientation="h", marker=dict(color=colors, cornerradius=4),
                           hovertemplate="%{y}: %{x:+,.0f} guest-days<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    fig = _layout(fig, None, max(260, 26 * len(d) + 60), "Change in guest-days (scenario − baseline)", None, legend=False)
    fig.update_layout(bargap=0.35)
    return fig


def comparison_bars(cmp: pd.DataFrame, active: str) -> go.Figure:
    d = cmp.sort_values("pct_change")
    colors = [S2 if m == active else S1 for m in d["model_id"]]
    fig = go.Figure(go.Bar(y=d["algorithm"], x=d["pct_change"], orientation="h", marker=dict(color=colors, cornerradius=4),
                           text=[f"{v:+.2f}%" for v in d["pct_change"]], textposition="outside", cliponaxis=False, textfont=dict(color=INK2),
                           hovertemplate="%{y}: %{x:+.2f}%<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    lim = max(1.0, float(d["pct_change"].abs().max()) * 1.7)  # headroom so outside labels are never clipped
    fig = _layout(fig, None, max(240, 44 * len(d) + 60), "Predicted change in guests (%)", None, legend=False)
    fig.update_xaxes(range=[-lim, lim])
    fig.update_layout(bargap=0.35)
    return fig
