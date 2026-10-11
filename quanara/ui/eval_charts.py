"""Evaluation figures shared by the ML Training Studio and Model Evaluation.

Conventions: residual = actual − predicted (positive = under-prediction); actual in charcoal, predicted in
emerald; one y-axis per chart; equal scales on actual-vs-predicted so distance from y = x is honest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .design import INK2, MUTED, style_fig

BLUE, ORANGE, AQUA, NAVY = "#059669", "#1f2937", "#9ca3af", "#1f2937"  # emerald (model / positive), charcoal (observed / negative), grey, charcoal
SOFT = "#b9cbc2"  # non-highlighted bars
SEQ = [[0, "#d1fae5"], [0.5, "#10b981"], [1, "#064e3b"]]


def _cd(pred: pd.DataFrame) -> pd.Series:
    cd = pred["Date"].dt.strftime("%Y-%m-%d")
    if "market_key" in pred:
        cd = cd + " · " + pred["market_key"].astype(str).str.title()
    return cd


def actual_vs_pred(pred: pd.DataFrame, color_residual: bool = True, top_n: int = 10, height: int = 420) -> go.Figure:
    lo = float(min(pred["y_true"].min(), pred["y_pred"].min()))
    hi = float(max(pred["y_true"].max(), pred["y_pred"].max()))
    pad = (hi - lo) * 0.04
    lo, hi = lo - pad, hi + pad
    r = pred["y_true"] - pred["y_pred"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x (perfect)", line=dict(color=MUTED, width=1.4, dash="dash"), hoverinfo="skip"))
    marker = dict(size=7, opacity=0.7, line=dict(width=0.8, color="#ffffff"))
    if color_residual:
        marker.update(color=r.abs(), colorscale=SEQ, colorbar=dict(title="|residual|", thickness=10, len=0.75, outlinewidth=0))
    else:
        marker.update(color=BLUE)
    fig.add_trace(go.Scattergl(x=pred["y_true"], y=pred["y_pred"], mode="markers", name="Holdout days", marker=marker,
                               customdata=np.c_[_cd(pred), r], hovertemplate="%{customdata[0]}<br>actual %{x:,.0f}<br>predicted %{y:,.0f}"
                               "<br>residual %{customdata[1]:+,.0f}<extra></extra>"))
    if top_n:
        t = pred.assign(_a=r.abs()).nlargest(top_n, "_a")
        fig.add_trace(go.Scatter(x=t["y_true"], y=t["y_pred"], mode="markers", name=f"{top_n} largest errors",
                                 marker=dict(size=12, symbol="circle-open", color=ORANGE, line=dict(width=2)), hoverinfo="skip"))
    style_fig(fig, height=height)
    fig.update_xaxes(title_text="Actual Guests", range=[lo, hi], separatethousands=True)
    fig.update_yaxes(title_text="Predicted Guests", range=[lo, hi], scaleanchor="x", scaleratio=1, separatethousands=True)
    fig.update_layout(legend=dict(orientation="h", y=1.04, x=0))
    return fig


def over_time(daily: pd.DataFrame, extra: pd.DataFrame | None = None, roll: int = 0, top_n: int = 5, height: int = 400) -> go.Figure:
    d = daily.sort_values("Date")
    fig = go.Figure()
    # shaded gap between actual and predicted: blue fill where the model under-predicts, orange where it over-predicts
    fig.add_trace(go.Scatter(x=d["Date"], y=d["y_true"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["Date"], y=np.minimum(d["y_true"], d["y_pred"]), mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(5,150,105,0.14)", name="Under-predicted", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["y_true"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["Date"], y=np.maximum(d["y_true"], d["y_pred"]), mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(31,41,55,0.14)", name="Over-predicted", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["y_true"], mode="lines", name="Actual Guests", line=dict(color=ORANGE, width=2.2),
                             hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra>actual</extra>"))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["y_pred"], mode="lines", name="Predicted Guests", line=dict(color=BLUE, width=2.2),
                             hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra>predicted</extra>"))
    if roll and roll > 1:
        fig.add_trace(go.Scatter(x=d["Date"], y=d["y_true"].rolling(roll, min_periods=roll).mean(), mode="lines", name=f"Actual, {roll}-day mean",
                                 line=dict(color=NAVY, width=1.4, dash="dot"), hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra>trend</extra>"))
    if top_n:
        t = d.assign(_a=(d["y_true"] - d["y_pred"]).abs()).nlargest(top_n, "_a")
        fig.add_trace(go.Scatter(x=t["Date"], y=t["y_true"], mode="markers", name=f"{top_n} largest errors",
                                 marker=dict(size=10, symbol="circle-open", color=NAVY, line=dict(width=2)),
                                 customdata=(t["y_true"] - t["y_pred"]), hovertemplate="%{x|%Y-%m-%d}<br>residual %{customdata:+,.0f}<extra></extra>"))
    if extra is not None and len(extra):
        e = extra.sort_values("Date")
        fig.add_trace(go.Scatter(x=e["Date"], y=e["y_pred"], mode="lines", name="Forecast — official test dates (no actuals)",
                                 line=dict(color=BLUE, width=2, dash="dot"), hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra>forecast</extra>"))
        fig.add_vrect(x0=e["Date"].min(), x1=e["Date"].max(), fillcolor="#e9f3ef", opacity=0.5, line_width=0, layer="below")
    style_fig(fig, height=height)
    fig.update_yaxes(title_text="Guests per day", separatethousands=True)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.04, x=0))
    return fig


def residuals_vs_pred(pred: pd.DataFrame, height: int = 330) -> go.Figure:
    r = pred["y_true"] - pred["y_pred"]
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=MUTED, width=1.4, dash="dash"))
    fig.add_trace(go.Scattergl(x=pred["y_pred"], y=r, mode="markers", marker=dict(size=6.5, color=np.where(r > 0, BLUE, ORANGE), opacity=0.65,
                               line=dict(width=0.6, color="#ffffff")), customdata=_cd(pred),
                               hovertemplate="%{customdata}<br>predicted %{x:,.0f}<br>residual %{y:+,.0f}<extra></extra>", showlegend=False))
    style_fig(fig, height=height, legend=False)
    fig.update_xaxes(title_text="Predicted Guests", separatethousands=True)
    fig.update_yaxes(title_text="Residual (actual − predicted)", separatethousands=True)
    fig.add_annotation(x=1, xref="paper", y=1, yref="paper", text="above 0: under-predicted", showarrow=False, xanchor="right", font=dict(size=11, color=BLUE))
    fig.add_annotation(x=1, xref="paper", y=0, yref="paper", text="below 0: over-predicted", showarrow=False, xanchor="right", font=dict(size=11, color=ORANGE))
    return fig


def residual_hist(pred: pd.DataFrame, bins: int = 40, height: int = 330) -> go.Figure:
    r = pred["y_true"] - pred["y_pred"]
    fig = go.Figure(go.Histogram(x=r, nbinsx=bins, marker=dict(color=BLUE, line=dict(color="#ffffff", width=1)),
                                 hovertemplate="residual %{x}<br>count %{y}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1.4, dash="dash"))
    fig.add_vline(x=float(r.mean()), line=dict(color=ORANGE, width=2), annotation_text=f"mean {r.mean():+,.0f}", annotation_position="top")
    style_fig(fig, height=height, legend=False)
    fig.update_xaxes(title_text="Residual (actual − predicted)", separatethousands=True)
    fig.update_yaxes(title_text="Count")
    return fig


def residual_ts(daily: pd.DataFrame, absolute: bool = False, height: int = 300) -> go.Figure:
    d = daily.sort_values("Date")
    r = d["y_true"] - d["y_pred"]
    v = r.abs() if absolute else r
    fig = go.Figure()
    if not absolute:
        fig.add_hline(y=0, line=dict(color=MUTED, width=1.4, dash="dash"))
    fig.add_trace(go.Bar(x=d["Date"], y=v, marker=dict(color=np.where(r > 0, BLUE, ORANGE) if not absolute else BLUE),
                         hovertemplate="%{x|%Y-%m-%d}: %{y:+,.0f}<extra></extra>" if not absolute else "%{x|%Y-%m-%d}: %{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=d["Date"], y=v.rolling(14, min_periods=7).mean(), mode="lines", line=dict(color=NAVY, width=2), name="14-day mean",
                             hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra>14-day mean</extra>"))
    style_fig(fig, height=height, legend=False)
    fig.update_yaxes(title_text="Absolute error" if absolute else "Residual (actual − predicted)", separatethousands=True)
    fig.update_layout(bargap=0.15)
    return fig


def breakdown(tab: pd.DataFrame, by: str, metric: str, height: int = 300) -> go.Figure:
    fig = go.Figure(go.Bar(x=tab[by].astype(str), y=tab[metric], marker=dict(color=BLUE, cornerradius=4),
                           text=[f"{v:,.0f}" if abs(v) >= 10 else f"{v:.3f}" for v in tab[metric]], textposition="outside", cliponaxis=False,
                           textfont=dict(color=INK2, size=11), hovertemplate="%{x}: %{y:,.2f}<extra></extra>"))
    style_fig(fig, height=height, legend=False)
    fig.update_yaxes(title_text=metric)
    return fig


def fold_bars(ft: pd.DataFrame, metric: str, holdout_value: float | None = None, pct: bool = False, height: int = 290) -> go.Figure:
    lab = [f"Fold {int(f)}<br><span style='font-size:10px'>{a} → {b}</span>" for f, a, b in zip(ft["fold"], ft["val_start"], ft["val_end"])]
    vals = ft[metric].tolist()
    colors = [SOFT] * len(vals)
    if holdout_value is not None and np.isfinite(holdout_value):
        lab.append("Holdout")
        vals.append(holdout_value)
        colors.append(BLUE)
    fig = go.Figure(go.Bar(x=lab, y=vals, marker=dict(color=colors, cornerradius=4), text=[(f"{v:.1%}" if pct else f"{v:,.0f}" if abs(v) >= 10 else f"{v:.3f}") for v in vals],
                           textposition="outside", cliponaxis=False, textfont=dict(color=INK2, size=11), hovertemplate="%{y:,.4f}<extra></extra>"))
    style_fig(fig, height=height, legend=False)
    fig.update_yaxes(title_text=metric, tickformat=".0%" if pct else None)
    return fig


def importance_bars(imp: pd.DataFrame, col: str, err: str | None = None, xlab: str = "", top: int = 20, height: int | None = None) -> go.Figure:
    d = imp.dropna(subset=[col]).sort_values(col).tail(top)
    fig = go.Figure(go.Bar(y=d["label"], x=d[col], orientation="h", marker=dict(color=np.where(d[col] < 0, ORANGE, BLUE), cornerradius=4),
                           error_x=dict(type="data", array=d[err], color=INK2, thickness=1.2, width=3) if err and err in d else None,
                           hovertemplate="%{y}: %{x:,.4f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#cfd1d0", width=1))
    style_fig(fig, height=height or max(260, 24 * len(d) + 70), legend=False)
    fig.update_xaxes(title_text=xlab)
    fig.update_layout(bargap=0.3)
    return fig


def metric_bars(df: pd.DataFrame, metric: str, label_col: str, highlight: str | None = None, pct: bool = False, height: int | None = None) -> go.Figure:
    higher_better = metric in ("R2", "Pearson", "Spearman")
    d = df.sort_values(metric, ascending=not higher_better)
    colors = [BLUE if (highlight is None or r == highlight) else SOFT for r in d.get("model_id", d[label_col])]
    fig = go.Figure(go.Bar(y=d[label_col], x=d[metric], orientation="h", marker=dict(color=colors, cornerradius=4),
                           text=[(f"{v:.1%}" if pct else f"{v:,.3f}" if abs(v) < 10 else f"{v:,.0f}") for v in d[metric]],
                           textposition="outside", cliponaxis=False, textfont=dict(color=INK2, size=11), hovertemplate="%{y}: %{x:,.4f}<extra></extra>"))
    style_fig(fig, height=height or max(240, 36 * len(d) + 70), legend=False)
    xmin, xmax = float(np.nanmin(d[metric])), float(np.nanmax(d[metric]))
    fig.update_xaxes(title_text=metric + (" (higher is better)" if higher_better else " (lower is better)"),
                     range=[min(0, xmin * 1.2), xmax * 1.25 if xmax > 0 else 0.05], tickformat=".0%" if pct else None)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(bargap=0.32, transition=dict(duration=400, easing="cubic-in-out"))
    return fig


def timeline(model_start: str, folds: list[dict], holdout: tuple[str, str], height: int = 220) -> go.Figure:
    """Chronological validation layout: expanding training windows, validation blocks, final holdout."""
    fig = go.Figure()
    rows = []
    for f in folds:
        rows.append((f"Fold {f['fold']}", f["train_start"], f["train_end"], "Training", BLUE))
        rows.append((f"Fold {f['fold']}", f["val_start"], f["val_end"], "Validation", AQUA))
    rows.append(("Final", model_start, str((pd.Timestamp(holdout[0]) - pd.Timedelta(days=1)).date()), "Training", BLUE))
    rows.append(("Final", holdout[0], holdout[1], "Holdout (never used for fitting or tuning)", ORANGE))
    seen = set()
    for name, a, b, kind, col in rows:
        span = ((pd.Timestamp(b) - pd.Timestamp(a)).days + 1) * 86_400_000
        fig.add_trace(go.Bar(y=[name], x=[span], base=[pd.Timestamp(a).isoformat()], orientation="h", marker=dict(color=col, cornerradius=3),
                             name=kind, showlegend=kind not in seen, opacity=0.9,
                             hovertemplate=f"{name} · {kind}<br>{a} → {b}<extra></extra>"))
        seen.add(kind)
    style_fig(fig, height=height)
    fig.update_xaxes(type="date")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(barmode="overlay", bargap=0.35, legend=dict(orientation="h", y=1.08, x=0))
    return fig
