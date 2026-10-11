"""AI Copilot: explains actual model outputs and simulation results.

Two engines:
  * Grounded rule-based engine (default, no API key, no network). Every sentence is generated
    from saved metadata, saved predictions and the latest scenario result.
  * Optional Claude engine (Anthropic API). Enabled only when the user supplies a key (env var
    ANTHROPIC_API_KEY or a session-only password field). It receives the SAME grounded context
    the rule engine uses — aggregate metrics, feature names and scenario summaries — never raw
    data rows, file paths or credentials, and is instructed to answer only from that context.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import config, registry
from . import model_identity as MI
from .evaluation import METRIC_HELP
from .features import CATALOG_BY_NAME

INTENTS = {
    "why_change": "Why did the prediction change?",
    "features": "Which features influenced the model?",
    "reliability": "How reliable is this forecast?",
    "compare": "How do two models compare?",
    "limitations": "What are the limitations?",
    "planning": "What does this mean for tourism planning?",
}

_KEYWORDS = {
    "why_change": ("why", "change", "changed", "differ", "increase", "decrease", "scenario", "drop", "rise"),
    "features": ("feature", "driver", "influence", "important", "importance", "variable", "matter"),
    "reliability": ("reliab", "accura", "trust", "error", "confiden", "uncertain", "good", "metric", "wmape", "r2", "mae"),
    "compare": ("compare", "comparison", "versus", " vs", "better", "two models", "other model", "difference between"),
    "limitations": ("limit", "caveat", "weakness", "assumption", "causal", "risk", "bias"),
    "planning": ("plan", "tourism", "mean for", "implication", "decision", "hotel operator", "destination", "practical", "use this"),
}


def route(question: str) -> str:
    q = question.lower()
    scores = {k: sum(w in q for w in ws) for k, ws in _KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def _pct(v) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.1%}"


def _num(v) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.3f}"


def _top_importance(model_id: str, n: int = 6) -> pd.DataFrame:
    imp = registry.load_table(model_id, "importance")
    if imp is None or imp.empty:
        return pd.DataFrame()
    col = "permutation_rmse_increase" if imp["permutation_rmse_increase"].notna().any() else "native"
    return imp.dropna(subset=[col]).sort_values(col, ascending=False).head(n).assign(score_col=col)


# ------------------------------------------------------------------ rule-based answers

def answer_why_change(scn, model_id: str) -> str:
    if scn is None:
        return "No scenario has been run in this session yet. Open the What-if Simulator, configure a scenario and press **Run Simulation** — I will then explain the result from its saved outputs."
    s, c = scn.summary, scn.config
    mk = "all departure countries" if c["market"] == "ALL" else c["market"].title()
    lines = [f"**This scenario** ({c['start']} → {c['end']}, {mk}) moved predicted guests from "
             f"**{_num(s['baseline_avg_daily_guests'])}** to **{_num(s['scenario_avg_daily_guests'])}** per day "
             f"(**{s['abs_change_avg_daily']:+,.0f}/day, {s['pct_change']:+.2f}%**).", "",
             "**What changed in the aviation inputs** (selected market and period):"]
    av = scn.aviation_changes.set_index("quantity")
    for q in ("Seats", "Flights", "PAX", "P2P passengers", "Transfer passengers", "Load factor", "P2P share"):
        if q in av.index:
            r = av.loc[q]
            if q in ("Load factor", "P2P share"):
                lines.append(f"- {q}: {_pct(r['baseline'])} → {_pct(r['scenario'])}")
            else:
                lines.append(f"- {q}: {_num(r['baseline'])} → {_num(r['scenario'])} ({r['change_%']:+.1f}%)")
    fc = scn.feature_changes
    ch = fc[fc["changed"]] if len(fc) else fc
    imp = _top_importance(model_id, 50)
    lines += ["", "**Which model inputs moved** (period mean, baseline → scenario):"]
    if ch.empty:
        lines.append("- None of the model's features changed, so the prediction could not change.")
    for _, r in ch.iterrows():
        rank = ""
        if not imp.empty and r["feature"] in set(imp["feature"]):
            pos = list(imp["feature"]).index(r["feature"]) + 1
            rank = f" — importance rank #{pos} of {len(imp)}"
        lines.append(f"- {r['label']}: {_num(r['baseline_mean'])} → {_num(r['scenario_mean'])} ({r['change_%']:+.1f}%){rank}")
    unchanged = fc[~fc["changed"]]["label"].tolist() if len(fc) else []
    if unchanged:
        lines.append(f"- Held fixed: {', '.join(unchanged[:8])}{' …' if len(unchanged) > 8 else ''}")
    lines += ["", "**Why the size of the effect is what it is**"]
    if scn.model["algorithm_label"].startswith(("Random", "CatBoost", "XGBoost")):
        lines.append("- This is a tree ensemble: its response is piecewise-constant and cannot extend beyond the largest values seen in training. "
                     "Increases that push inputs past the historical maximum are therefore damped, while decreases that stay inside the range respond more.")
    elif scn.model["algorithm_label"].startswith("Ridge"):
        lines.append("- Ridge is linear in its inputs: the response is proportional to the change and symmetric for increases and decreases, including outside the training range.")
    ex = scn.extrapolation
    if len(ex) and (ex["pct_days_outside"] > 0).any():
        worst = ex.sort_values("pct_days_outside", ascending=False).iloc[0]
        lines.append(f"- {int((ex['pct_days_outside'] > 0).sum())} feature(s) leave the training range — e.g. {worst['label']} on {worst['pct_days_outside']:.0f}% of days.")
    if scn.comparison is not None and len(scn.comparison) > 1:
        rng = scn.comparison[scn.comparison["model_id"].map(lambda x: registry.load_metadata(x)["algorithm"] != "seasonal_naive")]["pct_change"]
        if len(rng) > 1:
            lines.append(f"- Across {len(rng)} comparable models the predicted change ranges from {rng.min():+.2f}% to {rng.max():+.2f}% — treat the spread as model uncertainty.")
    lines += ["", "_This is a model-based predictive scenario, not a causal estimate._"]
    return "\n".join(lines)


def answer_features(model_id: str) -> str:
    md = registry.load_metadata(model_id)
    imp = _top_importance(model_id, 8)
    if md["algorithm"] == "seasonal_naive":
        return "The Seasonal Naive baseline uses no learned features: it predicts each day as the observed Guests on the same weekday 364 days earlier."
    if imp.empty:
        return "No importance table is saved for this model."
    col = imp["score_col"].iloc[0]
    lines = [f"**{MI.algorithm(md)}** ({MI.target(md).lower()}) — top drivers by "
             f"{'permutation importance on the holdout (increase in RMSE when the feature is shuffled)' if col == 'permutation_rmse_increase' else 'native model importance'}:"]
    for i, (_, r) in enumerate(imp.iterrows(), 1):
        unit = " guests RMSE" if col == "permutation_rmse_increase" else ""
        lines.append(f"{i}. **{r['label']}** ({r['group']}) — {r[col]:,.2f}{unit}")
    groups = imp.groupby("group")[col].sum().sort_values(ascending=False)
    lines += ["", f"By group, the top drivers are mostly **{groups.index[0]}** features."]
    lines.append("Importance describes what the model relies on for prediction; it is not evidence of a causal effect, and correlated features (e.g. seats and P2P) share credit.")
    return "\n".join(lines)


def answer_reliability(model_id: str, scn=None) -> str:
    md = registry.load_metadata(model_id)
    h, p = md["metrics_holdout"], md["periods"]
    lines = [f"**{MI.algorithm(md)}** ({MI.target(md).lower()}) — evaluated out-of-sample on a chronological holdout **{p['holdout'][0]} → {p['holdout'][1]}** "
             f"({md['n_rows']['holdout']:,} rows never seen during fitting):",
             f"- MAE **{_num(h['MAE'])}** guests · RMSE **{_num(h['RMSE'])}** · WMAPE **{_pct(h['WMAPE'])}** · R² **{h['R2']:.3f}** · Pearson {h['Pearson']:.3f} · Spearman {h['Spearman']:.3f}"]
    if md.get("metrics_holdout_daily_total"):
        d = md["metrics_holdout_daily_total"]
        lines.append(f"- Summed to daily totals across markets: WMAPE **{_pct(d['WMAPE'])}**, R² {d['R2']:.3f}.")
    if md.get("metrics_cv_mean"):
        cv = md["metrics_cv_mean"]
        lines.append(f"- Expanding-window CV ({len(p['cv_folds'])} folds before the holdout): mean WMAPE {_pct(cv['WMAPE'])}, mean MAE {_num(cv['MAE'])}.")
    lines.append(f"- 90% of holdout predictions were within ±{md['interval']['rel_q90']:.0%} of the predicted value (empirical, not a formal interval for scenario differences).")
    lines += ["", "**Caveats that affect reliability**",
              "- The official test files contain no Guests, so test-period forecasts cannot be scored.",
              f"- The deployment model is refit on all observed dates through {p['deployment_fit'][1]}; forecasts beyond that rely on the model generalising forward.",
              "- Model rankings differ between CV and holdout windows — reported side by side in Model Evaluation."]
    if scn is not None and len(scn.extrapolation) and (scn.extrapolation["pct_days_outside"] > 0).any():
        lines.append("- The latest scenario pushes some features outside the training range, so its prediction is an extrapolation.")
    return "\n".join(lines)


def answer_compare(id_a: str, id_b: str, scn=None) -> str:
    if id_a == id_b:
        return "Choose two different models to compare."
    a, b = registry.load_metadata(id_a), registry.load_metadata(id_b)
    labels = MI.labels(registry.list_models())
    la, lb = labels.get(id_a, MI.name(a)), labels.get(id_b, MI.name(b))
    if not MI.comparable(a, b):
        why = (f"they predict different targets ({MI.target(a)} vs {MI.target(b)})" if a["scope"] != b["scope"]
               else "they were scored on different holdout periods")
        return (f"**{la}** and **{lb}** cannot be ranked against each other: {why}, so their errors measure different tasks. "
                "Pick two models with the same prediction target and holdout period.")
    ha, hb = a["metrics_holdout"], b["metrics_holdout"]
    same_hold = True
    lines = [f"| metric | {la} | {lb} |", "|---|---|---|"]
    for k in ("MAE", "RMSE", "WMAPE", "R2", "Pearson", "Spearman"):
        fa, fb = (_pct(ha[k]), _pct(hb[k])) if k == "WMAPE" else (_num(ha[k]), _num(hb[k]))
        lines.append(f"| {k} | {fa} | {fb} |")
    better = a if ha["MAE"] < hb["MAE"] else b
    lines += ["", f"On the identical holdout period ({MI.validation(a).split(' · ')[1]}), **{la if better is a else lb}** has the lower MAE "
                  f"({METRIC_HELP['MAE'].lower().rstrip('.')})."]
    if a["mode"] != b["mode"]:
        lines.append("Note: one model is a Competition model that also uses same-day hotel arrivals, so its lower error reflects extra information, "
                     "not a better aviation model. Only What-if models can run scenarios.")
    if scn is not None and scn.comparison is not None:
        cmp = scn.comparison.set_index("model_id")
        if id_a in cmp.index and id_b in cmp.index:
            lines.append(f"In the latest scenario they predict {cmp.loc[id_a, 'pct_change']:+.2f}% and {cmp.loc[id_b, 'pct_change']:+.2f}% respectively.")
    return "\n".join(lines)


def limitation_facts(a, md: dict | None = None, scn=None) -> dict:
    """Numbers quoted by the limitations answer, computed from the analytical dataset and the active model."""
    tr = a.hotel_market[a.hotel_market["split"] == "train"]
    f = {"matched": len(a.markets), "nationalities": int(tr["market_key"].nunique()),
         "unobserved": int(tr["Date"].nunique() * tr["market_key"].nunique() - len(tr)),
         "model_start": config.MODEL_START, "years": (config.MODEL_END - config.MODEL_START).days / 365.25,
         "daily_total_wmape": (md.get("metrics_holdout_daily_total") or {}).get("WMAPE") if md else None, "band": None}
    if scn is not None and scn.summary.get("daily_error_band_rel_q90") is not None:
        f["band"] = (scn.summary["daily_error_band_rel_q90"], scn.summary.get("daily_error_band_basis", "holdout days"))
    return f


def answer_limitations(facts: dict | None = None) -> str:
    f = facts or {}
    m, n = f.get("matched"), f.get("nationalities")
    lines = [
        "- **No observed target in the test files** — accuracy is measured on held-out training months only.",
        ("- **Departure country ≠ nationality** — " + (f"{m} of {n} nationalities are linked to flights by an exact name match; the other {n - m} have no "
         "direct flight market. " if m and n else "nationalities are linked to flights by an exact country-name match. ")
         + "Passengers departing a country are not all its nationals, and nationals also arrive via other hubs."),
        ("- **2022 flights are monthly** — daily models start " + (f"{f['model_start']:%d %b %Y} (about {f['years']:.1f} years of daily data)"
         if f.get("model_start") is not None else "in 2023") + "; year-over-year aviation features are not possible."),
        ("- **Unobserved and blank values** — " + (f"{f['unobserved']} nationality-days have no row (the files do not say why); " if f.get("unobserved") is not None else "")
         + "Same-Day Guests blanks cannot be interpreted uniquely; the test extract follows a different inclusion rule (New Arrivals ≥ 10)."),
        "- **No holiday or event calendar** in the supplied data; Ramadan, Eid and events enter only indirectly through seasonality and last year's guests.",
        "- **Domestic demand** is not plausibly driven by inbound flights; the domestic What-if model reflects co-movement, not a mechanism.",
        "- **Trees cannot extrapolate** — scenarios beyond historical capacity are damped; Ridge extrapolates linearly but fits the holdout worse.",
    ]
    if f.get("daily_total_wmape") is not None:
        lines.append(f"- **Level error when markets are summed** — the active source-market model's daily-total WMAPE is {f['daily_total_wmape']:.1%}, "
                     "so absolute totals are less reliable than relative scenario comparisons.")
    if f.get("band"):
        lines.append(f"- **Wide daily error band** — for the latest scenario's selection, 90% of {f['band'][1]} fell within ±{f['band'][0]:.0%} of the prediction.")
    lines.append("- **Correlation, not causation** — scenario differences are what the model predicts when inputs change with everything else held fixed; "
                 "airlines add capacity where demand is expected, so causality runs both ways.")
    return "\n".join(lines)


def answer_planning(scn, model_id: str, ms: dict | None = None) -> str:
    """Practical reading of the latest scenario for destination planning — every number from the scenario or observed history."""
    if scn is None:
        return "Run a scenario in the What-if Simulator first; I will then translate its predicted change into planning terms."
    s, c = scn.summary, scn.config
    md = registry.load_metadata(model_id)
    days = int(s.get("days") or len(pd.date_range(c["start"], c["end"])))
    tot = s.get("abs_change_guest_days", s["abs_change_avg_daily"] * days)
    mk = "all departure countries" if c["market"] == "ALL" else c["market"].title()
    lines = [f"**In planning terms** ({c['start']} → {c['end']}, flights from {mk}):",
             f"- The model predicts **{s['abs_change_avg_daily']:+,.0f} guests per day** ({s['pct_change']:+.1f}%), about **{tot:+,.0f} guest-days** over {days} days.",
             "- Guest-days are not guest nights: length of stay is not in the files, so room-night demand cannot be derived."]
    if ms and np.isfinite(ms.get("period_vs_annual", np.nan)):
        lines.append(f"- Season: this period averages {ms['period_mean']:,.0f} observed guests/day for this population, "
                     f"{abs(ms['period_vs_annual']):.0%} {'below' if ms['period_vs_annual'] < 0 else 'above'} its annual mean.")
    if ms and ms.get("market_seat_share") is not None:
        lines.append(f"- Scale: {mk} supplies {ms['market_seat_share']:.1%} of arriving seats in the period, which bounds how much one market can move totals.")
    if scn.comparison is not None and len(scn.comparison) > 1:
        resp = scn.comparison[scn.comparison["model_id"].map(lambda x: registry.load_metadata(x)["algorithm"] != "seasonal_naive")]
        if len(resp) > 1:  # the seasonal baseline uses no aviation input, so it cannot respond and is left out of the spread
            lines.append(f"- Model spread: comparable models predict {resp['pct_change'].min():+.1f}% to {resp['pct_change'].max():+.1f}% — plan for a range, not a point.")
    lines += ["", "**How to use it:** as a directional, model-based signal for capacity, staffing and marketing discussions alongside airline schedules — "
              f"not as a guaranteed outcome. The estimate comes from {MI.algorithm(md)} (holdout WMAPE {md['metrics_holdout']['WMAPE']:.1%}) and is conditional, not causal."]
    return "\n".join(lines)


def grounded_context(model_id: str, scn=None, facts: dict | None = None) -> dict:
    md = registry.load_metadata(model_id)
    imp = _top_importance(model_id, 10)
    ctx = {"model": {**{k: md[k] for k in ("features", "metrics_holdout", "metrics_cv_mean", "periods", "interval", "monotone", "log_target")},
                     **MI.facts(md)},
           "top_importance": imp[["label", "group", imp["score_col"].iloc[0]]].to_dict(orient="records") if not imp.empty else []}
    if scn is not None:
        ctx["scenario"] = {"id": scn.scenario_id, "config": scn.config, "summary": scn.summary,
                           "aviation_changes": scn.aviation_changes.round(4).to_dict(orient="records"),
                           "feature_changes": scn.feature_changes.round(4).to_dict(orient="records"),
                           "extrapolation": scn.extrapolation.round(3).to_dict(orient="records"), "warnings": scn.warnings,
                           "comparison": scn.comparison.round(4).to_dict(orient="records") if scn.comparison is not None else None}
    ctx["limitations"] = answer_limitations(facts)
    return ctx


def answer(question: str, model_id: str | None, scn=None, compare_ids: tuple[str, str] | None = None, facts: dict | None = None,
           season: dict | None = None) -> tuple[str, str]:
    intent = route(question)
    if model_id is None:
        return intent, "No model is active. Train or activate a What-if model first (ML Training Studio → Activate)."
    if intent == "why_change":
        return intent, answer_why_change(scn, model_id)
    if intent == "features":
        return intent, answer_features(model_id)
    if intent == "reliability":
        return intent, answer_reliability(model_id, scn)
    if intent == "compare":
        if not compare_ids:
            return intent, "Choose a second model under “Compare with”, then ask again."
        return intent, answer_compare(*compare_ids, scn)
    if intent == "limitations":
        return intent, answer_limitations(facts)
    if intent == "planning":
        return intent, answer_planning(scn, model_id, season)
    return intent, ("I answer questions about the active model and the latest scenario. Try: " + "; ".join(f"*{v}*" for v in INTENTS.values()))


# ------------------------------------------------------------------ optional Claude engine

SYSTEM_PROMPT = (
    "You are the QuanAra Copilot, explaining a machine-learning model that predicts Abu Dhabi hotel guests from flight data. "
    "Answer ONLY from the JSON context provided in the user turn. If the context does not contain the answer, say so. "
    "Never invent metrics, dates or numbers; quote numbers exactly as given. Always distinguish prediction from causation: "
    "scenario differences are model predictions under changed inputs, not causal effects. Be concise (under 250 words) and use plain language suitable for a judging panel."
)


def llm_available(api_key: str | None) -> bool:
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False
    return bool(api_key or os.environ.get("ANTHROPIC_API_KEY"))


def ask_claude(question: str, context: dict, api_key: str | None = None) -> str:
    """Single grounded request to Claude. The key is used in memory only and never logged or saved."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model="claude-opus-5-5",
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "medium"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Context (JSON):\n{json.dumps(context, default=str)}\n\nQuestion: {question}"}],
        )
    except anthropic.AuthenticationError:
        return "The API key was rejected. The built-in grounded answer above is unaffected."
    except anthropic.RateLimitError:
        return "The Claude API is rate-limiting requests; please retry shortly. The built-in grounded answer above is unaffected."
    except anthropic.APIStatusError as e:
        return f"Claude API error ({e.status_code}). The built-in grounded answer above is unaffected."
    except anthropic.APIConnectionError:
        return "Could not reach the Claude API (network). The built-in grounded answer above is unaffected."
    if response.stop_reason == "refusal":
        return "Claude declined to answer this request. The built-in grounded answer above is unaffected."
    return "\n".join(b.text for b in response.content if b.type == "text").strip() or "(empty response)"
