"""Section 8 — AI Copilot: evidence-based explanations of the active model and the latest scenario.

Left: the conversation (quick questions, free text, answers). Right: the evidence the answers are
built from — the active model, the latest scenario and an optional comparison model. The built-in
engine (quanara.copilot) writes every sentence from saved outputs; Claude is optional and receives
only the same aggregate context.
"""
from __future__ import annotations

import os
from html import escape

import streamlit as st

from .. import model_identity as MI
from .. import registry
from .. import science as SC
from ..copilot import INTENTS, answer, ask_claude, grounded_context, limitation_facts, llm_available, route
from . import brand, design, heroes
from .common import active_id, analytic, next_step
from .design import ICON

CSS = """
<style>
[class*="st-key-cp_"] [data-testid="stMarkdownContainer"] { margin-bottom:0 !important; }
.st-key-cp_chat, .st-key-cp_evidence { background:#fff; border:1px solid #e5e8e6; border-radius:16px; padding:16px 18px 14px; gap:.6rem;
    box-shadow:0 1px 2px rgba(15,23,20,.04),0 8px 24px rgba(15,23,20,.06); animation:dsFadeUp .5s ease both; }
.st-key-cp_evidence { animation-delay:.08s; }
.cp-h { display:flex; gap:10px; align-items:center; font-family:var(--q-font); }
.cp-h b { font-size:15px; color:#101614; font-weight:780; } .cp-h .ds-ico { width:34px; height:34px; } .cp-h .ds-ico svg { width:18px; height:18px; }
.cp-sub { font-family:var(--q-font); font-size:12.6px; color:#6f7572; line-height:1.45; }
.st-key-cp_questions button { min-height:44px; height:100%; text-align:left; justify-content:flex-start; }
.st-key-cp_questions button p { white-space:normal !important; font-size:13.2px; line-height:1.3; text-align:left; }
.st-key-cp_questions [data-testid="stHorizontalBlock"] { align-items:stretch; }
.cp-card { border:1px solid #e9eceb; border-radius:12px; padding:10px 12px; background:#fbfcfb; font-family:var(--q-font); }
.cp-card .k { font-size:11.5px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:#047857; }
.cp-card .t { font-size:14.5px; font-weight:750; color:#101614; margin-top:3px; line-height:1.3; }
.cp-card .d { font-size:12.4px; color:#4b4f4d; margin-top:3px; line-height:1.45; }
.cp-tiles { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; margin-top:8px; }
.cp-tiles div { background:#f0f5f2; border-radius:9px; padding:6px 9px; } .cp-tiles span { display:block; font-size:10.8px; color:#6f7572; }
.cp-tiles b { font-size:15px; color:#101614; white-space:nowrap; }
.cp-warn { font-family:var(--q-font); font-size:12.3px; line-height:1.45; border-radius:10px; padding:8px 11px; background:#fff7e8; color:#8a5a00; border:1px solid #fbe3b4; }
.cp-empty { font-family:var(--q-font); font-size:13px; color:#6f7572; padding:12px 2px; }
.st-key-cp_chat [data-testid="stChatMessage"] { background:#f7f9f8; border:1px solid #edf0ee; border-radius:12px; padding:10px 12px; }
.st-key-cp_chat [data-testid="stChatMessage"] table { font-size:13px; }
@media (max-width: 1100px) {
  .st-key-cp_main > div > [data-testid="stHorizontalBlock"] { flex-wrap:wrap; row-gap:12px; }
  .st-key-cp_main > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { flex:1 1 100% !important; min-width:100% !important; }
}
</style>
"""


@st.cache_data(show_spinner=False)
def _season(model_id: str, market: str, start: str, end: str) -> dict:
    return SC.market_season(analytic(), registry.load_metadata(model_id), market, start, end)


def _head(icon: str, title: str) -> None:
    design.html(f'<div class="cp-h"><div class="ds-ico teal">{ICON[icon]}</div><b>{escape(title)}</b></div>')


def render() -> None:
    design.inject()
    st.markdown(heroes.CSS + CSS, unsafe_allow_html=True)
    design.html(heroes.banner("copilot", "Section 8", "AI <span>Copilot</span>",
                              "Plain-language answers about the active model and your latest scenario, built from saved outputs."))
    st.write("")
    a = analytic()
    mid = active_id()
    md = registry.load_metadata(mid) if mid else None
    scn = st.session_state.get("last_scenario")
    scn_mid = scn.config["model_id"] if scn is not None else None
    labels = MI.labels(registry.list_models())
    facts = limitation_facts(a, md, scn)
    season = _season(scn_mid, scn.config["market"], scn.config["start"], scn.config["end"]) if scn is not None else None

    with st.container(key="cp_main"):
        left, right = st.columns([1.75, 1], gap="medium")
        # ------------------------------------------------------------ evidence (right)
        with right, st.container(key="cp_evidence"):
            _head("shield", "Evidence in use")
            design.html('<div class="cp-sub">Every answer is assembled from these saved results — nothing else.</div>')
            if md:
                h = md["metrics_holdout"]
                design.html(f'<div class="cp-card"><div class="k">Active model</div><div class="t">{escape(MI.algorithm(md))}</div>'
                            f'<div class="d">{escape(MI.target(md))} · {escape(MI.scope(md).lower())}<br>{escape(MI.validation(md))}</div>'
                            f'<div class="cp-tiles"><div><span>WMAPE</span><b>{h["WMAPE"]:.1%}</b></div><div><span>R²</span><b>{h["R2"]:.3f}</b></div>'
                            f'<div><span>MAE</span><b>{h["MAE"]:,.0f}</b></div></div></div>')
            else:
                design.html('<div class="cp-warn">No active model. Activate a What-if model in the ML Training Studio.</div>')
            if scn is not None:
                s, c = scn.summary, scn.config
                mk = "All departure countries" if c["market"] == "ALL" else c["market"].title()
                design.html(f'<div class="cp-card"><div class="k">Latest scenario</div><div class="t">{s["pct_change"]:+.1f}% predicted guests '
                            f'({s["abs_change_avg_daily"]:+,.0f}/day)</div><div class="d">{escape(mk)} · {c["start"]} → {c["end"]}<br>'
                            f'Seats {c["seats_change_pct"]:+.0f}%{" · frequency " + format(c["frequency_change_pct"], "+.0f") + "%" if c["frequency_mode"] == "independent" else ""}'
                            f'{" · load factor " + format(c["load_factor_value"], "+.1f") + " pp" if c["load_factor_mode"] != "baseline" else ""}'
                            f'{" · P2P share " + format(c["p2p_share_value"], "+.1f") + " pp" if c["p2p_share_mode"] != "baseline" else ""}</div></div>')
                if scn_mid != mid:
                    design.html(f'<div class="cp-warn">This scenario was run with {escape(labels.get(scn_mid, "another model"))}; '
                                'scenario answers refer to that model. Re-run it in the Simulator to use the active model.</div>')
            else:
                design.html('<div class="cp-card"><div class="k">Latest scenario</div><div class="d">None yet — run one in the What-if Simulator, '
                            'then ask “Why did the prediction change?”.</div></div>')
                from .pages import PAGES
                st.page_link(PAGES["sim"], label="Open the What-if Simulator", icon=":material/flight_takeoff:")
            comparable = [m["model_id"] for m in registry.list_models() if md and m["model_id"] != mid and MI.comparable(md, m)]
            b_ = st.selectbox("Compare with", comparable, format_func=lambda x: labels.get(x, x), key="cp_b", index=None,
                              placeholder="Choose a model with the same target", disabled=not comparable,
                              help="Only models with the same prediction target and holdout period are listed, so their errors measure the same task.")
            with st.expander("Optional: Claude explanations", icon=":material/auto_awesome:"):
                st.markdown("Off by default. When enabled, the **same aggregate evidence** shown here (metrics, feature names, scenario summary — never raw rows, "
                            "file paths or keys) is sent to `claude-opus-5-5` with instructions to answer only from it.")
                env_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
                key = None if env_key else st.text_input("API key (kept in this browser session only; never saved)", type="password", key="cp_key")
                use_llm = st.toggle("Use Claude for the next answers", value=False, key="cp_llm", disabled=not llm_available(key))
                if env_key:
                    st.caption("Using ANTHROPIC_API_KEY from the environment.")
                if mid:
                    with st.popover("Show exactly what would be sent"):
                        st.json(grounded_context(mid, scn, facts), expanded=False)

        # ------------------------------------------------------------ conversation (left)
        with left, st.container(key="cp_chat"):
            _head("bulb", "Ask a question")
            design.html('<div class="cp-sub">Pick a question or type your own. Answers quote saved metrics and scenario outputs, and say when something cannot be estimated.</div>')
            asked = None
            items = list(INTENTS.items())
            with st.container(key="cp_questions"):
                for row in (items[:3], items[3:]):
                    cols = st.columns(3, gap="small")
                    for col, (k, q) in zip(cols, row):
                        if col.button(q, key=f"cp_q_{k}", width="stretch"):
                            asked = q
            free = st.chat_input("Ask about the model or the scenario …", key="cp_input")
            question = free or asked
            hist = st.session_state.setdefault("copilot_history", [])
            if question:
                # scenario questions are answered with the model that produced the scenario; everything else with the active model
                model_for = scn_mid if scn is not None and route(question) in ("why_change", "planning") else mid
                intent, text = answer(question, model_for, scn, (mid, b_) if b_ else None, facts, season)
                hist.append({"q": question, "intent": intent, "a": text, "llm": None})
            if not hist:
                design.html('<div class="cp-empty">No questions yet.</div>')
            for i, h in enumerate(reversed(hist[-8:])):
                with st.chat_message("user"):
                    st.markdown(h["q"])
                with st.chat_message("assistant", avatar=str(brand.FAVICON)):
                    st.markdown(h["a"])
                    if i == 0 and question and use_llm and mid and h["llm"] is None:
                        with st.spinner("Asking Claude …"):
                            h["llm"] = ask_claude(h["q"], grounded_context(mid, scn, facts), key or None)
                    if h["llm"]:
                        st.markdown("---\n**Claude (from the same evidence):**")
                        st.markdown(h["llm"])
            if hist and st.button("Clear conversation", key="cp_clear", icon=":material/delete_sweep:"):
                st.session_state["copilot_history"] = []
                st.rerun()
    next_step("copilot")
