"""Section 9 — Methodology & Limitations.

Eight sections a judge can read quickly (data sources → preparation → features → training →
chronological validation → simulation → limitations → reproducibility). Every figure is computed from
the audit report, the feature catalogue, the registry or executed checks; long technical material
(hashes, file names, model IDs, feature lists, build log) sits in collapsible panels.
"""
from __future__ import annotations

from collections import Counter
from html import escape

import pandas as pd
import streamlit as st

from .. import config, leakage, registry
from .. import model_identity as MI
from ..copilot import answer_limitations, limitation_facts
from ..features import CATALOG, MONOTONE_INCREASING, available_features
from ..intel_facts import dictionary_reconciliation
from ..models import available_algorithms
from . import design, heroes
from .common import active_id, analytic, audit, next_step
from .design import ICON

CSS = """
<style>
.mt-steps { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; font-family:var(--q-font); }
.mt-step { background:#fff; border:1px solid #e5e8e6; border-radius:14px; padding:12px 14px; box-shadow:0 1px 2px rgba(15,23,20,.04); animation:dsFadeUp .5s ease both; }
.mt-step .n { font-size:11px; font-weight:800; letter-spacing:.12em; color:#047857; } .mt-step .t { font-size:14.5px; font-weight:750; color:#101614; margin-top:2px; }
.mt-step .d { font-size:12.4px; color:#4b4f4d; line-height:1.45; margin-top:4px; }
.mt-body { font-family:var(--q-font); font-size:14px; color:#2b302d; line-height:1.6; }
.mt-body ul { margin:6px 0 0; padding-left:20px; } .mt-body li { margin-bottom:5px; } .mt-body b { color:#101614; }
.mt-facts { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:6px 0 12px; font-family:var(--q-font); }
.mt-facts div { background:#f3f6f4; border-radius:10px; padding:8px 11px; } .mt-facts span { display:block; font-size:11.5px; color:#6f7572; }
.mt-facts b { font-size:16px; color:#101614; }
.mt-ok { display:flex; gap:9px; align-items:flex-start; font-family:var(--q-font); font-size:13.4px; color:#2b302d; padding:7px 0; border-bottom:1px solid #eef1ef; }
.mt-ok svg { width:16px; height:16px; color:#047857; flex:none; margin-top:2px; } .mt-ok.bad svg { color:#9b1c1c; } .mt-ok small { display:block; color:#6f7572; font-size:12.2px; }
@media (max-width: 1100px) { .mt-steps { grid-template-columns:repeat(2,minmax(0,1fr)); } .mt-facts { grid-template-columns:repeat(2,minmax(0,1fr)); } }
</style>
"""

LABEL = {"intl_train": "International guests · train", "intl_test": "International guests · test", "dom_train": "Domestic guests · train",
         "dom_test": "Domestic guests · test", "flight": "Flight operations"}


def _qstats() -> dict:
    from .data_intel import _qstats as di_qstats  # same cached statistics as the Data Intelligence page
    return di_qstats()


@st.cache_data(show_spinner="Running the leakage tests …")
def _leak(model_id: str | None) -> list[dict]:
    return leakage.run_checks(analytic(), registry.load_metadata(model_id) if model_id else None)


def _facts(items: list[tuple[str, str]]) -> str:
    return '<div class="mt-facts">' + "".join(f"<div><span>{escape(k)}</span><b>{escape(v)}</b></div>" for k, v in items) + "</div>"


def _body(html: str) -> None:
    design.html(f'<div class="mt-body">{html}</div>')


def render() -> None:
    design.inject()
    st.markdown(heroes.CSS + CSS, unsafe_allow_html=True)
    design.html(heroes.banner("methodology", "Section 9", "Methodology &amp; <span>Limitations</span>",
                              "How QuanAra was built, what it assumes, and where it should not be trusted."))
    st.write("")
    a, s = audit(), _qstats()
    h, f, mm = a["hotel"], a["flight"], a["market_matching"]
    mid = active_id()
    md = registry.load_metadata(mid) if mid else None
    hold_days = (config.MODEL_END - config.DEFAULT_HOLDOUT_START).days + 1
    n_wi = [len(available_features(sc_, "whatif")) for sc_ in config.SCOPES]
    model_days = (config.MODEL_END - config.MODEL_START).days + 1

    steps = [("01", "Data", f"Five workbooks and the dictionary; daily grain from {pd.Timestamp(f['grain']['daily']['start']):%b %Y}."),
             ("02", "Features", f"{min(n_wi)}–{max(n_wi)} scenario-ready predictors (by scope), built on trailing windows."),
             ("03", "Models", f"{len(available_algorithms())} algorithms, one shared pipeline, fixed seed {config.RANDOM_STATE}."),
             ("04", "Validation", f"Chronological holdout of {hold_days} days plus expanding-window CV."),
             ("05", "Simulation", "Change aviation inputs, rebuild features, re-predict with the saved model.")]
    design.html('<div class="mt-steps">' + "".join(f'<div class="mt-step"><div class="n">STEP {n}</div><div class="t">{t}</div><div class="d">{escape(d)}</div></div>'
                                                   for n, t, d in steps) + "</div>")
    st.write("")
    names = ["Data", "Preparation", "Features", "Training", "Validation", "Simulation", "Limitations", "Reproducibility"]
    with st.container(key="ds_panel_mt"):
        tabs = st.tabs([f"{i + 1} · {n}" for i, n in enumerate(names)], key="mt_tabnav", on_change="rerun")

        if tabs[0].open:
            with tabs[0]:
                inv = pd.DataFrame(a["inventory"])
                rows = {"intl_train": h["intl_train"]["rows"], "intl_test": h["intl_test"]["rows"], "dom_train": h["dom_train"]["rows"],
                        "dom_test": h["dom_test"]["rows"], "flight": f["rows"]}
                dates = {"intl_train": h["intl_train"]["dates"], "intl_test": h["intl_test"]["dates"], "dom_train": h["dom_train"]["dates"],
                         "dom_test": h["dom_test"]["dates"], "flight": f["dates"]}
                tab = pd.DataFrame([{"Dataset": LABEL[k], "Rows": rows[k], "From": dates[k]["start"], "To": dates[k]["end"],
                                     "Guests observed": "Yes" if k in ("intl_train", "dom_train") else ("—" if k == "flight" else "No (test period)")}
                                    for k in LABEL])
                _body(f"QuanAra uses only the six supplied files: four hotel workbooks, the flight-operations workbook and Data_Dictionary.pdf. "
                      f"No external data, holiday calendars or APIs are used. The originals are never edited; the app reads byte-identical copies "
                      f"({len(inv)} files verified by SHA-256 — see Reproducibility).")
                st.dataframe(tab, hide_index=True, width="stretch", column_config={"Rows": st.column_config.NumberColumn(format="%,d")})
                with st.expander("Where the dictionary and the workbooks disagree", icon=":material/compare:"):
                    rec = dictionary_reconciliation(s)
                    st.dataframe(pd.DataFrame([{"Topic": r["topic"], "Dictionary says": r["documented"], "Workbooks show": r["observed"],
                                                "Consequence": r["implication"]} for r in rec]), hide_index=True, width="stretch")

        if tabs[1].open:
            with tabs[1]:
                design.html(_facts([("Modelling window", f"{config.MODEL_START:%d %b %Y} – {config.MODEL_END:%d %b %Y}"), ("Days", f"{model_days:,}"),
                                    ("Feature warm-up", f"{config.FEATURE_WARMUP_DAYS} days"),
                                    ("Matched source markets", f"{len(mm['matched'])} of {mm['nationalities']}")]))
                _body(f"""<ul>
<li><b>Blank cells stay missing.</b> The dictionary’s “*” arrives as an empty cell and is never read as zero. One explicit exception: summed Same-Day Guests count blanks as 0 and are labelled a lower bound.</li>
<li><b>Daily grain.</b> Flights are monthly in 2022 ({f['grain']['monthly']['rows']:,} rows) and daily from {f['grain']['daily']['start']} ({f['grain']['daily']['rows']:,} rows). Monthly rows are never split into days; they appear only in monthly views.</li>
<li><b>Sums, then ratios.</b> Seats, passengers and flights are summed per day (or day × market); load factor and P2P share are recomputed from those sums, never averaged.</li>
<li><b>Daily flights recovered exactly</b> from Average Weekly Frequency (largest decode error {f['frequency_decode_max_error']:.0e}).</li>
<li><b>Kept as recorded:</b> load factors above 100% ({f['load_factor_gt_100_daily_pct']:.1f}% of daily rows). <b>Excluded:</b> Total Transit (non-zero on {f['transit_nonzero_daily_rows']} daily rows) and cabin-level passenger counts that do not add up to the totals.</li>
<li><b>Source markets</b> pair a nationality with the departure country of the same name ({mm['guest_share_in_matched_nationalities_pct']:.1f}% of guest-days). This is a proxy: passengers from a country are not all its nationals.</li>
<li><b>Guests is a stock</b> (guests in-house each day) fed by arrivals over several days, which is why trailing 7- and 28-day aviation windows are used. Length of stay itself is not in the files.</li>
</ul>""")

        if tabs[2].open:
            with tabs[2]:
                groups = Counter(x.group for x in CATALOG)
                design.html(_facts([("Aviation", str(groups["aviation"])), ("Derived ratios & growth", str(groups["derived"])), ("Calendar", str(groups["time"])),
                                    ("Last year's guests", str(groups["historical"]))]))
                last_lag = config.HOTEL_TRAIN_END + pd.Timedelta(days=config.SEASONAL_LAG_DAYS)
                _body(f"""<ul>
<li><b>Aviation:</b> seats, P2P, PAX, transfer, flights and load factor — same day, trailing 7 and 28 days, and the previous week.</li>
<li><b>Derived:</b> P2P and transfer shares, seats per flight, business-seat and Etihad seat shares, active routes, cities and airlines, 28-day growth.</li>
<li><b>Calendar:</b> day of week, weekend (Sat–Sun), month, quarter and seasonal sine/cosine terms.</li>
<li><b>Last year's guests:</b> Guests on the same weekday 364 days earlier and its 7-day mean — available for every date up to {last_lag:%d %b %Y}.</li>
<li><b>Competition only ({groups['competition']}):</b> New Arrivals and Same-Day Guests, which the official test files provide but which do not exist on scenario dates.</li>
<li><b>When values are known:</b> calendar terms and scheduled capacity are known in advance; realised passenger figures only on the day. For scenarios this is fine (the user sets them); a true advance forecast would need airline schedules.</li>
<li><b>Scenario-safe shape constraint</b> (tree models, on by default): predictions cannot fall when {', '.join(sorted(MONOTONE_INCREASING))} rise.</li>
</ul>""")

        if tabs[3].open:
            with tabs[3]:
                algos = available_algorithms()
                design.html(_facts([("Algorithms available", str(len(algos))), ("Random seed", str(config.RANDOM_STATE)),
                                    ("Modes", "What-if · Competition"), ("Scopes", str(len(config.SCOPES)))]))
                _body(f"""<ul>
<li><b>Algorithms:</b> {', '.join(MI.ALGORITHMS.get(k, k) for k in algos)}. The seasonal naive baseline (same weekday last year) is the reference every model must beat.</li>
<li><b>One pipeline</b> for training, evaluation, test-period forecasts and the simulator: imputation and scaling are fitted inside the training rows only.</li>
<li><b>What-if mode</b> uses aviation, calendar and last-year guests, so its models can run scenarios. <b>Competition mode</b> may add same-day hotel arrivals to score the official test dates; those models cannot run scenarios.</li>
<li><b>Prediction targets:</b> {'; '.join(MI.TARGETS[k] for k in config.SCOPES)}.</li>
<li><b>Training is explicit:</b> nothing retrains when a control changes; every model is saved with its predictions, settings and software versions.</li>
</ul>""")

        if tabs[4].open:
            with tabs[4]:
                design.html(_facts([("Fit on", f"{config.MODEL_START:%b %Y} – {(config.DEFAULT_HOLDOUT_START - pd.Timedelta(days=1)):%b %Y}"),
                                    ("Holdout", f"{config.DEFAULT_HOLDOUT_START:%d %b %Y} – {config.MODEL_END:%d %b %Y}"), ("Holdout days", f"{hold_days}"),
                                    ("Random splits", "None")]))
                _body("""<ul>
<li><b>Chronological holdout:</b> models are fitted on earlier months and scored on the most recent six observed months, which they never see.</li>
<li><b>Expanding-window cross-validation</b> before the holdout: each fold trains on all earlier dates and validates on the next block.</li>
<li><b>Deployment model:</b> refitted with identical settings on all modelling dates and used for forecasts and scenarios.</li>
<li><b>Metrics</b> (MAE, RMSE, R², WMAPE, Pearson, Spearman) are recomputed from the saved holdout predictions whenever they are shown.</li>
<li><b>Comparisons</b> rank models only when they share the same prediction target and holdout period.</li>
</ul>""")
                design.html('<div class="ds-ph" style="margin-top:10px">Leakage tests executed on this page</div>')
                design.html("".join(f'<div class="mt-ok {"" if c["ok"] else "bad"}">{ICON["check" if c["ok"] else "warning"]}<span>{escape(c["check"])}'
                                    f'<small>{"Passed" if c["ok"] else ("Not run — no active model" if c["ok"] is None else "Failed")} · {escape(c["evidence"])}</small></span></div>'
                                    for c in _leak(mid)))

        if tabs[5].open:
            with tabs[5]:
                _body("""<ul>
<li><b>Controls:</b> seat capacity, flight frequency (aircraft size absorbs the difference, so seats are never double-counted), load factor and P2P share, for one departure country or all, over any supported period.</li>
<li><b>Recomputed quantities:</b> PAX = load factor × seats; P2P = P2P share × PAX; transfer = PAX − P2P − transit. Consistency checks run on every changed route-day.</li>
<li><b>Held fixed:</b> calendar terms, last year's guests, route network, airline and cabin mix.</li>
<li><b>Sustained changes</b> are applied across the 56-day feature look-back, so trailing features reflect the new level.</li>
<li><b>Same pipeline:</b> features are rebuilt with the training code and the saved deployment model predicts baseline and scenario on identical dates; a zero change reproduces the baseline exactly.</li>
<li><b>Sensitivity:</b> each input is changed alone and re-predicted, so the inputs can be ranked by the model's response.</li>
<li><b>Aviation-to-hospitality pathway:</b> seats, passengers and P2P arrivals are observed; hotel guests are predicted; visitor share, capture rate and length of stay are not estimable from the files and are never invented.</li>
<li><b>Extrapolation</b> beyond the training range is flagged per feature.</li>
</ul>""")

        if tabs[6].open:
            with tabs[6]:
                _body("<b>Correlation, not causation.</b> Flights and hotel demand share drivers — seasons, events, economic growth — and airlines schedule capacity "
                      "where they expect demand. The models learn associations. A scenario result is the model’s prediction when aviation inputs change and "
                      "everything else is held fixed: useful for planning and stress-testing, not proof of what a route change would cause.")
                st.markdown(answer_limitations(limitation_facts(analytic(), md, st.session_state.get("last_scenario"))))

        if tabs[7].open:
            with tabs[7]:
                _body("Everything needed to reproduce the reported results. These details are for auditors and are not required to use the app.")
                with st.expander("Source files and SHA-256 checksums", icon=":material/fingerprint:"):
                    inv = pd.DataFrame(a["inventory"])
                    st.dataframe(inv.rename(columns={"dataset": "Dataset", "file": "File", "bytes": "Bytes", "sha256": "SHA-256"}), hide_index=True, width="stretch")
                with st.expander("Reference and active models", icon=":material/inventory_2:"):
                    rows = [{"Model": MI.labels(registry.list_models()).get(m["model_id"]), "Model ID": m["model_id"], "Trained": m["created_at"][:16].replace("T", " "),
                             "Active": m["model_id"] == mid} for m in registry.list_models() if m.get("reference") or m["model_id"] == mid]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                if md:
                    with st.expander("Active model: features, settings and software", icon=":material/tune:"):
                        st.markdown("**Features (pipeline order):** " + ", ".join(md["features"]))
                        st.markdown("**Settings:** " + ", ".join(f"{k} = {v}" for k, v in md["params"].items()) + f" · seed {md.get('seed', config.RANDOM_STATE)}")
                        st.markdown("**Software:** " + ", ".join(f"{k} {v}" for k, v in (md.get("versions") or {}).items()))
                with st.expander("Integration build log", icon=":material/list_alt:"):
                    st.dataframe(pd.DataFrame(analytic().log.steps), hide_index=True, width="stretch")
                _body("The reference models can be rebuilt with the project's reference-training script (see the README); results are deterministic for the fixed seed.")
    next_step("method")
