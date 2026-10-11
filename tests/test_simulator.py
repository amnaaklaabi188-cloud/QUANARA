"""Section 7 — What-if Simulator: the redesigned page reports real model predictions, every control moves
the intended inputs, sensitivity re-predicts with the saved model, and the scenario record matches the run."""
import shutil

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from quanara import config, registry
from quanara import scenario as S
from quanara import science as SC

MODELS = registry.list_models()
REF = next((m["model_id"] for m in MODELS if m["model_id"].startswith("MKT-WI-catboost")), None)
needs_ref = pytest.mark.skipif(REF is None, reason="reference market model not trained")
PERIOD = dict(market="RUSSIAN FEDERATION", start="2025-09-01", end="2025-09-30")


def _run(analytic, **kw):
    return S.run(S.ScenarioConfig(model_id=REF, **{**PERIOD, "seats_change_pct": 0.0, **kw}), analytic, save=False)


def _av(res):
    return res.aviation_changes.set_index("quantity")


@needs_ref
def test_zero_change_reproduces_the_baseline_exactly(analytic):
    r = _run(analytic)
    assert np.array_equal(r.daily["scenario"].to_numpy(), r.daily["baseline"].to_numpy())
    assert r.summary["abs_change_avg_daily"] == 0 and r.summary["pct_change"] == 0


@needs_ref
def test_each_control_moves_only_its_intended_inputs(analytic):
    seats = _av(_run(analytic, seats_change_pct=20.0))
    assert seats.loc["Seats", "change_%"] == pytest.approx(20) and seats.loc["Flights", "change_%"] == pytest.approx(20, abs=1)
    assert seats.loc["Load factor", "change_%"] == pytest.approx(0, abs=1e-9) and seats.loc["P2P share", "change_%"] == pytest.approx(0, abs=1e-9)

    freq = _av(_run(analytic, frequency_mode="independent", frequency_change_pct=20.0))
    assert freq.loc["Seats", "change"] == pytest.approx(0, abs=1e-6) and freq.loc["Flights", "change_%"] > 10
    assert freq.loc["Seats per flight", "change_%"] < 0

    lf = _av(_run(analytic, load_factor_mode="delta", load_factor_value=5.0))
    assert lf.loc["Seats", "change"] == pytest.approx(0, abs=1e-6) and lf.loc["PAX", "change"] > 0
    assert lf.loc["Load factor", "scenario"] == pytest.approx(lf.loc["Load factor", "baseline"] + 0.05, abs=0.002)

    sh = _av(_run(analytic, p2p_share_mode="delta", p2p_share_value=10.0))
    assert sh.loc["PAX", "change"] == pytest.approx(0, abs=1e-6) and sh.loc["P2P passengers", "change"] > 0
    assert sh.loc["Transfer passengers", "change"] < 0


@needs_ref
def test_sensitivity_uses_the_same_inference_as_a_full_run(analytic):
    sens = SC.sensitivity(REF, PERIOD["market"], PERIOD["start"], PERIOD["end"], analytic)
    row = sens[(sens["input"] == "seats") & (sens["change"] == 20)].iloc[0]
    r = _run(analytic, seats_change_pct=20.0)
    assert row["baseline"] == pytest.approx(r.summary["baseline_avg_daily_guests"], rel=1e-9)
    assert row["scenario"] == pytest.approx(r.summary["scenario_avg_daily_guests"], rel=1e-9)
    assert sens["valid"].any() and set(SC.sensitivity_ranking(sens)["input"]) <= set(SC.SENSITIVITY_INPUTS)


@needs_ref
def test_cards_record_and_predictions_agree(analytic):
    r = _run(analytic, seats_change_pct=-30.0)
    d = r.daily[r.daily["market_key"] == PERIOD["market"]]
    assert d["baseline"].mean() == pytest.approx(r.summary["baseline_avg_daily_guests"])
    assert d["scenario"].mean() == pytest.approx(r.summary["scenario_avg_daily_guests"])
    assert (r.daily[["baseline", "scenario"]] >= 0).all().all()
    rec = SC.scenario_record(r, registry.load_metadata(REF), None)
    assert rec["summary"] == r.summary and rec["config"] == r.config and len(rec["daily"]) == len(r.daily)
    chain = {c["stage"]: c for c in SC.conversion_chain(r, registry.load_metadata(REF))}
    assert [c["kind"] for c in chain.values()].count("unavailable") == 2      # visitors and guest nights are never fabricated


def _render():
    from quanara.ui.common import inject_css
    inject_css()
    from quanara.ui import simulator
    simulator.render()


@needs_ref
def test_page_controls_tabs_and_audit_render_with_real_predictions():
    before_dirs = set(config.SCENARIOS_DIR.glob("*")) if config.SCENARIOS_DIR.exists() else set()
    active = registry.active_model_id()
    at = AppTest.from_function(_render, default_timeout=600)
    at.run()
    try:
        at.selectbox(key="sim_model").set_value(REF).run()
        at.selectbox(key="sim_market").set_value(PERIOD["market"]).run()
        at.selectbox(key="sim_period").set_value("Sep 2025  · forecast").run()
        at.slider(key="sim_seats").set_value(0).run()
        at.button(key="sim_run").click().run()
        assert not at.exception, [e.value for e in at.exception]
        res = at.session_state["last_scenario"]
        assert res.summary["pct_change"] == 0                      # the UI's zero scenario equals the baseline
        at.slider(key="sim_seats").set_value(20).run()
        assert "controls changed" in " ".join(m.value for m in at.markdown)
        at.button(key="sim_run").click().run()
        res = at.session_state["last_scenario"]
        text = " ".join(m.value for m in at.markdown)
        assert f"<b>{res.summary['pct_change']:+.1f}%</b>" in text and f"<b>{res.summary['baseline_avg_daily_guests']:,.0f}</b>" in text
        for tab in ("Sensitivity analysis", "Conversion chain", "Market & seasonal view", "Demand over time"):
            at.session_state["sim_tabnav"] = tab
            at.run()
            assert not at.exception, (tab, [e.value for e in at.exception])
        at.button(key="sim_audit_btn").click().run()
        assert not at.exception, [e.value for e in at.exception]
        rec = at.session_state["scenario_record"]
        assert rec["scenario_id"] == res.scenario_id and rec["sensitivity"]
    finally:
        if active:
            registry.activate(active)
        for p in set(config.SCENARIOS_DIR.glob("*")) - before_dirs:
            shutil.rmtree(p, ignore_errors=True)
