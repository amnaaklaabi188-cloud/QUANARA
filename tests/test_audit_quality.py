"""Checks added in the end-to-end audit: one model identity everywhere, simulator eligibility that is
actually executable, honest algorithm lists, reproducible Judges' Demo, and no hard-coded data figures."""
import json
import re
import shutil
from collections import Counter

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from quanara import config, registry
from quanara import model_identity as MI
from quanara import scenario as S
from quanara.models import HAS_LIGHTGBM, available_algorithms

MODELS = registry.list_models()
needs_models = pytest.mark.skipif(not MODELS, reason="reference models not trained")


@needs_models
def test_display_names_are_unique_and_never_show_raw_ids():
    labels = MI.labels(MODELS)
    assert len(set(labels.values())) == len(labels), [k for k, v in Counter(labels.values()).items() if v > 1]
    assert not any(m["model_id"] in labels[m["model_id"]] for m in MODELS)
    md = MODELS[0]
    facts = MI.facts(md)
    assert set(facts) == {"Algorithm", "Prediction target", "Model scope", "Validation"} and all(facts.values())


@needs_models
def test_every_simulator_ready_model_runs_a_scenario(analytic):
    ready = [m for m in MODELS if MI.scenario_ready(m)]
    assert ready and not any(m["algorithm"] == "seasonal_naive" or m["mode"] != "whatif" for m in ready)
    for m in ready:
        market = "INDIA" if m["scope"] == "international_market" else "ALL"
        r = S.run(S.ScenarioConfig(model_id=m["model_id"], market=market, start="2025-03-01", end="2025-03-07", seats_change_pct=10.0), analytic, save=False)
        assert np.isfinite(r.summary["pct_change"]) and (r.daily[["baseline", "scenario"]] >= 0).all().all(), m["model_id"]


@needs_models
def test_comparable_requires_same_target_and_holdout():
    a = next(m for m in MODELS if m["scope"] == "international_total")
    other_scope = next(m for m in MODELS if m["scope"] != a["scope"])
    assert MI.comparable(a, a) and not MI.comparable(a, other_scope)
    shifted = {**a, "periods": {**a["periods"], "holdout": ["2025-03-01", a["periods"]["holdout"][1]]}}
    assert not MI.comparable(a, shifted)


def _render(module):
    import importlib

    from quanara.ui.common import inject_css
    inject_css()
    importlib.import_module(f"quanara.ui.{module}").render()


@needs_models
def test_training_studio_offers_only_installed_algorithms():
    at = AppTest.from_function(_render, args=("training_page",), default_timeout=240)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    opts = set(at.selectbox(key="tr_algo").options)
    names = {MI.ALGORITHMS[k] for k in available_algorithms()}
    assert opts <= names
    assert ("LightGBM" in opts) == HAS_LIGHTGBM
    text = " ".join(m.value for m in at.markdown).lower()
    assert "not installed" not in text and "pip install" not in text


@needs_models
def test_judges_demo_reproduces_the_saved_result_and_compares_like_for_like():
    p = config.ARTIFACTS_DIR / "judges_demo.json"
    if not p.exists():
        pytest.skip("no reference scenario saved")
    saved = json.loads(p.read_text())
    at = AppTest.from_function(_render, args=("demo",), default_timeout=300)
    at.run()
    at.button(key="dm_run").click().run()
    assert not at.exception, [e.value for e in at.exception]
    res = at.session_state["demo_result"]
    assert np.isclose(res.summary["baseline_avg_daily_guests"], saved["summary"]["baseline_avg_daily_guests"])
    assert np.isclose(res.summary["scenario_avg_daily_guests"], saved["summary"]["scenario_avg_daily_guests"])
    md = registry.load_metadata(saved["model_id"])
    assert all(MI.comparable(md, registry.load_metadata(x)) for x in res.comparison["model_id"])
    assert any("Reproduced exactly" in m.value for m in at.markdown)


def test_dictionary_notes_hold_no_hard_coded_data_figures():
    from quanara.dictionary import FIELDS
    allowed = {"7", "10", "100%", "2022", "2023", "0.4702", "47.02%", "1", "2"}  # documented examples, calendar, and the "2" in "P2P"
    for f in FIELDS:
        for k in ("audit_finding", "missing_meaning", "dtype_units"):
            nums = [x for x in re.findall(r"\d[\d,.]*%?", f[k]) if x not in allowed]
            assert not nums, (f["column"], k, nums)


@needs_models
def test_copilot_page_answers_every_quick_question(analytic):
    from quanara.copilot import INTENTS
    before = set(config.SCENARIOS_DIR.glob("*")) if config.SCENARIOS_DIR.exists() else set()
    at = AppTest.from_function(_render, args=("copilot_page",), default_timeout=300)
    at.run()
    mid = registry.active_model_id()
    at.session_state["last_scenario"] = S.run(S.ScenarioConfig(model_id=mid, market="ALL", start="2025-09-01", end="2025-09-14", seats_change_pct=10.0),
                                              analytic, save=False)
    at.run()
    try:
        for k in INTENTS:
            at.button(key=f"cp_q_{k}").click().run()
            assert not at.exception, (k, [e.value for e in at.exception])
            ans = at.session_state["copilot_history"][-1]["a"]
            assert ans and "SCN-" not in ans and mid not in ans, k
    finally:
        for p in set(config.SCENARIOS_DIR.glob("*")) - before:
            shutil.rmtree(p, ignore_errors=True)
