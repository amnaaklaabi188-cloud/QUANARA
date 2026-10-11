"""Headless smoke test of every UI section with Streamlit's AppTest (uses the saved reference models)."""
import shutil

import pytest
from streamlit.testing.v1 import AppTest

from quanara import config, registry

PAGES = ["overview", "data_intel", "features_page", "eda", "training_page", "evaluation_page", "simulator", "copilot_page", "methodology", "demo"]
needs_models = pytest.mark.skipif(not registry.list_models(), reason="reference models not trained (run scripts/train_reference_models.py)")


def _render(module):
    import importlib

    from quanara.ui.common import inject_css
    inject_css()
    importlib.import_module(f"quanara.ui.{module}").render()


@needs_models
@pytest.mark.parametrize("module", PAGES)
def test_page_renders_without_exception(module):
    at = AppTest.from_function(_render, args=(module,), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


@needs_models
def test_simulator_run_button_produces_results():
    before = set(config.SCENARIOS_DIR.glob("*")) if config.SCENARIOS_DIR.exists() else set()
    at = AppTest.from_function(_render, args=("simulator",), default_timeout=240)
    at.run()
    run = [b for b in at.button if b.label == "Run Simulation"]
    assert run, "Run Simulation button missing"
    run[0].click().run()
    try:
        assert not at.exception, [e.value for e in at.exception]
        text = " ".join(m.value for m in at.markdown)
        for card in ("Baseline", "Scenario", "Difference", "Change"):
            assert f"<span>{card}</span><b>" in text
        assert "guests/day" in text and "Run the simulation" not in text
    finally:
        for p in set(config.SCENARIOS_DIR.glob("*")) - before:
            shutil.rmtree(p, ignore_errors=True)


def test_train_model_button_trains_and_saves(tmp_registry):
    at = AppTest.from_function(_render, args=("training_page",), default_timeout=240)
    at.run()
    at.selectbox(key="tr_algo").set_value("ridge").run()
    at.slider(key="tr_folds").set_value(1).run()
    [b for b in at.button if b.label == "Train model"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    saved = registry.list_models()
    assert len(saved) == 1 and saved[0]["algorithm"] == "ridge"
    assert any("Holdout MAE" in m.value for m in at.markdown)
