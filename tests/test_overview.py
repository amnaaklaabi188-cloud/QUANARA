"""Page 1 (Welcome & Overview): every displayed fact is computed, the model card follows the
active model, the hero animates (with a reduced-motion fallback) and the buttons exist."""
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from quanara import registry
from quanara.features import default_features
from quanara.overview_facts import data_facts, model_facts
from quanara.training import TrainConfig, train
from quanara.ui.hero import hero_svg


def test_data_facts_equal_the_raw_files(data):
    f = data_facts(data)
    assert f["flight_rows"] == len(data["flight"])
    assert f["intl_rows"] == len(data["intl_train"])
    assert f["dom_days"] == len(data["dom_train"])
    assert f["hotel_days"] == data["intl_train"]["Date"].nunique()
    assert f["intl_nationalities"] == data["intl_train"]["Nationality"].nunique()
    assert f["flight_start"] == data["flight"]["Date"].min() and f["flight_end"] == data["flight"]["Date"].max()
    assert not f["test_files_have_guests"]
    assert np.isclose(sum(f["flight_monthly_pax"]), data["flight"]["Total PAX"].sum())
    assert -1 <= f["p2p7_guests_r"] <= 1


def test_model_card_follows_the_active_model(analytic, tmp_registry):
    assert model_facts() is None
    a = train(TrainConfig("international_total", "whatif", "ridge", default_features("international_total", "whatif"),
                          cv_folds=0, permutation_importance=False), analytic)
    registry.activate(a.model_id)
    m = model_facts()
    assert m["model_id"] == a.model_id and m["algorithm"] == "Ridge Regression"
    assert m["mae"] == a.metadata["metrics_holdout"]["MAE"] and m["r2"] == a.metadata["metrics_holdout"]["R2"]
    assert m["holdout"] == a.metadata["periods"]["holdout"]
    b = train(TrainConfig("international_total", "whatif", "random_forest", default_features("international_total", "whatif"),
                          params={"n_estimators": 40}, cv_folds=0, permutation_importance=False), analytic)
    registry.activate(b.model_id)
    assert model_facts()["model_id"] == b.model_id and model_facts()["algorithm"] == "Random Forest"


def test_hero_animates_and_respects_reduced_motion():
    svg = hero_svg()
    assert "<animateMotion" in svg and 'repeatCount="indefinite"' in svg
    assert 'class="ov-flight-static"' in svg and 'class="ov-flight-anim"' in svg
    from quanara.ui.design import CSS
    assert "prefers-reduced-motion" in CSS


@pytest.mark.skipif(not registry.list_models(), reason="reference models not trained")
def test_overview_renders_with_navigation_buttons():
    def page():
        from quanara.ui import overview
        overview.render()
    at = AppTest.from_function(page, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    labels = {b.label for b in at.button}
    assert {"Open the Simulator", "Judges' Demo"} <= labels
    html = " ".join(m.value for m in at.markdown)
    assert "At a glance" in html and "How it works" in html and "Challenge question" in html
    for claim in ("most international visitors", "nationality by nationality"):
        assert claim not in html  # removed unsupported/misleading wording
