"""Section 3 — Feature Engineering: facts, scenario sensitivity, matrix identity and the selection flow."""
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from quanara import config
from quanara.feature_facts import (feature_rows, leakage, matrix_columns, matrix_fingerprint, scenario_sensitivity, scenario_text,
                                    training_matrix)
from quanara.features import CATALOG_BY_NAME, available_features, default_features
from quanara.training import TrainConfig, train


@pytest.fixture(scope="module")
def moved(analytic):
    return scenario_sensitivity(analytic)


@pytest.mark.parametrize("scope", list(config.SCOPES))
@pytest.mark.parametrize("mode", list(config.MODES))
def test_missing_counts_come_from_the_training_rows(analytic, moved, scope, mode):
    tr = training_matrix(analytic, scope, mode)
    for r in feature_rows(analytic, scope, mode, moved):
        if r.name in tr:
            assert r.missing_n == int(tr[r.name].isna().sum())
            assert np.isclose(r.missing_pct, 100 * tr[r.name].isna().mean())
        assert r.available == (r.name in available_features(scope, mode))


def test_market_load_factor_missing_is_real(analytic, moved):
    r = next(x for x in feature_rows(analytic, "international_market", "whatif", moved) if x.name == "load_factor")
    tr = training_matrix(analytic, "international_market", "whatif")
    assert r.missing_n == int(tr["load_factor"].isna().sum()) > 0
    assert r.rows == len(tr) == 29172


def test_scenario_labels_are_measured(moved):
    assert scenario_text("seats_7d", moved)[0] == "Yes"
    assert scenario_text("load_factor", moved)[0] == "Yes"          # moved by the load-factor input
    assert scenario_text("business_seat_share_7d", moved)[0] == "Indirect"
    assert scenario_text("n_routes", moved)[0] == "No"
    assert scenario_text("guests_lag364", moved)[0] == "No"
    assert "seats" in moved["seats_7d"] and "load_factor" in moved["p2p_7d"]


def test_leakage_labels():
    assert leakage("new_arrivals", "whatif")[0] == "Not allowed"
    assert leakage("new_arrivals", "competition")[0] == "High"
    assert leakage("guests_lag364", "whatif")[0] == "Low"
    assert leakage("doy_sin", "whatif")[0] == "None"
    assert leakage("seats_7d", "whatif")[0] == "None" and leakage("p2p_7d", "whatif")[0] == "Low"
    for f in available_features("international_total", "whatif"):
        assert CATALOG_BY_NAME[f].group != "competition"


def test_preview_matrix_is_the_training_matrix(analytic, tmp_registry):
    feats = default_features("international_total", "whatif")
    tr = training_matrix(analytic, "international_total", "whatif")
    fp = matrix_fingerprint(tr[matrix_columns("international_total", feats)])
    r = train(TrainConfig("international_total", "whatif", "ridge", feats, cv_folds=0, permutation_importance=False), analytic, save=False)
    assert r.metadata["matrix_fingerprint"] == fp
    assert r.metadata["n_rows"]["modelling"] == len(tr)


def _render():
    from quanara.ui.common import inject_css
    inject_css()
    from quanara.ui import features_page
    features_page.render()


def test_selection_apply_and_reset_flow():
    at = AppTest.from_function(_render, default_timeout=240)
    at.run()
    assert not at.exception
    sel = lambda: at.session_state["feature_selection"]["international_total|whatif"] if "feature_selection" in at.session_state and "international_total|whatif" in at.session_state["feature_selection"] else default_features("international_total", "whatif")
    seats = next(c for c in at.checkbox if c.key == "fe_cb_international_total_whatif_seats")
    assert not seats.value
    seats.check().run()
    assert "seats" not in sel()                     # ticking only changes the draft
    [b for b in at.button if b.key == "fe_apply"][0].click().run()
    assert "seats" in sel()                         # Apply hands it to the Training Studio
    [b for b in at.button if b.key == "fe_reset"][0].click().run()
    assert at.session_state["fe_draft"]["international_total|whatif"] == default_features("international_total", "whatif")
    assert "seats" in sel()                         # Reset restores the draft only, until applied
    at.radio(key="fe_group_pick").set_value("competition").run()
    hotel = [c for c in at.checkbox if c.key and c.key.startswith("fe_cb_")]
    assert len(hotel) == 3 and all(c.disabled for c in hotel)   # same-day hotel observations are locked in What-if mode
    at.radio(key="fe_mode").set_value("competition").run()
    assert all(not c.disabled for c in at.checkbox if c.key and c.key.startswith("fe_cb_"))
