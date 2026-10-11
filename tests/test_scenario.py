"""What-if engine: consistency, no double counting, no silent no-ops, and rejection of invalid input."""
import numpy as np
import pytest

from quanara import registry
from quanara import scenario as S
from quanara.features import default_features
from quanara.training import TrainConfig, train


@pytest.fixture()
def market_model(analytic, tmp_registry):
    r = train(TrainConfig("international_market", "whatif", "ridge", default_features("international_market", "whatif"), cv_folds=0,
                          permutation_importance=False), analytic)
    registry.activate(r.model_id)
    return r.model_id


def _cfg(mid, **kw):
    base = dict(model_id=mid, market="INDIA", start="2025-03-01", end="2025-03-31", seats_change_pct=0.0)
    return S.ScenarioConfig(**{**base, **kw})


def test_zero_change_gives_identical_predictions(analytic, market_model):
    r = S.run(_cfg(market_model), analytic, save=False)
    assert np.allclose(r.daily["baseline"], r.daily["scenario"])
    assert r.summary["pct_change"] == pytest.approx(0)


def test_zero_change_leaves_every_route_day_unchanged(analytic):
    """All markets, over a window that contains a route-day with transit passengers (Jordan, 19 Aug 2025 is in the
    56-day look-back): an unchanged scenario must reproduce every recorded value exactly, so baseline = scenario."""
    rows = analytic.flight_daily_rows
    new, mask = S.apply_scenario(rows, S.ScenarioConfig(model_id="none", market="ALL", start="2025-09-01", end="2025-09-30"))
    assert (rows.loc[mask, "Total Transit"] > 0).any()
    for c in ("Total Seats", "Total PAX", "Total P2P", "Total Transfer", "Total Transit", "flights", "Load Factor"):
        assert np.array_equal(new[c].astype(float).to_numpy(), rows[c].astype(float).to_numpy(), equal_nan=True), c


def test_p2p_share_is_a_share_of_all_passengers_on_transit_rows(analytic):
    rows = analytic.flight_daily_rows
    new, mask = S.apply_scenario(rows, S.ScenarioConfig(model_id="none", market="ALL", start="2025-09-01", end="2025-09-30", seats_change_pct=10))
    t = mask & (rows["Total Transit"] > 0)
    assert t.any()
    assert np.allclose(new.loc[t, "Total P2P"] / new.loc[t, "Total PAX"], rows.loc[t, "Total P2P"] / rows.loc[t, "Total PAX"])
    assert np.allclose(new.loc[t, "Total PAX"], new.loc[t, ["Total P2P", "Total Transfer", "Total Transit"]].sum(axis=1))


def test_seat_change_recomputes_dependent_quantities(analytic, market_model):
    rows = analytic.flight_daily_rows
    new, mask = S.apply_scenario(rows, _cfg(market_model, seats_change_pct=20))
    old = rows[mask]
    chg = new[mask]
    assert np.allclose(chg["Total Seats"], old["Total Seats"] * 1.2)
    assert np.allclose(chg["Total PAX"], old["Total PAX"] * 1.2)
    assert np.allclose(chg["Total P2P"], old["Total P2P"] * 1.2)
    assert np.allclose(chg["Total PAX"], chg["Total P2P"] + chg["Total Transfer"] + chg["Total Transit"])
    assert np.allclose(chg["flights"], old["flights"] * 1.2)
    assert (new.loc[~mask, "Total Seats"].to_numpy() == rows.loc[~mask, "Total Seats"].to_numpy()).all()


def test_frequency_and_seats_are_not_double_counted(analytic, market_model):
    rows = analytic.flight_daily_rows
    cfg = _cfg(market_model, seats_change_pct=20, frequency_mode="independent", frequency_change_pct=50)
    new, mask = S.apply_scenario(rows, cfg)
    assert np.isclose(new.loc[mask, "Total Seats"].sum(), rows.loc[mask, "Total Seats"].sum() * 1.2)
    assert np.isclose(new.loc[mask, "flights"].sum(), rows.loc[mask, "flights"].sum() * 1.5)
    assert cfg.seats_per_flight_change_pct() == pytest.approx((1.2 / 1.5 - 1) * 100)


def test_load_factor_and_p2p_share_assumptions(analytic, market_model):
    rows = analytic.flight_daily_rows
    new, mask = S.apply_scenario(rows, _cfg(market_model, load_factor_mode="set", load_factor_value=0.8, p2p_share_mode="set", p2p_share_value=0.5))
    x = new[mask]
    assert np.allclose(x["Total PAX"] / x["Total Seats"], 0.8)
    assert np.allclose(x["Total P2P"] / x["Total PAX"], 0.5)


def test_scenario_changes_prediction_and_features(analytic, market_model):
    r = S.run(_cfg(market_model, seats_change_pct=20), analytic, save=False)
    assert r.summary["pct_change"] != 0
    assert r.feature_changes["changed"].any()
    assert all(c["passed"] for c in r.checks[:5])
    # other markets unchanged in the market model
    mc = r.market_contrib.set_index("market_key")
    assert np.isclose(mc.drop(index="INDIA")["change"].abs().max(), 0)


def test_invalid_inputs_rejected(analytic, market_model):
    with pytest.raises(S.ScenarioError):
        S.run(_cfg(market_model, start="2022-06-01", end="2022-06-30"), analytic, save=False)
    with pytest.raises(S.ScenarioError):
        S.run(_cfg(market_model, load_factor_mode="set", load_factor_value=2.0), analytic, save=False)
    with pytest.raises(S.ScenarioError):
        S.run(_cfg(market_model, market="AUSTRALIA"), analytic, save=False)


def test_competition_models_cannot_drive_simulator(analytic, tmp_registry):
    r = train(TrainConfig("international_total", "competition", "ridge", default_features("international_total", "competition"), cv_folds=0,
                          permutation_importance=False), analytic)
    with pytest.raises(ValueError):
        registry.activate(r.model_id)
    with pytest.raises(S.ScenarioError):
        S.run(S.ScenarioConfig(model_id=r.model_id, market="INDIA", start="2025-03-01", end="2025-03-31"), analytic, save=False)


def test_scenario_saved_with_assumptions(analytic, market_model, tmp_registry):
    r = S.run(_cfg(market_model, seats_change_pct=10), analytic, save=True)
    d = tmp_registry / "scenarios" / r.scenario_id
    assert (d / "assumptions.json").exists() and (d / "results.csv").exists() and (d / "daily_predictions.csv").exists()
    assert "not a causal estimate" in S.judges_summary(r)


def test_copilot_answers_are_grounded(analytic, market_model):
    from quanara import copilot
    r = S.run(_cfg(market_model, seats_change_pct=20), analytic, save=False)
    intent, text = copilot.answer("Why did the prediction change?", market_model, r)
    assert intent == "why_change" and f"{r.summary['pct_change']:+.2f}%" in text and r.scenario_id not in text  # no internal IDs in answers
    assert copilot.route("what does this mean for tourism planning?") == "planning"
    plan = copilot.answer_planning(r, market_model)
    assert "guest-days" in plan and "not guest nights" in plan
    lim = copilot.answer_limitations(copilot.limitation_facts(analytic))
    assert "withheld" not in lim and f"{len(analytic.markets)} of" in lim   # numbers come from the data, no unconfirmed suppression claim
    assert copilot.route("how reliable is this forecast?") == "reliability"
    assert copilot.route("what are the limitations") == "limitations"
    ctx = copilot.grounded_context(market_model, r)
    assert "scenario" in ctx and "model" in ctx
