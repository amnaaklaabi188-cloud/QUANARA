"""Analytical dataset integrity and feature construction (no look-ahead, correct aggregation)."""
import numpy as np
import pandas as pd

from quanara import config
from quanara.features import aggregate_aviation, available_features, aviation_features
from quanara.integration import model_frame, training_rows


def test_aggregation_preserves_totals(analytic):
    rows = analytic.flight_daily_rows
    t = analytic.av_total
    assert np.isclose(t["seats"].sum(), rows["Total Seats"].sum())
    assert np.isclose(t["p2p"].sum(), rows["Total P2P"].sum())
    m = analytic.av_market
    matched = rows[rows["market_key"].isin(analytic.markets)]
    assert np.isclose(m["p2p"].sum(), matched["Total P2P"].sum())


def test_load_factor_is_ratio_of_sums_not_mean(analytic):
    rows = analytic.flight_daily_rows
    day = rows[rows["Date"] == "2024-06-15"]
    expected = day["Total PAX"].sum() / day["Total Seats"].sum()
    got = analytic.av_total.loc[analytic.av_total["Date"] == "2024-06-15", "load_factor"].item()
    assert np.isclose(got, expected)
    assert not np.isclose(got, day["Load Factor"].mean())


def test_markets_and_matching(analytic):
    assert len(analytic.markets) == 33
    assert len(analytic.unmatched_nationalities) == 12
    assert "AUSTRALIA" in analytic.unmatched_nationalities


def test_hotel_total_equals_sum_of_rows(analytic, data):
    it = data["intl_train"]
    d = "2024-03-10"
    assert analytic.hotel_total.loc[analytic.hotel_total["Date"] == d, "intl_guests"].item() == it.loc[it["Date"] == d, "Guests"].sum()


def test_no_row_multiplication_and_modelling_window(analytic):
    for scope in config.SCOPES:
        for mode in config.MODES:
            f = model_frame(analytic, scope, mode)
            keys = ["Date", "market_key"] if scope == "international_market" else ["Date"]
            assert not f.duplicated(keys).any()
            tr = training_rows(f)
            assert tr["Date"].min() == config.MODEL_START and tr["Date"].max() == config.MODEL_END
            assert tr["y"].notna().all()
            assert (f.loc[f["split"] == "test", "y"].isna()).all()


def test_whatif_mode_excludes_hotel_observations():
    for scope in config.SCOPES:
        wi = available_features(scope, "whatif")
        assert not {"new_arrivals", "new_arrivals_7d", "same_day_guests"} & set(wi)
        assert "new_arrivals" in available_features(scope, "competition")


def test_trailing_features_do_not_look_ahead(analytic):
    rows = analytic.flight_daily_rows
    base = aviation_features(aggregate_aviation(rows, by_market=False), by_market=False).set_index("Date")
    future = rows.copy()
    future.loc[future["Date"] >= "2024-07-01", "Total Seats"] *= 10
    changed = aviation_features(aggregate_aviation(future, by_market=False), by_market=False).set_index("Date")
    past = base.index < "2024-07-01"
    pd.testing.assert_frame_equal(base[past], changed[past])
    assert changed.loc["2024-07-01", "seats_7d"] > base.loc["2024-07-01", "seats_7d"]


def test_guest_lag_is_364_days(analytic):
    f = model_frame(analytic, "international_total", "whatif").set_index("Date")
    ht = analytic.hotel_total.set_index("Date")
    assert f.loc["2024-05-20", "guests_lag364"] == ht.loc[pd.Timestamp("2024-05-20") - pd.Timedelta(days=364), "intl_guests"]
