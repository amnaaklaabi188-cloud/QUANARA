"""Section 4 — Correlation & Visualization Lab: statistics reproduced independently from the raw workbooks."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quanara import config
from quanara import correlation as C
from quanara.data_loading import discover_files


@pytest.fixture(scope="module")
def raw_daily():
    """Daily international Guests and daily P2P straight from the .xlsx files (no project loaders)."""
    f = discover_files()
    it = pd.read_excel(f["intl_train"], sheet_name="Export", engine="openpyxl")
    fl = pd.read_excel(f["flight"], sheet_name="Export", engine="openpyxl", usecols=["Date", "Total P2P"])
    fl["Date"] = pd.to_datetime(fl["Date"])
    fl = fl[fl["Date"] >= datetime(2023, 1, 1)]               # 2022 rows are monthly
    g = it.groupby(pd.to_datetime(it["Date"]))["Guests"].sum()
    cal = pd.date_range("2023-01-01", "2026-02-28")
    p2p = fl.groupby("Date")["Total P2P"].sum().reindex(cal, fill_value=0)
    return g, p2p


def test_headline_correlation_matches_raw_files(analytic, raw_daily):
    g, p2p = raw_daily
    p7 = p2p.rolling(7, min_periods=7).sum()
    win = pd.date_range(config.MODEL_START, config.HOTEL_TRAIN_END)
    x, y = p7.reindex(win), g.reindex(win)
    df = C.eda_frame(analytic, "International")
    d = df[df["Date"].between(config.MODEL_START, config.HOTEL_TRAIN_END) & df["Guests"].notna()]
    s = C.pair_stats(d["p2p_7d"], d["Guests"])
    assert s["n"] == len(win) == 884
    assert np.isclose(s["pearson"], np.corrcoef(x, y)[0, 1], atol=1e-12)
    slope, intercept = np.polyfit(x.to_numpy(float), y.to_numpy(float), 1)
    assert np.isclose(s["slope"], slope) and np.isclose(s["intercept"], intercept)
    assert np.isclose(s["r2"], s["pearson"] ** 2)
    assert round(s["pearson"], 3) == 0.784


def test_effective_sample_size_is_much_smaller_than_n(analytic):
    df = C.eda_frame(analytic, "International")
    d = df[df["Date"].between(config.MODEL_START, config.HOTEL_TRAIN_END)]
    s = C.pair_stats(d["p2p_7d"], d["Guests"])
    assert 3 <= s["n_eff"] < s["n"] / 10 and s["rho_x"] > 0.9


def test_market_frame_is_calendar_complete(analytic):
    hm = analytic.hotel_market
    tr = hm[(hm["split"] == "train") & hm["market_key"].isin(analytic.markets) & (hm["Date"] >= config.FLIGHT_DAILY_START)]
    gappy = (tr["Date"].nunique() - tr.groupby("market_key").size()).sort_values(ascending=False)
    market = gappy.index[0]
    m = C.eda_frame(analytic, "International — one source market", market)
    assert m["Date"].is_unique and len(m) == (config.FLIGHT_END - config.FLIGHT_DAILY_START).days + 1
    assert m["Date"].diff().dropna().dt.days.eq(1).all()
    win = m[m["Date"] <= config.HOTEL_TRAIN_END]
    assert int(win["Guests"].isna().sum()) == int(gappy.iloc[0])      # unobserved days stay missing; no row is dropped
    assert m.loc[m["Date"] > config.HOTEL_TRAIN_END, "Guests"].isna().all()   # unlabelled test period


def test_windows_never_show_partial_sums(analytic):
    df = C.eda_frame(analytic, "International")
    s = C.aviation_series(df, ["seats"], 7)
    assert s["seats"].iloc[:6].isna().all() and s["seats"].iloc[6] == df["seats"].iloc[:7].sum()


def test_lag_zero_equals_scatter_and_lags_shift_calendar_days(analytic):
    df = C.eda_frame(analytic, "International")
    lc = C.lag_correlation(df, "p2p", 7, "Guests", 3, "pearson", config.MODEL_START, config.HOTEL_TRAIN_END)
    d = df[df["Date"].between(config.MODEL_START, config.HOTEL_TRAIN_END)]
    assert np.isclose(lc.loc[0, "r"], C.pair_stats(d["p2p_7d"], d["Guests"])["pearson"])
    x = df["p2p"].rolling(7, min_periods=7).sum().shift(2)
    ok = df["Date"].between(config.MODEL_START, config.HOTEL_TRAIN_END)
    assert np.isclose(lc.loc[2, "r"], x[ok].corr(df.loc[ok, "Guests"]))


def test_investigations_are_real_properties(analytic):
    df = C.eda_frame(analytic, "International")
    d = df[df["Date"].between(config.MODEL_START, config.HOTEL_TRAIN_END)]
    fs = C.flights_vs_seats(d)
    assert fs["r"] > 0.99 and fs["gauge_cv"] < 0.05           # near-collinear because aircraft size is stable
    we = C.weekend_effect(d)
    assert abs(we["r"]) < 0.1 and np.isclose(we["r"], d["is_weekend"].corr(d["Guests"]))
    assert np.isclose(d[["p2p_share_7d", "transfer_share_7d"]].corr().iloc[0, 1], -1, atol=1e-3)   # deterministic mirror


def test_outlier_mask_flags_only_extremes():
    v = pd.DataFrame({"a": list(range(100)) + [10_000]})
    m = C.outlier_mask(v, ["a"])
    assert m.sum() == 1 and m.iloc[-1]


def test_guest_series_uses_training_files_only(analytic):
    gs = C.guests_series(analytic, "International")
    assert gs["Date"].min() == config.HOTEL_TRAIN_START and gs["Date"].max() == config.HOTEL_TRAIN_END
    assert gs["Guests"].notna().all() and gs["mean_7d"].iloc[:6].isna().all()
