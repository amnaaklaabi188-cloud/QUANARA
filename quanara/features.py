"""Feature catalogue and feature construction.

The same functions build features for training, evaluation, competition prediction and the
What-if simulator, so a scenario can never be scored with features constructed differently
from those the model was trained on.

Aviation aggregation rules
  * additive quantities (seats, passengers, flights, cabin seats) are SUMMED;
  * ratios are recomputed from the sums (load factor = ΣPAX / ΣSeats, P2P share = ΣP2P / ΣPAX),
    never averaged across routes;
  * a market-day with no flight rows has zero seats (the file lists operated route-days only);
    ratios on zero-seat days are NaN, not zero.
All windows are trailing (they end on day t), so no feature looks into the future.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import config

ETIHAD = "Etihad Airways"

# ------------------------------------------------------------------------------------
# Catalogue
# ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    group: str            # aviation | time | historical | derived | competition | market
    definition: str
    source: str
    modes: tuple = ("whatif", "competition")
    scopes: tuple = ("international_total", "international_market", "domestic", "overall")
    kind: str = "numeric"  # numeric | categorical
    leakage: str = "none"
    scenario_affected: bool = False
    default: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["modes"] = ", ".join(self.modes)
        d["scopes"] = ", ".join(self.scopes)
        return d


_ALL = ("international_total", "international_market", "domestic", "overall")
_TOTALS = ("international_total", "domestic", "overall")

CATALOG: list[FeatureSpec] = [
    # --- aviation, same day ---------------------------------------------------------
    FeatureSpec("seats", "Total Seats (day)", "aviation", "Σ Total Seats on arrivals into AUH on day t (market or all markets).", "Total Seats", scenario_affected=True, default=False),
    FeatureSpec("p2p", "Total P2P (day)", "aviation", "Σ Total P2P passengers on day t — journeys terminating in Abu Dhabi.", "Total P2P", scenario_affected=True),
    FeatureSpec("pax", "Total PAX (day)", "aviation", "Σ Total PAX on day t (P2P + transfer + transit).", "Total PAX", scenario_affected=True, default=False),
    FeatureSpec("transfer", "Total Transfer (day)", "aviation", "Σ Total Transfer passengers on day t (connect onward at AUH).", "Total Transfer", scenario_affected=True, default=False),
    FeatureSpec("flights", "Flights (day)", "aviation", "Σ daily flights decoded from Average Weekly Frequency (AWF × days_in_month / 7; exact on every daily row).", "Average Weekly Frequency", scenario_affected=True, default=False),
    FeatureSpec("load_factor", "Load Factor (day)", "aviation", "ΣPAX / ΣSeats on day t (recomputed from sums, never averaged).", "Total PAX, Total Seats", scenario_affected=True, default=False),
    # --- aviation, trailing windows -------------------------------------------------
    FeatureSpec("seats_7d", "Total Seats (trailing 7 days)", "aviation", "Σ seats over days t-6…t. Hotel Guests are an in-house stock fed by arrivals over the previous days.", "Total Seats", scenario_affected=True),
    FeatureSpec("p2p_7d", "Total P2P (trailing 7 days)", "aviation", "Σ P2P passengers over days t-6…t.", "Total P2P", scenario_affected=True),
    FeatureSpec("pax_7d", "Total PAX (trailing 7 days)", "aviation", "Σ PAX over days t-6…t.", "Total PAX", scenario_affected=True, default=False),
    FeatureSpec("transfer_7d", "Total Transfer (trailing 7 days)", "aviation", "Σ transfer passengers over days t-6…t.", "Total Transfer", scenario_affected=True),
    FeatureSpec("flights_7d", "Flights (trailing 7 days)", "aviation", "Σ decoded flights over days t-6…t.", "Average Weekly Frequency", scenario_affected=True),
    FeatureSpec("load_factor_7d", "Load Factor (trailing 7 days)", "aviation", "ΣPAX / ΣSeats over days t-6…t.", "Total PAX, Total Seats", scenario_affected=True),
    FeatureSpec("seats_28d", "Total Seats (trailing 28 days)", "aviation", "Σ seats over days t-27…t.", "Total Seats", scenario_affected=True, default=False),
    FeatureSpec("p2p_28d", "Total P2P (trailing 28 days)", "aviation", "Σ P2P over days t-27…t.", "Total P2P", scenario_affected=True),
    # --- derived ---------------------------------------------------------------------
    FeatureSpec("p2p_share_7d", "P2P Share (7 days)", "derived", "ΣP2P / ΣPAX over days t-6…t — the share of arriving passengers whose journey ends in Abu Dhabi.", "Total P2P, Total PAX", scenario_affected=True),
    FeatureSpec("transfer_share_7d", "Transfer Share (7 days)", "derived", "ΣTransfer / ΣPAX over days t-6…t. ≈ 1 − P2P share because Total Transit is zero on all but 3 daily rows.", "Total Transfer, Total PAX", scenario_affected=True, default=False),
    FeatureSpec("seats_per_flight_7d", "Seats per Flight (7 days)", "derived", "Σseats / Σflights over days t-6…t (average aircraft gauge).", "Total Seats, Average Weekly Frequency", scenario_affected=True, default=False),
    FeatureSpec("business_seat_share_7d", "Business Seat Share (7 days)", "derived", "ΣBusiness Class Seat Capacity / ΣTotal Seats over t-6…t (cabin mix; exact on daily rows).", "Business Class Seat Capacity", scenario_affected=False, default=False),
    FeatureSpec("etihad_seat_share_7d", "Etihad Seat Share (7 days)", "derived", "Share of seats over t-6…t operated by Etihad Airways (derived from Airline Name).", "Airline Name, Total Seats", scenario_affected=False, default=False),
    FeatureSpec("n_routes", "Active Routes (day)", "derived", "Number of distinct departure-city × airline routes operating on day t.", "Departure City, Airline Name", scenario_affected=False, default=False),
    FeatureSpec("n_cities", "Active Departure Cities (day)", "derived", "Number of distinct departure cities with an arrival on day t.", "Departure City", scenario_affected=False, default=False),
    FeatureSpec("n_airlines", "Active Airlines (day)", "derived", "Number of distinct airlines operating on day t.", "Airline Name", scenario_affected=False, default=False),
    FeatureSpec("seats_growth_28d", "Seats Growth (28 d vs prior 28 d)", "derived", "seats_28d(t) / seats_28d(t-28) − 1. YoY growth is impossible: 2022 flights are monthly.", "Total Seats", scenario_affected=True),
    FeatureSpec("p2p_growth_28d", "P2P Growth (28 d vs prior 28 d)", "derived", "p2p_28d(t) / p2p_28d(t-28) − 1.", "Total P2P", scenario_affected=True),
    # --- historical aviation ------------------------------------------------------------
    FeatureSpec("seats_prev7d", "Previous Total Seats (prior week)", "historical", "Σ seats over days t-13…t-7 (the week before the trailing week).", "Total Seats", scenario_affected=True, default=False),
    FeatureSpec("p2p_prev7d", "Previous Total P2P (prior week)", "historical", "Σ P2P over days t-13…t-7.", "Total P2P", scenario_affected=True, default=False),
    # --- historical guests ----------------------------------------------------------------
    FeatureSpec("guests_lag364", "Previous Guests (same weekday last year)", "historical", "Observed Guests on day t-364. Known for every date up to 2026-07-30, including the whole test period, so it is valid in both modes.", "Guests (train files)", leakage="low — 364-day lag is always in the observed past"),
    FeatureSpec("guests_lag364_7d", "Previous Guests rolling mean (last year, 7 d)", "historical", "Mean observed Guests over days t-370…t-364.", "Guests (train files)", leakage="low — 364-day lag"),
    # --- time --------------------------------------------------------------------------
    FeatureSpec("dow", "Day of Week", "time", "0 = Monday … 6 = Sunday.", "Date", kind="categorical"),
    FeatureSpec("is_weekend", "Weekend Indicator", "time", "1 on Saturday/Sunday (UAE official weekend since 1 Jan 2022).", "Date"),
    FeatureSpec("month", "Month", "time", "Calendar month 1–12.", "Date"),
    FeatureSpec("quarter", "Quarter", "time", "Calendar quarter 1–4.", "Date", default=False),
    FeatureSpec("year", "Year", "time", "Calendar year. Off by default: a linear model would extrapolate the trend into the test period.", "Date", default=False),
    FeatureSpec("doy_sin", "Seasonal encoding (sin)", "time", "sin(2π·day_of_year/365.25).", "Date"),
    FeatureSpec("doy_cos", "Seasonal encoding (cos)", "time", "cos(2π·day_of_year/365.25).", "Date"),
    # --- market ------------------------------------------------------------------------
    FeatureSpec("market", "Source Market (Nationality ↔ Departure Country)", "market", "Guest nationality whose name matches a flight Departure Country Name (33 markets).", "Nationality, Departure Country Name", scopes=("international_market",), kind="categorical"),
    # --- competition-only hotel observations ------------------------------------------------
    FeatureSpec("new_arrivals", "New Arrivals (day)", "competition", "Observed check-ins on day t (in the official test files). Accounting component of Guests.", "New Arrivals", modes=("competition",), leakage="HIGH in What-if (unknown future hotel data) — competition mode only"),
    FeatureSpec("new_arrivals_7d", "New Arrivals (trailing 7 days)", "competition", "Σ New Arrivals over days t-6…t; train and test files are concatenated so the window is complete on the first test days.", "New Arrivals", modes=("competition",), leakage="HIGH in What-if — competition mode only"),
    FeatureSpec("same_day_guests", "Same-Day Guests (day)", "competition", "Observed same-day guests; '*' (empty) counted as 0, i.e. a lower bound.", "Same-Day Guests", modes=("competition",), leakage="HIGH in What-if — competition mode only", default=False),
]

CATALOG_BY_NAME = {f.name: f for f in CATALOG}

# Short guest lags are deliberately NOT offered; listed here so the UI can explain why.
REJECTED_FEATURES = [
    {"name": "guests_lag1 / guests_lag7", "reason": "Guests are not observed after 2025-07-31, so these lags are unknown for the whole test period and for any What-if date beyond the training data; using them would make validation scores unrepresentative."},
    {"name": "Departure City / Airline Name (raw)", "reason": "Many cities and airlines arrive each day, so a single categorical value per hotel-day does not exist. They enter as active-route/city/airline counts and Etihad seat share."},
    {"name": "Total Transit, transit shares", "reason": "Total Transit is non-zero on only 3 of 116,395 daily rows (164 passengers) — no usable variance."},
    {"name": "Cabin-class P2P / transfer / transit counts", "reason": "Do not sum to the corresponding totals on 28–72% of rows (infants not assigned a cabin; transit cabins inconsistent with Total Transit = 0). Totals are used instead."},
    {"name": "Year-over-year aviation growth", "reason": "2022 flight data is monthly, so a daily 364-day aviation lag is undefined for 2023."},
    {"name": "Public holidays / Ramadan", "reason": "No holiday calendar is included in the supplied files and external data is not allowed. Seasonality is captured by time encodings and the 364-day guest lag."},
]


# Shape constraint (optional, on by default for tree models): holding every other input fixed,
# more arriving seats / passengers / flights must not LOWER predicted guests. Ratios, growth
# rates, time and history are unconstrained.
MONOTONE_INCREASING = {
    "seats", "p2p", "pax", "transfer", "flights", "seats_7d", "p2p_7d", "pax_7d", "transfer_7d", "flights_7d",
    "seats_28d", "p2p_28d", "seats_prev7d", "p2p_prev7d",
}


def catalog_frame() -> pd.DataFrame:
    return pd.DataFrame([f.to_dict() for f in CATALOG])


def available_features(scope: str, mode: str) -> list[str]:
    return [f.name for f in CATALOG if scope in f.scopes and mode in f.modes]


def default_features(scope: str, mode: str) -> list[str]:
    return [f.name for f in CATALOG if scope in f.scopes and mode in f.modes and f.default]


ABLATION_GROUPS = {
    "Aviation only": ("aviation", "derived"),
    "Aviation + Seasonality": ("aviation", "derived", "time", "market"),
    "Aviation + Seasonality + Historical": ("aviation", "derived", "time", "market", "historical"),
}


def ablation_features(base: list[str], groups: tuple) -> list[str]:
    return [f for f in base if CATALOG_BY_NAME[f].group in groups]


def categorical_features(features: list[str]) -> list[str]:
    return [f for f in features if CATALOG_BY_NAME[f].kind == "categorical"]


def numeric_features(features: list[str]) -> list[str]:
    return [f for f in features if CATALOG_BY_NAME[f].kind == "numeric"]


# ------------------------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------------------------

AVIATION_SUM_COLS = {
    "Total Seats": "seats", "Total P2P": "p2p", "Total PAX": "pax", "Total Transfer": "transfer",
    "Total Transit": "transit", "flights": "flights",
    "Business Class Seat Capacity": "business_seats", "First Class Seat Capacity": "first_seats",
}


def daily_flight_rows(flight: pd.DataFrame) -> pd.DataFrame:
    """Daily-grain flight rows only (2022 monthly rows cannot be placed on a day)."""
    return flight[flight["grain"] == "daily"].copy()


def aggregate_aviation(rows: pd.DataFrame, by_market: bool, markets: list[str] | None = None,
                       start: pd.Timestamp = config.FLIGHT_DAILY_START, end: pd.Timestamp = config.FLIGHT_END) -> pd.DataFrame:
    """Sum route-day rows to day (or day × market) on a complete calendar."""
    r = rows.copy()
    r["etihad_seats"] = np.where(r["Airline Name"].eq(ETIHAD), r["Total Seats"], 0.0)
    r["route"] = r["Departure City"] + "|" + r["Airline Name"]
    keys = ["Date", "market_key"] if by_market else ["Date"]
    sums = r.groupby(keys)[list(AVIATION_SUM_COLS) + ["etihad_seats"]].sum().rename(columns=AVIATION_SUM_COLS)
    counts = r.groupby(keys).agg(n_routes=("route", "nunique"), n_cities=("Departure City", "nunique"), n_airlines=("Airline Name", "nunique"))
    agg = sums.join(counts)
    dates = pd.date_range(start, end, freq="D", name="Date")
    if by_market:
        mk = sorted(markets) if markets is not None else sorted(r["market_key"].unique())
        idx = pd.MultiIndex.from_product([dates, mk], names=["Date", "market_key"])
    else:
        idx = dates
    agg = agg.reindex(idx).fillna(0.0)  # absent route-day = no operated flight
    return agg.reset_index()


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.where(den > 0)).astype(float)


def aviation_features(agg: pd.DataFrame, by_market: bool) -> pd.DataFrame:
    """Trailing-window aviation features from a complete daily aggregate."""
    df = agg.sort_values(["market_key", "Date"] if by_market else ["Date"]).reset_index(drop=True)
    grp = df.groupby("market_key", sort=False) if by_market else None

    def roll(col: str, w: int) -> pd.Series:
        if grp is not None:
            return grp[col].transform(lambda s: s.rolling(w, min_periods=w).sum())
        return df[col].rolling(w, min_periods=w).sum()

    def shift(s: pd.Series, k: int) -> pd.Series:
        if grp is not None:
            return s.groupby(df["market_key"], sort=False).shift(k)
        return s.shift(k)

    out = df[["Date"] + (["market_key"] if by_market else [])].copy()
    for c in ("seats", "p2p", "pax", "transfer", "flights", "n_routes", "n_cities", "n_airlines"):
        out[c] = df[c]
    out["load_factor"] = _ratio(df["pax"], df["seats"])
    s7 = {c: roll(c, 7) for c in ("seats", "p2p", "pax", "transfer", "flights", "business_seats", "etihad_seats")}
    out["seats_7d"], out["p2p_7d"], out["pax_7d"] = s7["seats"], s7["p2p"], s7["pax"]
    out["transfer_7d"], out["flights_7d"] = s7["transfer"], s7["flights"]
    out["load_factor_7d"] = _ratio(s7["pax"], s7["seats"])
    out["p2p_share_7d"] = _ratio(s7["p2p"], s7["pax"])
    out["transfer_share_7d"] = _ratio(s7["transfer"], s7["pax"])
    out["seats_per_flight_7d"] = _ratio(s7["seats"], s7["flights"])
    out["business_seat_share_7d"] = _ratio(s7["business_seats"], s7["seats"])
    out["etihad_seat_share_7d"] = _ratio(s7["etihad_seats"], s7["seats"])
    out["seats_28d"], out["p2p_28d"] = roll("seats", 28), roll("p2p", 28)
    out["seats_growth_28d"] = _ratio(out["seats_28d"], shift(out["seats_28d"], 28)) - 1
    out["p2p_growth_28d"] = _ratio(out["p2p_28d"], shift(out["p2p_28d"], 28)) - 1
    out["seats_prev7d"] = shift(out["seats_7d"], 7)
    out["p2p_prev7d"] = shift(out["p2p_7d"], 7)
    return out


def time_features(dates: pd.Series) -> pd.DataFrame:
    d = pd.to_datetime(dates)
    doy = d.dt.dayofyear
    return pd.DataFrame({
        "dow": d.dt.dayofweek.astype(int),
        "is_weekend": d.dt.dayofweek.isin([5, 6]).astype(int),
        "month": d.dt.month.astype(int),
        "quarter": d.dt.quarter.astype(int),
        "year": d.dt.year.astype(int),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }, index=dates.index)


def guest_history_features(target: pd.DataFrame, by_market: bool, value_col: str = "Guests") -> pd.DataFrame:
    """364-day lag features from the observed target history (calendar-complete)."""
    keys = ["Date", "market_key"] if by_market else ["Date"]
    t = target[keys + [value_col]].copy()
    start, end = t["Date"].min(), config.FLIGHT_END + pd.Timedelta(days=config.SEASONAL_LAG_DAYS + 7)
    dates = pd.date_range(start, end, freq="D", name="Date")
    if by_market:
        idx = pd.MultiIndex.from_product([dates, sorted(t["market_key"].unique())], names=keys)
        s = t.set_index(keys)[value_col].reindex(idx)  # withheld rows stay NaN (not 0)
        s = s.reset_index().sort_values(["market_key", "Date"])
        g = s.groupby("market_key", sort=False)[value_col]
        s["guests_lag364"] = g.shift(config.SEASONAL_LAG_DAYS)
        s["guests_lag364_7d"] = g.transform(lambda x: x.shift(config.SEASONAL_LAG_DAYS).rolling(7, min_periods=4).mean())
    else:
        s = t.set_index("Date")[value_col].reindex(dates).reset_index()
        s["guests_lag364"] = s[value_col].shift(config.SEASONAL_LAG_DAYS)
        s["guests_lag364_7d"] = s[value_col].shift(config.SEASONAL_LAG_DAYS).rolling(7, min_periods=4).mean()
    return s[keys + ["guests_lag364", "guests_lag364_7d"]]


def build_feature_frame(av_feats: pd.DataFrame, hist: pd.DataFrame, by_market: bool,
                        dates: pd.DatetimeIndex | None = None, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join aviation, time, guest-history and optional competition features on the key."""
    keys = ["Date", "market_key"] if by_market else ["Date"]
    df = av_feats
    if dates is not None:
        df = df[df["Date"].isin(dates)]
    n0 = len(df)
    df = df.merge(hist, on=keys, how="left", validate="1:1")
    if extra is not None:
        df = df.merge(extra, on=keys, how="left", validate="1:1")
    assert len(df) == n0, "feature join changed the row count"
    df = pd.concat([df.reset_index(drop=True), time_features(df["Date"]).reset_index(drop=True)], axis=1)
    if by_market:
        df["market"] = df["market_key"]
    return df
