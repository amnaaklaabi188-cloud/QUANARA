"""Facts behind the Feature Engineering page, computed from the analytical dataset.

Everything here is derived from the same objects the training code uses (`model_frame`,
`training_rows`, the feature catalogue and the scenario engine), so the page can never
describe a feature differently from how it is built:

  * missing counts are counted on the exact training rows of the chosen scope and mode;
  * "changes in scenarios" is measured by running the scenario engine (+20 % seats) and
    checking which recomputed features actually move;
  * the matrix fingerprint is a hash of the matrix the ML Training Studio fits on.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .features import (CATALOG, CATALOG_BY_NAME, MONOTONE_INCREASING, aggregate_aviation, available_features,
                       aviation_features, default_features)
from .integration import Analytic, model_frame, training_rows

GROUP_ORDER = ["aviation", "time", "derived", "historical", "market", "competition"]
GROUP_LABEL = {"aviation": "Aviation", "time": "Time", "derived": "Derived", "historical": "Historical", "market": "Market",
               "competition": "Hotel (competition)"}

# ------------------------------------------------------------------ documentation

# base quantity of every aviation-derived feature → (unit, what it measures, when it is known)
_QTY = {
    "seats": ("seats", "seat capacity offered on arriving flights", "schedule"),
    "flights": ("flights", "arriving flights, decoded from Average Weekly Frequency", "schedule"),
    "p2p": ("passengers", "point-to-point passengers — journeys that end in Abu Dhabi", "realised"),
    "pax": ("passengers", "all arriving passengers (P2P + transfer + transit)", "realised"),
    "transfer": ("passengers", "transfer passengers who connect onward at AUH", "realised"),
}

_FORMULA = {
    "seats": "Seats(t) = Σ routes Total Seats(t)",
    "p2p": "P2P(t) = Σ routes Total P2P(t)",
    "pax": "PAX(t) = Σ routes Total PAX(t)",
    "transfer": "Transfer(t) = Σ routes Total Transfer(t)",
    "flights": "Flights(t) = Σ routes AWF × days_in_month ÷ 7",
    "load_factor": "LF(t) = Σ PAX(t) ÷ Σ Seats(t)",
    "seats_7d": "Seats₇(t) = Σ_{d=t−6…t} Seats(d)",
    "p2p_7d": "P2P₇(t) = Σ_{d=t−6…t} P2P(d)",
    "pax_7d": "PAX₇(t) = Σ_{d=t−6…t} PAX(d)",
    "transfer_7d": "Transfer₇(t) = Σ_{d=t−6…t} Transfer(d)",
    "flights_7d": "Flights₇(t) = Σ_{d=t−6…t} Flights(d)",
    "load_factor_7d": "LF₇(t) = Σ_{t−6…t} PAX ÷ Σ_{t−6…t} Seats",
    "seats_28d": "Seats₂₈(t) = Σ_{d=t−27…t} Seats(d)",
    "p2p_28d": "P2P₂₈(t) = Σ_{d=t−27…t} P2P(d)",
    "p2p_share_7d": "Share₇(t) = Σ_{t−6…t} P2P ÷ Σ_{t−6…t} PAX",
    "transfer_share_7d": "TShare₇(t) = Σ_{t−6…t} Transfer ÷ Σ_{t−6…t} PAX",
    "seats_per_flight_7d": "Gauge₇(t) = Σ_{t−6…t} Seats ÷ Σ_{t−6…t} Flights",
    "business_seat_share_7d": "Biz₇(t) = Σ_{t−6…t} Business seats ÷ Σ_{t−6…t} Seats",
    "etihad_seat_share_7d": "EY₇(t) = Σ_{t−6…t} Etihad seats ÷ Σ_{t−6…t} Seats",
    "n_routes": "Routes(t) = #{departure city × airline with a row on t}",
    "n_cities": "Cities(t) = #{departure cities with a row on t}",
    "n_airlines": "Airlines(t) = #{airlines with a row on t}",
    "seats_growth_28d": "g(t) = Seats₂₈(t) ÷ Seats₂₈(t−28) − 1",
    "p2p_growth_28d": "g(t) = P2P₂₈(t) ÷ P2P₂₈(t−28) − 1",
    "seats_prev7d": "Seats₇(t−7) = Σ_{d=t−13…t−7} Seats(d)",
    "p2p_prev7d": "P2P₇(t−7) = Σ_{d=t−13…t−7} P2P(d)",
    "guests_lag364": "G(t−364)",
    "guests_lag364_7d": "mean_{d=t−370…t−364} G(d)",
    "dow": "weekday(t) ∈ {0 = Mon … 6 = Sun}",
    "is_weekend": "1 if weekday(t) ∈ {Sat, Sun} else 0",
    "month": "month(t) ∈ {1 … 12}",
    "quarter": "quarter(t) ∈ {1 … 4}",
    "year": "year(t)",
    "doy_sin": "sin(2π · day_of_year(t) ÷ 365.25)",
    "doy_cos": "cos(2π · day_of_year(t) ÷ 365.25)",
    "market": "Nationality = Departure Country Name (exact name match)",
    "new_arrivals": "NA(t) as recorded in the hotel file",
    "new_arrivals_7d": "Σ_{d=t−6…t} NA(d)",
    "same_day_guests": "SDG(t), empty cells read as 0",
}

_UNIT = {
    "load_factor": "ratio (passengers per seat)", "load_factor_7d": "ratio (passengers per seat)",
    "p2p_share_7d": "share 0–1", "transfer_share_7d": "share 0–1", "business_seat_share_7d": "share 0–1", "etihad_seat_share_7d": "share 0–1",
    "seats_per_flight_7d": "seats per flight", "n_routes": "routes", "n_cities": "cities", "n_airlines": "airlines",
    "seats_growth_28d": "relative change (0.10 = +10 %)", "p2p_growth_28d": "relative change (0.10 = +10 %)",
    "guests_lag364": "guests", "guests_lag364_7d": "guests", "dow": "category (7 levels)", "is_weekend": "0 / 1",
    "month": "1–12", "quarter": "1–4", "year": "calendar year", "doy_sin": "−1 … 1", "doy_cos": "−1 … 1",
    "market": "category (source market)", "new_arrivals": "guests", "new_arrivals_7d": "guests", "same_day_guests": "guests",
}

_WINDOW = {
    "7d": "Trailing 7 days (t−6 … t), on a complete calendar; the first 6 days of daily flight data have no value.",
    "28d": "Trailing 28 days (t−27 … t), on a complete calendar.",
    "growth": "28-day window compared with the 28 days before it, so 56 days of history are needed (from 26 Feb 2023).",
    "prev7d": "The week before the trailing week (t−13 … t−7).",
    "day": "Same day t only.",
}


def _base(name: str) -> str:
    for q in _QTY:
        if name == q or name.startswith(q + "_"):
            return q
    return ""


def _window(name: str) -> str:
    if name.endswith("_growth_28d"):
        return _WINDOW["growth"]
    if name.endswith("_prev7d"):
        return _WINDOW["prev7d"]
    if name.endswith("_28d"):
        return _WINDOW["28d"]
    if name.endswith("_7d"):
        return _WINDOW["7d"]
    return _WINDOW["day"]


def leakage(name: str, mode: str) -> tuple[str, str]:
    """(level, justification). Levels: None, Low, High, Not allowed."""
    fs = CATALOG_BY_NAME[name]
    if fs.group == "competition":
        if mode == "whatif":
            return "Not allowed", ("A same-day hotel observation. It is unknown on any future or scenario date, so a What-if "
                                   "model cannot use it.")
        return "High", ("A same-day hotel observation and an accounting component of Guests. It is present in the official "
                        "test files, so a Competition model may use it, but the result is a nowcast — it cannot forecast or simulate.")
    if fs.group == "time":
        return "None", "Calendar arithmetic on the date; known for any date in advance."
    if fs.group == "market":
        return "None", "The market identity of the row; it carries no outcome information."
    if name.startswith("guests_lag364"):
        return "Low", ("Uses Guests at least 364 days before the prediction date. The latest test date (28 Feb 2026) needs "
                       "Guests from 1 Mar 2025 or earlier, which are observed, so no future value is ever used.")
    q = _base(name)
    kind = _QTY[q][2] if q else "schedule"
    if name in ("load_factor", "load_factor_7d", "p2p_share_7d", "transfer_share_7d"):
        kind = "realised"
    if kind == "schedule" or name in ("n_routes", "n_cities", "n_airlines", "business_seat_share_7d", "etihad_seat_share_7d"):
        return "None", "Aviation capacity from airline schedules, published before the flights operate; no hotel data is involved."
    return "Low", ("Realised passenger counts are only known once the flights have operated. They are aviation data, not hotel "
                   "data, so they do not leak the target; in the What-if Simulator they are set by the scenario (seats × load "
                   "factor × P2P share). A live forecast would need them forecast first.")


def missing_treatment(name: str) -> str:
    fs = CATALOG_BY_NAME[name]
    if fs.kind == "categorical":
        return "Never missing. One-hot encoded for Ridge, SVR, Random Forest, Extra Trees and XGBoost; native categorical handling in CatBoost."
    if name == "same_day_guests":
        return ("Empty cells (the dictionary's “*”) are read as 0, so the value is a lower bound. Remaining gaps: median imputation "
                "fitted on training rows only, plus a missing-indicator column.")
    return ("Inside the model pipeline: median imputation fitted on the training rows of each fold only, plus a binary "
            "missing-indicator column (Ridge, SVR, Random Forest, Extra Trees, XGBoost). CatBoost handles NaN natively. "
            "Nothing is filled with zero.")


def missing_reason(name: str, scope: str, mode: str) -> str:
    if CATALOG_BY_NAME[name].group == "time" or name == "market":
        return "Derived from the date or market name, so it always has a value."
    if name.startswith("guests_lag364"):
        return ("Missing when the same nationality had no row 364 days earlier (a missing nationality-day combination)."
                if scope == "international_market" else "The hotel totals exist for every day, so the lag is always defined.")
    if CATALOG_BY_NAME[name].group == "competition":
        return "Empty New Arrivals cells, or a nationality-day with no row in the hotel file."
    parts = []
    if scope == "international_market" and mode == "competition":
        parts.append("nationalities without a matching departure country have no aviation features, so these are left empty "
                     "(NaN), never set to zero")
    if name in ("load_factor", "load_factor_7d", "p2p_share_7d", "transfer_share_7d", "seats_per_flight_7d",
                "business_seat_share_7d", "etihad_seat_share_7d", "seats_growth_28d", "p2p_growth_28d"):
        parts.append("a ratio is undefined (NaN, not zero) when its denominator is zero — e.g. a market-day without any operated flight")
    if not parts:
        return "Sums over route-days; a day without a flight row is a genuine zero (no operated flight), not a missing value."
    text = "; ".join(parts)
    return text[:1].upper() + text[1:] + "."


SCENARIO_INPUTS = {
    "seats": "Seat capacity",
    "load_factor": "Load factor",
    "p2p_share": "P2P share",
    "frequency": "Frequency (set independently)",
    "onset": "Scenario start (when not sustained)",
    "mix": "Single-market change (mix of the total)",
}


def scenario_text(name: str, moved: dict[str, list[str]]) -> tuple[str, str]:
    """(label, explanation) from the measured sensitivity `moved` = {feature: [input keys that change it]}."""
    fs = CATALOG_BY_NAME[name]
    if fs.group == "competition":
        return "No", "Not a scenario input; What-if models never use it."
    if fs.group in ("time", "market") or name.startswith("guests_lag364"):
        return "No", "Fixed by the date (and market); unchanged by any aviation scenario."
    keys = moved.get(name, [])
    direct = [k for k in keys if k in ("seats", "load_factor", "p2p_share", "frequency")]
    if direct:
        inputs = ", ".join(SCENARIO_INPUTS[k].lower() for k in direct)
        return "Yes", (f"Recomputed by the scenario engine from the modified route-day rows and aggregated with the same feature code used "
                       f"in training. Verified: changing the {inputs} moves this feature.")
    if keys:
        why = []
        if "mix" in keys:
            why.append("when only one market changes, because the airline / cabin mix of the total shifts")
        if "onset" in keys:
            why.append("around the start date of a change that is not sustained, because the window mixes changed and unchanged days")
        return "Indirect", ("No scenario input sets it directly; it moves only " + " and ".join(why) + ". Verified with the scenario engine.")
    return "No", "Counts routes, cities or airlines; capacity or passenger changes do not add or remove routes. Verified with the scenario engine."


def assumptions(name: str, scope: str) -> list[str]:
    fs = CATALOG_BY_NAME[name]
    out = []
    q = _base(name)
    if q or fs.group == "derived" or name.endswith("_prev7d"):
        out.append("Daily flight rows exist only from 1 Jan 2023; the 2022 file is monthly and is never spread over days.")
        if scope == "international_market":
            out.append("Market features use the departure country as a proxy for guest nationality; travellers who connect through "
                       "other hubs or fly from a third country are not attributed to their nationality.")
        elif scope == "domestic":
            out.append("The flight file covers international arrivals into AUH; for domestic guests it is an indirect signal of "
                       "overall activity, not their own travel.")
        if q in ("p2p", "pax", "transfer") or name in ("load_factor", "load_factor_7d", "p2p_share_7d"):
            out.append("Not every P2P passenger becomes a hotel guest (residents, visiting friends and relatives, day trips); "
                       "the model learns an association, not a conversion rate.")
        if name in ("load_factor", "load_factor_7d"):
            out.append("8.5 % of daily route rows have a load factor above 100 % (up to 107.6 %); they are kept as recorded.")
        if name in ("flights", "flights_7d", "seats_per_flight_7d"):
            out.append("Flights are decoded from Average Weekly Frequency, which equals flights × 7 ÷ days in the month on every daily row.")
        if q == "transfer" or name == "transfer_share_7d":
            out.append("Transfer passengers usually connect onward without staying; Total Transit is non-zero on only 3 daily rows.")
        if name in ("seats", "p2p", "pax", "transfer", "flights", "load_factor", "n_routes", "n_cities", "n_airlines"):
            out.append("The last day of the flight file (28 Feb 2026) has 9 airlines instead of about 20 and roughly half the usual seats — "
                       "probably an incomplete final day; it is kept as recorded and affects only the last test dates.")
    if name.startswith("guests_lag364"):
        out.append("Assumes last year's level on the same weekday is informative; holidays that move between years (Ramadan, Eid) "
                   "are not aligned because no holiday calendar is supplied.")
    if fs.group == "time":
        out.append("Seasonality only; no public-holiday calendar is supplied and external data is not used.")
        if name == "year":
            out.append("Off by default: a trend term extrapolates linearly into the forecast period.")
    if fs.group == "competition":
        out.append("Harmonised with the official test extract, which omits nationality-days with New Arrivals below 10.")
        out.append("Valid only for scoring the official test dates; cannot be used for What-if scenarios or genuine forecasts.")
    if fs.group == "market":
        out.append("33 of the 45 nationalities have a departure country with the same name; the other 12 are excluded from What-if market models.")
    return out


# ------------------------------------------------------------------ measured facts

@dataclass
class FeatureRow:
    name: str
    label: str
    group: str
    available: bool
    reason: str
    missing_n: int
    missing_pct: float
    rows: int
    leakage: str
    leakage_why: str
    scenario: str
    scenario_why: str
    monotone: bool
    default: bool


def training_matrix(a: Analytic, scope: str, mode: str) -> pd.DataFrame:
    """The training rows the ML Training Studio fits on (all available feature columns)."""
    return training_rows(model_frame(a, scope, mode))


def matrix_columns(scope: str, features: list[str]) -> list[str]:
    return ["Date"] + (["market_key"] if scope == "international_market" else []) + ["y"] + list(features)


def matrix_fingerprint(df: pd.DataFrame) -> str:
    """Order-sensitive hash of a matrix (values, column names and row order)."""
    h = hashlib.sha256()
    h.update("|".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()[:12]


def scenario_sensitivity(a: Analytic, market: str = "INDIA") -> dict[str, list[str]]:
    """Which aviation features each scenario input actually moves, measured with the scenario engine
    on the daily-total features (Jan–Mar 2025). Returns {feature: [input keys]}."""
    from .scenario import ScenarioConfig, apply_scenario
    rows = a.flight_daily_rows
    start, end = "2025-01-01", "2025-03-31"
    base = aviation_features(aggregate_aviation(rows, by_market=False), by_market=False).set_index("Date")
    tests = {
        "seats": dict(seats_change_pct=20.0),
        "load_factor": dict(load_factor_mode="delta", load_factor_value=5.0),
        "p2p_share": dict(p2p_share_mode="delta", p2p_share_value=5.0),
        "frequency": dict(frequency_mode="independent", frequency_change_pct=10.0),
        "onset": dict(seats_change_pct=20.0, sustained=False),
        "mix": dict(seats_change_pct=20.0, market=market),
    }
    win = slice(pd.Timestamp(start), pd.Timestamp(end))
    out: dict[str, list[str]] = {}
    for tag, kw in tests.items():
        cfg = ScenarioConfig(**{"model_id": "", "market": "ALL", "start": start, "end": end, **kw})
        mod, _ = apply_scenario(rows, cfg)
        new = aviation_features(aggregate_aviation(mod, by_market=False), by_market=False).set_index("Date")
        for c in new.columns:
            if c in CATALOG_BY_NAME and np.nanmax(np.abs(new.loc[win, c].to_numpy(float) - base.loc[win, c].to_numpy(float))) > 1e-9:
                out.setdefault(c, []).append(tag)
    return out


def feature_rows(a: Analytic, scope: str, mode: str, moved: dict[str, set]) -> list[FeatureRow]:
    tr = training_matrix(a, scope, mode)
    avail = set(available_features(scope, mode))
    defaults = set(default_features(scope, mode))
    out = []
    for fs in CATALOG:
        if scope not in fs.scopes:
            continue
        ok = fs.name in avail
        reason = "" if ok else "Competition mode only — a same-day hotel observation that does not exist on scenario dates."
        if fs.name in tr:
            n = int(tr[fs.name].isna().sum())
            pct = 100.0 * n / len(tr) if len(tr) else float("nan")
        else:
            n, pct = None, float("nan")
        lv, why = leakage(fs.name, mode)
        sc, scw = scenario_text(fs.name, moved)
        out.append(FeatureRow(fs.name, fs.label, fs.group, ok, reason, n if n is not None else -1, pct, len(tr), lv, why, sc, scw,
                              fs.name in MONOTONE_INCREASING, fs.name in defaults))
    return out


def empty_same_day_cells(a: Analytic, scope: str, mode: str) -> int | None:
    """Training rows whose Same-Day Guests value includes an empty cell read as 0."""
    if mode != "competition":
        return None
    hm = a.hotel_market
    tr = hm[(hm["split"] == "train") & hm["Date"].between(config.MODEL_START, config.MODEL_END)]
    if scope == "international_market":
        tr = tr[tr["New Arrivals"] >= 10]
        return int(tr["Same-Day Guests"].isna().sum())
    if scope == "domestic":
        return 0
    return int(tr.groupby("Date")["Same-Day Guests"].apply(lambda s: s.isna().any()).sum())


def zero_days(tr: pd.DataFrame, name: str) -> int:
    return int((tr[name] == 0).sum()) if name in tr and CATALOG_BY_NAME[name].kind == "numeric" else 0
