"""Cleaning rules. Every rule here is justified in reports/audit_report.md.

* The dictionary documents "*" as a suppression marker. In the delivered .xlsx files the
  suppressed cells are EMPTY (no literal "*" exists), so both empty cells and "*" become NaN.
  Suppressed values are never converted to zero here; any later zero-fill is explicit and logged.
* flight_data.xlsx mixes grains: 2022 rows are monthly (dated the 1st), 2023+ rows are daily.
  The grain is detected from the data, not assumed, and stored in a `grain` column.
* On daily rows "Average Weekly Frequency" equals flights_that_day * 7 / days_in_month
  (verified to 1e-15 on all 116,395 daily rows), so the daily flight count is recovered exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import FLIGHT_NUMERIC, HOTEL_NUMERIC

SUPPRESSION_MARKERS = {"*", "* ", " *"}


def _to_numeric(s: pd.Series) -> pd.Series:
    s = s.map(lambda v: np.nan if isinstance(v, str) and v.strip() in SUPPRESSION_MARKERS else v)
    return pd.to_numeric(s, errors="coerce")


def _clean_dates(s: pd.Series) -> pd.Series:
    # Dates arrive as Excel datetimes; tolerate the documented string formats too.
    return pd.to_datetime(s, errors="coerce", format="mixed").dt.normalize()


def clean_hotel(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["Date"] = _clean_dates(df["Date"])
    for c in HOTEL_NUMERIC:
        if c in df.columns:
            df[c] = _to_numeric(df[c])
    if "Nationality" in df.columns:
        df["Nationality"] = df["Nationality"].astype(str).str.strip().str.upper()
    df["Residence (groups)"] = df["Residence (groups)"].astype(str).str.strip()
    if "Same-Day Guests" in df.columns:
        df["same_day_suppressed"] = df["Same-Day Guests"].isna()
    return df.sort_values([c for c in ("Date", "Nationality") if c in df.columns]).reset_index(drop=True)


def detect_flight_grain(dates: pd.Series) -> pd.Series:
    """Label each row 'monthly' or 'daily' from the data itself.

    A calendar month is monthly-grain when all of its rows carry a single date that is the
    1st of the month; otherwise it is daily-grain.
    """
    ym = dates.dt.to_period("M")
    stats = pd.DataFrame({"ym": ym, "d": dates}).groupby("ym")["d"].agg(["nunique", "min"])
    monthly = (stats["nunique"] == 1) & (stats["min"].dt.day == 1)
    return ym.map(lambda p: "monthly" if monthly.get(p, False) else "daily")


def clean_flight(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["Date"] = _clean_dates(df["Date"])
    for c in ["Departure Country Name", "Departure City", "Arrival City", "Airline Name", "Destination"]:
        df[c] = df[c].astype(str).str.strip()
    for c in FLIGHT_NUMERIC:
        df[c] = _to_numeric(df[c])
    df["grain"] = detect_flight_grain(df["Date"])
    daily = df["grain"].eq("daily")
    flights = df["Average Weekly Frequency"] * df["Date"].dt.days_in_month / 7.0
    df["flights"] = np.where(daily, flights.round(), np.nan)
    df["flights_decode_error"] = np.where(daily, (flights - flights.round()).abs(), np.nan)
    # Upper-case join key; departure country is NOT assumed to equal guest nationality —
    # this key only enables an explicit, reported name match.
    df["market_key"] = df["Departure Country Name"].str.upper()
    return df.sort_values(["Date", "Departure Country Name", "Departure City", "Airline Name"]).reset_index(drop=True)


def clean_dataset(dataset: str, raw: pd.DataFrame) -> pd.DataFrame:
    return clean_flight(raw) if dataset == "flight" else clean_hotel(raw)
