"""Expected schemas for the five workbooks and a validator.

Column names are the exact headers found in the files (verified in the audit).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

HOTEL_NUMERIC = ["Guests", "New Arrivals", "Same-Day Guests"]

FLIGHT_CATEGORICAL = [
    "Departure Country Name", "Departure City", "Arrival City", "Airline Name", "Destination",
]
FLIGHT_NUMERIC = [
    "Average Weekly Frequency",
    "Business Class P2P Count", "Business Class Seat Capacity",
    "Economy Class P2P Count", "Economy Class Seat Capacity",
    "First Class P2P Count", "First Class Seat Capacity",
    "Load Factor", "Total P2P", "Total PAX", "Total PAX Excluding Infant", "Total Seats",
    "Total Transfer", "Total Transit",
    "Transfer Business Class Count", "Transfer Economy Class Count", "Transfer First Class Count",
    "Transit Business Class Count", "Transit Economy Class Count", "Transit First Count",
]

EXPECTED_COLUMNS = {
    "intl_train": ["Date", "Guests", "New Arrivals", "Same-Day Guests", "Nationality", "Residence (groups)"],
    "intl_test": ["Date", "New Arrivals", "Same-Day Guests", "Nationality", "Residence (groups)"],
    "dom_train": ["Date", "Guests", "New Arrivals", "Same-Day Guests", "Residence (groups)"],
    "dom_test": ["Date", "New Arrivals", "Same-Day Guests", "Residence (groups)"],
    "flight": ["Date", "Departure Country Name", "Departure City", "Arrival City", "Airline Name",
               *FLIGHT_NUMERIC[:1], *FLIGHT_NUMERIC[1:], "Destination"],
}

PRIMARY_KEYS = {
    "intl_train": ["Date", "Nationality"],
    "intl_test": ["Date", "Nationality"],
    "dom_train": ["Date"],
    "dom_test": ["Date"],
    "flight": ["Date", "Departure Country Name", "Departure City", "Airline Name"],
}


@dataclass
class ValidationResult:
    dataset: str
    ok: bool
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    duplicate_keys: int = 0
    messages: list[str] = field(default_factory=list)


def validate(dataset: str, df: pd.DataFrame) -> ValidationResult:
    expected = EXPECTED_COLUMNS[dataset]
    missing = [c for c in expected if c not in df.columns]
    unexpected = [c for c in df.columns if c not in expected]
    res = ValidationResult(dataset, ok=not missing, missing_columns=missing, unexpected_columns=unexpected)
    if missing:
        res.messages.append(f"Missing required columns: {missing}")
    if unexpected:
        res.messages.append(f"Unexpected columns (ignored): {unexpected}")
    key = [k for k in PRIMARY_KEYS[dataset] if k in df.columns]
    if key:
        res.duplicate_keys = int(df.duplicated(key).sum())
        if res.duplicate_keys:
            res.ok = False
            res.messages.append(f"{res.duplicate_keys} duplicate rows on key {key}")
    if "Date" in df.columns and df["Date"].isna().any():
        res.ok = False
        res.messages.append(f"{int(df['Date'].isna().sum())} unparseable dates")
    return res
