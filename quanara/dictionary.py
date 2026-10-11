"""Field-definition table built from Data_Dictionary.pdf plus audit findings.

`definition` paraphrases/quotes the dictionary text for that exact column; `audit_finding`
records what the delivered files actually contain where it differs. Nothing here is inferred
from a similarly named column.

known_whatif      — can the value be known when forecasting a scenario (no hotel observations)?
known_competition — is the value present in the official test files / flight file at test dates?
scenario_editable — can a What-if user change it (directly or as a derived consequence)?
"""
from __future__ import annotations

import pandas as pd

_HOTEL_FILES = "intl train/test, dom train/test"

FIELDS: list[dict] = [
    # ------------------------------------------------------------------ hotel files
    dict(column="Date", source=f"{_HOTEL_FILES}",
         definition="Calendar date to which the guest counts apply (documented as string YYYY/M/D). International: one row per nationality per date. Domestic: exactly one row per date.",
         dtype_units="date (stored as Excel datetime)", missing_meaning="never missing",
         role="key", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="Stored as real Excel dates, not strings. Daily with no gaps in every hotel file; the computed date ranges are listed in the profile."),
    dict(column="Guests", source="intl train, dom train",
         definition="TARGET VARIABLE. Total number of guests recorded for the nationality (international) or in total (domestic) on the given date.",
         dtype_units="integer, persons recorded on the date", missing_meaning="never missing in train; column ABSENT from both test files",
         role="target", known_whatif=False, known_competition=False, scenario_editable=False,
         audit_finding="Absent from both official test files (no observed target to score against). Some nationality-day combinations have no row; suppression of small counts is a possible but unconfirmed explanation. Guests counts everyone recorded on the date, continuing stays included, so summed over days it gives guest-days — not guest nights, because length of stay is unknown and same-day visitors are included."),
    dict(column="New Arrivals", source=_HOTEL_FILES,
         definition="Guests newly arrived (check-in) on the date, as opposed to continuing stays. Always a subset of Guests.",
         dtype_units="integer, persons", missing_meaning="empty cell = suppressed or unavailable (a few rows; see the profile)",
         role="predictor (competition mode only)", known_whatif=False, known_competition=True, scenario_editable=False,
         audit_finding="Never exceeds Guests on any training row. Its accounting link to Guests means it is allowed only in Competition mode, where the test files supply it. Test rows never have New Arrivals below 10 while some training rows do: the two extracts follow different inclusion rules."),
    dict(column="Same-Day Guests", source=_HOTEL_FILES,
         definition="Guests who visited and departed on the same day without an overnight stay. International: '*' indicates zero, suppressed, or not applicable. Domestic: documented as always numeric.",
         dtype_units="integer, persons", missing_meaning="empty cell = '*' = zero OR suppressed OR n/a (cannot be distinguished)",
         role="predictor (competition mode only)", known_whatif=False, known_competition=True, scenario_editable=False,
         audit_finding="International: blank in a large share of rows, and a blank's original meaning cannot be uniquely recovered from the delivered files. Domestic: never blank, as documented. Never exceeds Guests or New Arrivals."),
    dict(column="Nationality", source="intl train/test",
         definition="Passport nationality / country of origin of the guests, upper-case names.",
         dtype_units="categorical", missing_meaning="never missing",
         role="key / market identifier", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="The same nationalities appear in train and test. Most share a name with a flight 'Departure Country Name', but nationality is not the same concept as departure country."),
    dict(column="Residence (groups)", source=_HOTEL_FILES,
         definition="High-level residence group: 'International' (non-resident visitor) in the international files, 'Domestic' (resident / in-country) in the domestic files.",
         dtype_units="categorical (constant per file)", missing_meaning="never missing",
         role="constant (excluded)", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="Constant within each file → carries no predictive information; used only to label scope."),
    # ------------------------------------------------------------------ flight file
    dict(column="Date", source="flight",
         definition="First day of the reporting month (e.g. '1/1/2022 12:00:00 AM' = January 2022) — documented as MONTHLY.",
         dtype_units="date", missing_meaning="never missing",
         role="key", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="Granularity differs from the dictionary: 2022 rows are monthly (dated the 1st); from January 2023 the file is daily."),
    dict(column="Departure Country Name", source="flight",
         definition="Full name of the country from which the flight departs.",
         dtype_units="categorical", missing_meaning="never missing",
         role="market identifier / aggregation key", known_whatif=True, known_competition=True, scenario_editable=True,
         audit_finding="Every departure country matches a guest nationality by name. A departure country is where the flight originates, not the passengers' nationality (for example, UK departures carry many non-UK nationals and mostly transfer traffic)."),
    dict(column="Departure City", source="flight",
         definition="City of origin; some countries have several departure cities.",
         dtype_units="categorical (many levels)", missing_meaning="never missing",
         role="derived predictor (count of active cities)", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="Too granular to join to hotel data (no city in hotel files). Used via a derived 'active departure cities' count."),
    dict(column="Arrival City", source="flight",
         definition="Destination city; 'Abu Dhabi' in all observed records.",
         dtype_units="categorical", missing_meaning="never missing",
         role="constant (excluded)", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="Constant: 'Abu Dhabi' on every row."),
    dict(column="Airline Name", source="flight",
         definition="Operating airline; 'Unknown' for a few unidentified records.",
         dtype_units="categorical", missing_meaning="'Unknown' label on a few rows",
         role="derived predictor (airline count, Etihad seat share)", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="Cannot be a direct feature at daily-aggregate grain (many airlines per day). Represented by active-airline count and Etihad Airways seat share."),
    dict(column="Average Weekly Frequency", source="flight",
         definition="Average number of flights per week on the route during the reporting month. Dictionary: 'Populated with * in most observed rows … may not be available'.",
         dtype_units="float, flights/week", missing_meaning="empty on every 2022 monthly row",
         role="predictor (decoded to daily flights)", known_whatif=True, known_competition=True, scenario_editable=True,
         audit_finding="Contradicts the dictionary: populated on every daily row. Value = flights that day × 7 ÷ days in the month exactly, giving an integer daily flight count, so frequency is a validated feature for daily data."),
    dict(column="Business Class P2P Count", source="flight",
         definition="Point-to-point Business Class passengers (Abu Dhabi is their final destination). * = not available.",
         dtype_units="integer, passengers", missing_meaning="empty on all 2022 monthly rows",
         role="excluded (approximate)", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="Populated on daily rows. Cabin P2P counts add up to Total P2P on only a minority of rows; adding infants explains most of the gap. Totals are used instead."),
    dict(column="Business Class Seat Capacity", source="flight",
         definition="Business Class seats available on the route for the month. * = n/a (e.g. no Business cabin).",
         dtype_units="integer, seats", missing_meaning="empty on all 2022 monthly rows",
         role="derived predictor (business seat share)", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="Cabin capacities add up exactly to Total Seats on every daily row. Many route-days have no Business cabin."),
    dict(column="Economy Class P2P Count", source="flight",
         definition="Point-to-point Economy Class passengers for the route.",
         dtype_units="integer, passengers", missing_meaning="empty on all 2022 monthly rows",
         role="excluded (approximate)", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="See Business Class P2P Count: cabin P2P rarely equals Total P2P."),
    dict(column="Economy Class Seat Capacity", source="flight",
         definition="Economy Class seats available on the route.",
         dtype_units="integer, seats", missing_meaning="empty on all 2022 monthly rows",
         role="component of Total Seats", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="Exact component of Total Seats on daily rows."),
    dict(column="First Class P2P Count", source="flight",
         definition="Point-to-point First Class passengers; predominantly * (few routes offer First).",
         dtype_units="integer, passengers", missing_meaning="empty on all 2022 monthly rows; zero where no First cabin",
         role="excluded (approximate)", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="Daily rows are populated (zeros, not '*', where there is no First cabin)."),
    dict(column="First Class Seat Capacity", source="flight",
         definition="First Class seats available; predominantly * as most airlines do not operate First.",
         dtype_units="integer, seats", missing_meaning="empty on all 2022 monthly rows",
         role="component of Total Seats", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="First Class capacity is present on only a small share of daily rows."),
    dict(column="Load Factor", source="flight",
         definition="Share of seat capacity occupied: Total PAX / Total Seats. Documented as a percentage string; values above 100% possible (overbooking / infant counting).",
         dtype_units="ratio (stored as a fraction, e.g. 0.4702 = 47.02%)", missing_meaning="never missing",
         role="predictor (recomputed after aggregation)", known_whatif=True, known_competition=True, scenario_editable=True,
         audit_finding="Stored as a numeric fraction, not a percentage string. Equals Total PAX ÷ Total Seats on every row and exceeds 100% on some rows (see the Data quality findings). Aggregates are recomputed as ΣPAX ÷ ΣSeats — never averaged."),
    dict(column="Total P2P", source="flight",
         definition="Point-to-point passengers across all cabins: journey originates at the departure city and terminates at Abu Dhabi.",
         dtype_units="integer, passengers", missing_meaning="never missing",
         role="key predictor", known_whatif=True, known_competition=True, scenario_editable=True,
         audit_finding="The aviation quantity closest to potential hotel demand — but P2P includes residents returning home, people visiting friends and relatives, and day visitors, so P2P ≠ hotel guests. About half of daily PAX are P2P."),
    dict(column="Total PAX", source="flight",
         definition="All passengers on board across classes, including P2P and transfer/transit.",
         dtype_units="integer, passengers", missing_meaning="never missing",
         role="predictor", known_whatif=True, known_competition=True, scenario_editable=True,
         audit_finding="Identity verified on every row: Total PAX = Total P2P + Total Transfer + Total Transit."),
    dict(column="Total PAX Excluding Infant", source="flight",
         definition="Total passengers minus lap infants; typically 1–5 below Total PAX.",
         dtype_units="integer, passengers", missing_meaning="rarely blank",
         role="excluded (near-duplicate of Total PAX)", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="Usually a few passengers below Total PAX and never above it. Redundant with Total PAX."),
    dict(column="Total Seats", source="flight",
         definition="Total seat capacity offered across all cabins; the Load Factor denominator.",
         dtype_units="integer, seats", missing_meaning="never missing (never zero)",
         role="key predictor (scenario lever)", known_whatif=True, known_competition=True, scenario_editable=True,
         audit_finding="Equals the sum of cabin capacities on every daily row."),
    dict(column="Total Transfer", source="flight",
         definition="Passengers who arrived on this route and transferred to another flight at Abu Dhabi (change of aircraft / flight number).",
         dtype_units="integer, passengers", missing_meaning="never missing",
         role="predictor (distinguished from P2P)", known_whatif=True, known_competition=True, scenario_editable=True,
         audit_finding="Close to half of daily PAX. Transfer passengers connect onward and are not expected to stay in Abu Dhabi hotels, unless on a stopover, which the data cannot identify."),
    dict(column="Total Transit", source="flight",
         definition="Passengers in transit through Abu Dhabi (same aircraft / simpler connection). Mostly 0.",
         dtype_units="integer, passengers", missing_meaning="never missing",
         role="excluded (near-zero variance)", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="Zero on all but a handful of rows, so it carries no usable information: it is not a model feature and is held at its observed value in scenarios."),
    dict(column="Transfer Business Class Count", source="flight",
         definition="Business Class passengers who transferred at Abu Dhabi. * = not available.",
         dtype_units="integer, passengers", missing_meaning="empty on 2022 monthly rows (rarely on daily rows)",
         role="excluded (approximate)", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="Cabin transfer counts add up to Total Transfer on only part of the rows."),
    dict(column="Transfer Economy Class Count", source="flight",
         definition="Economy Class passengers who transferred at Abu Dhabi. * = not available.",
         dtype_units="integer, passengers", missing_meaning="empty on 2022 monthly rows (rarely on daily rows)",
         role="excluded (approximate)", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="See Transfer Business Class Count."),
    dict(column="Transfer First Class Count", source="flight",
         definition="First Class passengers who transferred at Abu Dhabi. * = not available.",
         dtype_units="integer, passengers", missing_meaning="empty on 2022 monthly rows (rarely on daily rows)",
         role="excluded (approximate)", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="See Transfer Business Class Count."),
    dict(column="Transit Business Class Count", source="flight",
         definition="Business Class passengers in transit at Abu Dhabi. * = not available.",
         dtype_units="integer, passengers", missing_meaning="empty on 2022 rows",
         role="excluded (inconsistent)", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="Cabin transit counts are non-zero on many daily rows although Total Transit is almost always zero; they appear to be counted within Total Transfer. Excluded."),
    dict(column="Transit Economy Class Count", source="flight",
         definition="Economy Class passengers in transit at Abu Dhabi. * = not available.",
         dtype_units="integer, passengers", missing_meaning="empty on 2022 rows",
         role="excluded (inconsistent)", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="See Transit Business Class Count."),
    dict(column="Transit First Count", source="flight",
         definition="First Class passengers in transit (note the abbreviated name vs. the 'Class Count' pattern). * = not available.",
         dtype_units="integer, passengers", missing_meaning="empty on 2022 rows",
         role="excluded (inconsistent)", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="Column name confirmed exactly as the dictionary warns ('Transit First Count')."),
    dict(column="Destination", source="flight",
         definition="IATA code of the arrival airport; 'AUH' in all observed records.",
         dtype_units="categorical", missing_meaning="never missing",
         role="constant (excluded)", known_whatif=True, known_competition=True, scenario_editable=False,
         audit_finding="Constant: 'AUH' on every row."),
]


def field_table() -> pd.DataFrame:
    df = pd.DataFrame(FIELDS)
    order = ["column", "source", "definition", "dtype_units", "missing_meaning", "role",
             "known_whatif", "known_competition", "scenario_editable", "audit_finding"]
    return df[order]


DICTIONARY_DISCREPANCIES = [
    "Flight data is documented as monthly; it is monthly only for 2022 and DAILY from 2023-01-01.",
    "Average Weekly Frequency is documented as mostly '*'; it is populated on 100% of daily rows and decodes exactly to daily flight counts.",
    "Cabin-class P2P/seat/transfer columns are documented as predominantly '*'; they are empty only on 2022 monthly rows.",
    "Suppression is documented as '*'; the .xlsx files contain empty cells instead (zero literal '*' cells).",
    "Load Factor is documented as a percentage string; it is stored as a numeric fraction.",
    "Dates are documented as strings; they are stored as Excel datetimes.",
    "International test coverage is documented as 'Aug 2025 – Aug 2025'; the file covers 2025-08-01 → 2026-02-28 (212 days).",
    "Guests is documented as the target in the test files; both test files do not contain a Guests column.",
    "The dictionary states 32 unique columns; the five files contain 31 distinct column names (Date is shared).",
]


def field_profiles(d: dict[str, pd.DataFrame]) -> dict[str, str]:
    """A short profile of every field computed from the delivered files at run time (key: 'column|source')."""
    def fmt(v: float) -> str:
        return f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:,.3f}"

    def desc(s: pd.Series) -> str:
        if pd.api.types.is_datetime64_any_dtype(s):
            return f"{s.min():%Y-%m-%d} → {s.max():%Y-%m-%d}, {s.nunique():,} dates"
        filled = s.notna().mean()
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            v = s.dropna()
            return f"{filled:.1%} filled · min {fmt(v.min())} · median {fmt(v.median())} · max {fmt(v.max())}"
        return f"{filled:.1%} filled · {s.nunique():,} distinct value{'s' if s.nunique() != 1 else ''}"

    hotel = {"International train": d["intl_train"], "International test": d["intl_test"], "Domestic train": d["dom_train"], "Domestic test": d["dom_test"]}
    fl = d["flight"]
    daily, monthly = fl[fl["grain"] == "daily"], fl[fl["grain"] == "monthly"]
    out = {}
    for f in FIELDS:
        col, src = f["column"], f["source"]
        if src == "flight":
            parts = [f"Daily rows ({len(daily):,}): {desc(daily[col])}"]
            parts.append(f"2022 monthly rows ({len(monthly):,}): {desc(monthly[col])}" if monthly[col].notna().any() else "2022 monthly rows: empty")
        else:
            parts = [f"{name} ({len(df):,} rows): {desc(df[col])}" for name, df in hotel.items() if col in df.columns]
            absent = [name.lower() for name, df in hotel.items() if col not in df.columns]
            if absent:
                parts.append("absent from " + " and ".join(absent))
        out[f"{col}|{src}"] = "; ".join(parts) + "."
    return out
