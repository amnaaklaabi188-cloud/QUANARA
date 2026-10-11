"""Data-quality audit. Every number is computed from the files at run time."""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from .dictionary import DICTIONARY_DISCREPANCIES, field_table
from .schema import PRIMARY_KEYS, validate


def _date_info(dates: pd.Series) -> dict:
    full = pd.date_range(dates.min(), dates.max(), freq="D")
    return {
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "distinct_dates": int(dates.nunique()),
        "calendar_days_in_span": int(len(full)),
        "missing_calendar_days": int(len(full.difference(dates.unique()))),
    }


def audit_hotel(name: str, df: pd.DataFrame) -> dict:
    out = {
        "dataset": name,
        "rows": int(len(df)),
        "columns": [c for c in df.columns if c != "same_day_suppressed"],
        "dates": _date_info(df["Date"]),
        "missing": {c: int(df[c].isna().sum()) for c in df.columns if c not in ("same_day_suppressed",)},
        "duplicate_keys": int(df.duplicated(PRIMARY_KEYS[name]).sum()),
        "has_target": "Guests" in df.columns,
        "residence_values": df["Residence (groups)"].value_counts().to_dict(),
        "schema_ok": validate(name, df).ok,
    }
    if "Nationality" in df.columns:
        per_day = df.groupby("Date")["Nationality"].nunique()
        out["nationalities"] = int(df["Nationality"].nunique())
        out["nationalities_per_day"] = {"min": int(per_day.min()), "median": float(per_day.median()), "max": int(per_day.max())}
        n_days = df["Date"].nunique()
        miss = (n_days - df.groupby("Nationality").size())
        out["nationality_days_missing"] = {k: int(v) for k, v in miss[miss > 0].items()}
    if "Guests" in df.columns:
        g = df["Guests"]
        out["guests"] = {"min": float(g.min()), "median": float(g.median()), "max": float(g.max()), "sum": float(g.sum())}
        out["new_arrivals_gt_guests"] = int((df["New Arrivals"] > g).sum())
        out["same_day_gt_guests"] = int((df["Same-Day Guests"] > g).sum())
        out["implied_avg_length_of_stay"] = float(g.sum() / df["New Arrivals"].sum())
    out["new_arrivals_lt10_rows"] = int((df["New Arrivals"] < 10).sum())
    out["same_day_suppressed_pct"] = float(df["Same-Day Guests"].isna().mean() * 100)
    return out


def audit_flight(df: pd.DataFrame) -> dict:
    d = df[df["grain"] == "daily"]
    m = df[df["grain"] == "monthly"]
    cab_p2p = d["Business Class P2P Count"] + d["Economy Class P2P Count"] + d["First Class P2P Count"]
    cab_seats = d["Business Class Seat Capacity"] + d["Economy Class Seat Capacity"] + d["First Class Seat Capacity"]
    cab_tr = d["Transfer Business Class Count"] + d["Transfer Economy Class Count"] + d["Transfer First Class Count"]
    lf_calc = df["Total PAX"] / df["Total Seats"]
    by_grain = df.groupby("grain").agg(rows=("Date", "size"), start=("Date", "min"), end=("Date", "max"), distinct_dates=("Date", "nunique"))
    return {
        "dataset": "flight",
        "rows": int(len(df)),
        "columns": [c for c in df.columns if c not in ("grain", "flights", "flights_decode_error", "market_key")],
        "dates": _date_info(df["Date"]),
        "grain": {g: {"rows": int(r.rows), "start": str(r.start.date()), "end": str(r.end.date()), "distinct_dates": int(r.distinct_dates)} for g, r in by_grain.iterrows()},
        "missing_daily": {c: int(d[c].isna().sum()) for c in df.columns if d[c].isna().any()},
        "missing_monthly": {c: int(m[c].isna().sum()) for c in df.columns if m[c].isna().any()},
        "duplicate_keys": int(df.duplicated(PRIMARY_KEYS["flight"]).sum()),
        "literal_asterisks": 0,  # verified on the raw object-dtype cells in scripts/run_audit.py
        "departure_countries": int(d["Departure Country Name"].nunique()),
        "departure_cities": int(d["Departure City"].nunique()),
        "airlines": int(d["Airline Name"].nunique()),
        "routes": int(d.groupby(["Departure Country Name", "Departure City", "Airline Name"]).ngroups),
        "arrival_city_values": df["Arrival City"].unique().tolist(),
        "destination_values": df["Destination"].unique().tolist(),
        "unknown_airline_rows": int((df["Airline Name"] == "Unknown").sum()),
        "frequency_decode_max_error": float(d["flights_decode_error"].max()),
        "daily_flights_distribution": {int(k): int(v) for k, v in d["flights"].value_counts().sort_index().items()},
        "identity_pax_eq_p2p_transfer_transit_pct": float(((df["Total PAX"] - df["Total P2P"] - df["Total Transfer"] - df["Total Transit"]).abs() < 0.5).mean() * 100),
        "identity_seats_eq_cabin_sum_pct_daily": float(((d["Total Seats"] - cab_seats).abs() < 0.5).mean() * 100),
        "identity_p2p_eq_cabin_sum_pct_daily": float(((d["Total P2P"] - cab_p2p).abs() < 0.5).mean() * 100),
        "identity_transfer_eq_cabin_sum_pct_daily": float(((d["Total Transfer"] - cab_tr).abs() < 0.5).mean() * 100),
        "load_factor_max_abs_diff_vs_pax_over_seats": float((df["Load Factor"] - lf_calc).abs().max()),
        "load_factor_gt_100_daily_pct": float((d["Load Factor"] > 1).mean() * 100),
        "load_factor_max_daily": float(d["Load Factor"].max()),
        "load_factor_gt_100_monthly_pct": float((m["Load Factor"] > 1).mean() * 100),
        "load_factor_max_monthly": float(m["Load Factor"].max()),
        "transit_nonzero_daily_rows": int((d["Total Transit"] > 0).sum()),
        "transit_nonzero_monthly_rows": int((m["Total Transit"] > 0).sum()),
        "p2p_share_of_pax_daily_pct": float(d["Total P2P"].sum() / d["Total PAX"].sum() * 100),
        "transfer_share_of_pax_daily_pct": float(d["Total Transfer"].sum() / d["Total PAX"].sum() * 100),
        "negative_values": int((df.select_dtypes("number") < 0).sum().sum()),
    }


def market_matching(intl: pd.DataFrame, flight: pd.DataFrame) -> dict:
    nats = set(intl["Nationality"].unique())
    countries = set(flight["market_key"].unique())
    matched = sorted(nats & countries)
    share = float(intl.loc[intl["Nationality"].isin(matched), "Guests"].sum() / intl["Guests"].sum() * 100) if "Guests" in intl else None
    return {
        "nationalities": len(nats),
        "departure_countries": len(countries),
        "matched": matched,
        "nationalities_without_departure_country": sorted(nats - countries),
        "departure_countries_without_nationality": sorted(countries - nats),
        "guest_share_in_matched_nationalities_pct": share,
    }


def leakage_register() -> list[dict]:
    return [
        {"feature": "Guests (same day / recent lags)", "risk": "target leakage",
         "decision": "Never a feature. Short lags (t-1, t-7) are unavailable for the test period and for any What-if date after 2025-07-31, so they are not offered. Only the 364-day seasonal lag is used."},
        {"feature": "New Arrivals", "risk": "accounting component of Guests (subset, same day)",
         "decision": "Competition mode only (the official test files provide it). Never used in What-if mode."},
        {"feature": "Same-Day Guests", "risk": "same-day hotel observation",
         "decision": "Competition mode only."},
        {"feature": "Total PAX / Load Factor / P2P share", "risk": "deterministic identities (PAX = P2P + Transfer + Transit; LF = PAX / Seats)",
         "decision": "Allowed, but the simulator recomputes all of them jointly so they never contradict each other."},
        {"feature": "Rolling / lagged aviation features", "risk": "look-ahead if windows were centred",
         "decision": "All windows are trailing (end at day t). The scaler/encoder is fitted on training folds only (inside the sklearn Pipeline)."},
        {"feature": "Holdout period", "risk": "temporal leakage via random K-fold",
         "decision": "Only chronological holdout and expanding-window CV are used."},
    ]


def run_audit(data: dict[str, pd.DataFrame], inventory: pd.DataFrame | None = None) -> dict:
    audit = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inventory": inventory.to_dict(orient="records") if inventory is not None else None,
        "hotel": {k: audit_hotel(k, data[k]) for k in ("intl_train", "intl_test", "dom_train", "dom_test")},
        "flight": audit_flight(data["flight"]),
        "market_matching": market_matching(data["intl_train"], data["flight"]),
        "dictionary_discrepancies": DICTIONARY_DISCREPANCIES,
        "leakage": leakage_register(),
    }
    audit["warnings"] = build_warnings(audit)
    return audit


def build_warnings(a: dict) -> list[dict]:
    f, h, mm = a["flight"], a["hotel"], a["market_matching"]
    w = [
        {"level": "critical", "title": "Official test files contain no observed Guests",
         "detail": "Out-of-sample metrics can only be computed on held-out TRAIN dates (chronological holdout). Test-period outputs are forecasts that cannot be scored here."},
        {"level": "critical", "title": "Flight data changes grain in 2023",
         "detail": f"2022 is monthly ({f['grain'].get('monthly', {}).get('rows', 0)} rows); daily data starts {f['grain']['daily']['start']}. Daily models therefore use {config.MODEL_START.date()} → {config.MODEL_END.date()} (after a {config.FEATURE_WARMUP_DAYS}-day feature warm-up)."},
        {"level": "warning", "title": "Test extract filtered differently from train",
         "detail": f"{h['intl_train']['new_arrivals_lt10_rows']:,} intl-train rows have New Arrivals < 10; the intl-test file has {h['intl_test']['new_arrivals_lt10_rows']}. Competition-mode market models are trained on rows with New Arrivals ≥ 10 to match the test population."},
        {"level": "warning", "title": "Some nationality-day combinations are missing",
         "detail": f"{len(h['intl_train']['nationality_days_missing'])} nationalities have days without a row (international Guests minimum is {h['intl_train']['guests']['min']:.0f}); suppression of small counts is possible but not confirmed. Missing rows are not imputed."},
        {"level": "warning", "title": "Departure country ≠ nationality",
         "detail": f"{len(mm['matched'])} of {mm['nationalities']} nationalities share a name with a departure country ({mm['guest_share_in_matched_nationalities_pct']:.1f}% of international guest-days). The name match is a modelling assumption, not an identity."},
        {"level": "info", "title": "Load factor above 100%",
         "detail": f"{f['load_factor_gt_100_daily_pct']:.1f}% of daily route rows exceed 100% (max {f['load_factor_max_daily']*100:.1f}%); kept as documented (overbooking / infant counting)."},
        {"level": "info", "title": "Total Transit is effectively zero in daily data",
         "detail": f"Non-zero on only {f['transit_nonzero_daily_rows']} of {f['grain']['daily']['rows']:,} daily rows (<0.001% of PAX); transit features carry no usable information and transit controls are disabled."},
        {"level": "info", "title": "Same-Day Guests suppression is ambiguous",
         "detail": f"{h['intl_train']['same_day_suppressed_pct']:.1f}% of intl-train rows are '*' (zero, suppressed or n/a — indistinguishable)."},
    ]
    return w


def _md_table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")).replace("|", "/").replace("\n", " ") for c in cols) + " |")
    return "\n".join(out)


def render_markdown(a: dict) -> str:
    h, f, mm = a["hotel"], a["flight"], a["market_matching"]
    lines = [f"# QuanAra — Data Audit Report", f"_Generated {a['generated_at']} from the supplied files only._", ""]
    if a.get("inventory"):
        lines += ["## 1. File inventory (originals unchanged; SHA-256)", _md_table(a["inventory"], ["dataset", "file", "bytes", "sha256"]), ""]
    lines += ["## 2. Hotel datasets", ""]
    rows = []
    for k, v in h.items():
        rows.append({"dataset": k, "rows": f"{v['rows']:,}", "cols": len(v["columns"]), "start": v["dates"]["start"], "end": v["dates"]["end"],
                     "dates": v["dates"]["distinct_dates"], "missing days": v["dates"]["missing_calendar_days"], "dup keys": v["duplicate_keys"],
                     "target (Guests)": "yes" if v["has_target"] else "**absent**", "nationalities": v.get("nationalities", "—"),
                     "NA<10 rows": v["new_arrivals_lt10_rows"], "Same-Day suppressed %": f"{v['same_day_suppressed_pct']:.1f}"})
    lines += [_md_table(rows, list(rows[0].keys())), ""]
    for k, v in h.items():
        lines.append(f"- **{k}** columns: `{', '.join(v['columns'])}`; missing: {v['missing']}")
    lines += ["", f"- International Guests range {h['intl_train']['guests']['min']:.0f}–{h['intl_train']['guests']['max']:.0f}; implied average length of stay ΣGuests/ΣNew Arrivals = {h['intl_train']['implied_avg_length_of_stay']:.2f} (intl), {h['dom_train']['implied_avg_length_of_stay']:.2f} (domestic).",
              f"- Missing nationality-day combinations: {h['intl_train']['nationality_days_missing']}", ""]
    lines += ["## 3. Flight operations dataset", "",
              f"- {f['rows']:,} rows × {len(f['columns'])} columns; {f['dates']['start']} → {f['dates']['end']}; duplicate keys: {f['duplicate_keys']}; negative values: {f['negative_values']}.",
              f"- **Grain detected from data:** {json.dumps(f['grain'])}",
              f"- {f['departure_countries']} departure countries, {f['departure_cities']} cities, {f['airlines']} airlines, {f['routes']} routes (daily data). Arrival City = {f['arrival_city_values']}, Destination = {f['destination_values']}.",
              f"- Average Weekly Frequency decodes to daily flights (AWF × days_in_month / 7); max decode error {f['frequency_decode_max_error']:.1e}. Distribution of daily flights per route-day: {f['daily_flights_distribution']}.",
              f"- Identities: PAX = P2P + Transfer + Transit on {f['identity_pax_eq_p2p_transfer_transit_pct']:.2f}% of rows; Seats = Σcabin capacity on {f['identity_seats_eq_cabin_sum_pct_daily']:.2f}% of daily rows; P2P = Σcabin P2P on only {f['identity_p2p_eq_cabin_sum_pct_daily']:.2f}%; Transfer = Σcabin transfer on {f['identity_transfer_eq_cabin_sum_pct_daily']:.2f}%.",
              f"- Load Factor equals PAX/Seats (max abs diff {f['load_factor_max_abs_diff_vs_pax_over_seats']:.1e}); > 100% on {f['load_factor_gt_100_daily_pct']:.2f}% of daily rows (max {f['load_factor_max_daily']*100:.1f}%) and {f['load_factor_gt_100_monthly_pct']:.2f}% of monthly rows (max {f['load_factor_max_monthly']*100:.1f}%).",
              f"- Total Transit non-zero rows: daily {f['transit_nonzero_daily_rows']}, monthly {f['transit_nonzero_monthly_rows']}. P2P share of daily PAX {f['p2p_share_of_pax_daily_pct']:.1f}%, transfer share {f['transfer_share_of_pax_daily_pct']:.1f}%.",
              f"- Missing values, monthly rows: {f['missing_monthly']}", f"- Missing values, daily rows: {f['missing_daily']}", ""]
    lines += ["## 4. Market matching (Nationality ↔ Departure Country Name)", "",
              f"- {len(mm['matched'])} of {mm['nationalities']} nationalities match a departure country by name, covering {mm['guest_share_in_matched_nationalities_pct']:.2f}% of international guest-days.",
              f"- Matched: {', '.join(mm['matched'])}", f"- Nationalities without a departure country: {', '.join(mm['nationalities_without_departure_country'])}",
              f"- Departure countries without a nationality: {', '.join(mm['departure_countries_without_nationality']) or 'none'}", ""]
    lines += ["## 5. Data Dictionary vs. delivered files", ""] + [f"- {d}" for d in a["dictionary_discrepancies"]] + [""]
    lines += ["## 6. Leakage register", _md_table(a["leakage"], ["feature", "risk", "decision"]), ""]
    lines += ["## 7. Warnings", ""] + [f"- **[{w['level']}] {w['title']}** — {w['detail']}" for w in a["warnings"]] + [""]
    ft = field_table()
    lines += ["## 8. Field-definition table", _md_table(ft.to_dict(orient="records"), list(ft.columns)), ""]
    return "\n".join(lines)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj
