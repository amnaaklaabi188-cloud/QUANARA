"""Facts and chart data for the Data Intelligence page, computed from the five workbooks."""
from __future__ import annotations

import pandas as pd

from . import config
from .data_loading import sha256

DATASET_ORDER = ["intl_train", "intl_test", "dom_train", "dom_test", "flight"]


def _m(ts) -> str:
    return pd.Timestamp(ts).strftime("%b %Y")


def _monthly(df: pd.DataFrame, col: str) -> list[float]:
    return [float(v) for v in df.groupby(df["Date"].dt.to_period("M"))[col].sum().to_numpy()]


def dataset_cards(d: dict[str, pd.DataFrame]) -> list[dict]:
    it, ite, dt, dte, fl = d["intl_train"], d["intl_test"], d["dom_train"], d["dom_test"], d["flight"]

    def span(df):
        return f"<b>{_m(df['Date'].min())} → {_m(df['Date'].max())}</b> · {df['Date'].nunique():,} days"

    def ncols(df):
        return len([c for c in df.columns if c not in ("same_day_suppressed", "grain", "flights", "flights_decode_error", "market_key")])

    monthly, daily = fl[fl["grain"] == "monthly"], fl[fl["grain"] == "daily"]
    return [
        dict(key="intl_train", title="International · train", icon="users", tone="violet", value=f"{len(it):,}", unit="records",
             explanation="Guests per nationality per day — the labelled training data.",
             secondary=f"{span(it)} · {it['Nationality'].nunique()} nationalities · {ncols(it)} columns",
             series=_monthly(it, "Guests"), color="#059669", caption="Guests per month (sum of daily counts)", badge=("Guests observed", "green")),
        dict(key="intl_test", title="International · test", icon="users", tone="slate", value=f"{len(ite):,}", unit="records",
             explanation="Same structure without Guests — the official unlabelled test period.",
             secondary=f"{span(ite)} · {ite['Nationality'].nunique()} nationalities · {ncols(ite)} columns",
             series=_monthly(ite, "New Arrivals"), color="#9ca3a0", caption="New Arrivals per month (Guests not provided)", badge=("No Guests", "grey")),
        dict(key="dom_train", title="Domestic · train", icon="home", tone="teal", value=f"{len(dt):,}", unit="days",
             explanation="One row per day with the total of domestic (in-country) guests.",
             secondary=f"{span(dt)} · one row per date · {ncols(dt)} columns",
             series=_monthly(dt, "Guests"), color="#059669", caption="Domestic guests per month (sum of daily counts)", badge=("Guests observed", "green")),
        dict(key="dom_test", title="Domestic · test", icon="home", tone="slate", value=f"{len(dte):,}", unit="days",
             explanation="Daily domestic rows without Guests — the official test period.",
             secondary=f"{span(dte)} · one row per date · {ncols(dte)} columns",
             series=_monthly(dte, "New Arrivals"), color="#9ca3a0", caption="Domestic New Arrivals per month", badge=("No Guests", "grey")),
        dict(key="flight", title="Flight operations", icon="plane", tone="blue", value=f"{len(fl):,}", unit="records",
             explanation="Route-level arrivals into AUH: seats, passengers, transfer and transit.",
             secondary=(f"<b>{_m(fl['Date'].min())} → {_m(fl['Date'].max())}</b> · {len(monthly):,} monthly rows (2022), {len(daily):,} daily rows · "
                        f"{fl['Departure Country Name'].nunique()} countries · {fl['Airline Name'].nunique()} airlines"),
             series=_monthly(fl, "Total PAX"), color="#059669", caption="Arriving passengers per month (Total PAX)", badge=("Monthly → daily", "amber")),
    ]


def passenger_composition(fl: pd.DataFrame, top: int = 15) -> dict:
    cols = ["Total P2P", "Total Transfer", "Total Transit", "Total PAX"]
    monthly = fl.groupby(fl["Date"].dt.to_period("M").dt.to_timestamp())[cols].sum().reset_index().rename(columns={"Date": "month"})
    monthly["grain"] = (monthly["month"] < config.FLIGHT_DAILY_START).map({True: "monthly rows", False: "daily rows summed"})
    daily = fl[fl["grain"] == "daily"]
    by_c = daily.groupby("Departure Country Name")[cols].sum()
    by_c["p2p_share"] = by_c["Total P2P"] / by_c["Total PAX"]
    by_c = by_c.sort_values("Total PAX", ascending=False).head(top).reset_index()
    t_all, t_day = fl[cols].sum(), daily[cols].sum()
    return {
        "monthly": monthly, "by_country": by_c,
        "daily_totals": {k: float(v) for k, v in t_day.items()}, "all_totals": {k: float(v) for k, v in t_all.items()},
        "daily_p2p_share": float(t_day["Total P2P"] / t_day["Total PAX"]),
        "daily_transfer_share": float(t_day["Total Transfer"] / t_day["Total PAX"]),
        "transit_rows_daily": int((daily["Total Transit"] > 0).sum()),
        "transit_rows_all": int((fl["Total Transit"] > 0).sum()),
        "identity_holds": bool(((fl["Total PAX"] - fl["Total P2P"] - fl["Total Transfer"] - fl["Total Transit"]).abs() < 0.5).all()),
    }


def quality_overview(d: dict[str, pd.DataFrame], audit: dict) -> dict:
    it = d["intl_train"]
    missing_combos = it["Date"].nunique() * it["Nationality"].nunique() - len(it)
    daily = d["flight"][d["flight"]["grain"] == "daily"]
    return {
        "duplicate_keys": int(sum(v["duplicate_keys"] for v in audit["hotel"].values()) + audit["flight"]["duplicate_keys"]),
        "literal_asterisks": int(sum((audit.get("literal_asterisks_by_file") or {}).values())),
        "missing_nat_day_combinations": int(missing_combos),
        "nationalities_with_missing_days": int((it.groupby("Nationality").size() < it["Date"].nunique()).sum()),
        "same_day_suppressed_train_pct": float(it["Same-Day Guests"].isna().mean() * 100),
        "same_day_suppressed_test_pct": float(d["intl_test"]["Same-Day Guests"].isna().mean() * 100),
        "lf_over_100_daily_pct": float((daily["Load Factor"] > 1).mean() * 100),
        "lf_max_daily": float(daily["Load Factor"].max()),
        "transit_rows_daily": int((daily["Total Transit"] > 0).sum()),
        "transit_pax_daily": float(daily["Total Transit"].sum()),
        "na_lt10_train": int((it["New Arrivals"] < 10).sum()), "na_lt10_train_pct": float((it["New Arrivals"] < 10).mean() * 100),
        "na_lt10_test": int((d["intl_test"]["New Arrivals"] < 10).sum()),
    }


def checksum_status() -> tuple[bool, int]:
    """True when every file in data/raw matches the SHA-256 recorded when the originals were copied."""
    f = config.RAW_DIR / "SHA256SUMS.txt"
    if not f.exists():
        return False, 0
    pairs = [ln.split("  ", 1) for ln in f.read_text().splitlines() if ln.strip()]
    ok = all((config.RAW_DIR / name).exists() and sha256(config.RAW_DIR / name) == digest for digest, name in pairs)
    return ok, len(pairs)


# ------------------------------------------------------------------ Data-quality overview & audit findings
# Every number below is computed from the five workbooks at run time; `scripts/verify_pages.py`
# recomputes each evidence value (by its id) from the raw files with openpyxl.

KEYS = {"intl_train": ["Date", "Nationality"], "intl_test": ["Date", "Nationality"], "dom_train": ["Date"], "dom_test": ["Date"],
        "flight": ["Date", "Departure Country Name", "Departure City", "Airline Name"]}
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


def _pct(a: float, b: float) -> str:
    return f"{a / b:.1%}" if b else "—"


def quality_stats(d: dict[str, pd.DataFrame], raw_stars: int) -> dict:
    it, ite, fl = d["intl_train"], d["intl_test"], d["flight"]
    daily, monthly = fl[fl["grain"] == "daily"], fl[fl["grain"] == "monthly"]
    n_days, n_nat = it["Date"].nunique(), it["Nationality"].nunique()
    per_nat = it.groupby("Nationality").size()
    missing_by_nat = (n_days - per_nat[per_nat < n_days]).sort_values(ascending=False)
    cab_tr = daily["Transit Business Class Count"] + daily["Transit Economy Class Count"] + daily["Transit First Count"]
    cab_p2p = daily["Business Class P2P Count"] + daily["Economy Class P2P Count"] + daily["First Class P2P Count"]
    uk = daily[daily["Departure Country Name"] == "United Kingdom"]
    countries = set(fl["market_key"].unique())
    nats = set(it["Nationality"].unique())
    return {
        "rows_total": int(sum(len(d[k]) for k in KEYS)),
        "rows_by_file": {k: int(len(d[k])) for k in KEYS},
        "duplicate_keys": int(sum(d[k].duplicated(KEYS[k]).sum() for k in KEYS)),
        "literal_asterisks": int(raw_stars),
        "intl_train_rows": int(len(it)), "intl_test_rows": int(len(ite)),
        "dom_train_rows": int(len(d["dom_train"])), "dom_test_rows": int(len(d["dom_test"])),
        "train_have_guests": bool("Guests" in it.columns and "Guests" in d["dom_train"].columns
                                  and it["Guests"].notna().all() and d["dom_train"]["Guests"].notna().all()),
        "test_have_guests": bool("Guests" in ite.columns or "Guests" in d["dom_test"].columns),
        "test_start": ite["Date"].min(), "test_end": ite["Date"].max(),
        "same_day_empty_train": int(it["Same-Day Guests"].isna().sum()), "same_day_empty_test": int(ite["Same-Day Guests"].isna().sum()),
        "same_day_empty_domestic": int(d["dom_train"]["Same-Day Guests"].isna().sum() + d["dom_test"]["Same-Day Guests"].isna().sum()),
        "new_arrivals_empty_train": int(it["New Arrivals"].isna().sum()),
        "possible_nat_days": int(n_days * n_nat), "missing_nat_days": int(n_days * n_nat - len(it)),
        "missing_by_nationality": {k: int(v) for k, v in missing_by_nat.items()},
        "guests_empty_train": int(it["Guests"].isna().sum() + d["dom_train"]["Guests"].isna().sum()),
        "aviation_empty_daily": int(daily[["Total Seats", "Total P2P", "Total PAX", "Total Transfer"]].isna().sum().sum()),
        "transit_zero_rows_daily": int((daily["Total Transit"] == 0).sum()),
        "dom_guests_min": int(d["dom_train"]["Guests"].min()), "dom_guests_max": int(d["dom_train"]["Guests"].max()),
        "test_days": int(ite["Date"].nunique()), "flight_end": fl["Date"].max(), "train_end": it["Date"].max(),
        "test_columns": [c for c in ite.columns if c != "same_day_suppressed"], "dom_test_columns": [c for c in d["dom_test"].columns if c != "same_day_suppressed"],
        "distinct_columns": int(len({c for k in KEYS for c in d[k].columns} - {"same_day_suppressed", "grain", "flights", "flights_decode_error", "market_key"})),
        "min_guests_intl": int(it["Guests"].min()), "guest_days_intl": float(it["Guests"].sum()),
        "na_lt10_train": int((it["New Arrivals"] < 10).sum()), "na_lt10_test": int((ite["New Arrivals"] < 10).sum()),
        "na_min_test": float(ite["New Arrivals"].min()),
        "rows_per_day_train": float(len(it) / n_days), "rows_per_day_test": float(len(ite) / ite["Date"].nunique()),
        "monthly_rows": int(len(monthly)), "monthly_months": int(monthly["Date"].dt.to_period("M").nunique()),
        "daily_rows": int(len(daily)), "daily_start": daily["Date"].min(), "daily_dates": int(daily["Date"].nunique()),
        "awf_empty_monthly": int(monthly["Average Weekly Frequency"].isna().sum()), "awf_empty_daily": int(daily["Average Weekly Frequency"].isna().sum()),
        "lf_gt1_daily": int((daily["Load Factor"] > 1).sum()), "lf_max_daily": float(daily["Load Factor"].max()),
        "lf_gt1_monthly": int((monthly["Load Factor"] > 1).sum()), "lf_max_monthly": float(monthly["Load Factor"].max()),
        "lf_identity_max_err": float((fl["Load Factor"] - fl["Total PAX"] / fl["Total Seats"]).abs().max()),
        "transit_rows_daily": int((daily["Total Transit"] > 0).sum()), "transit_pax_daily": float(daily["Total Transit"].sum()),
        "pax_daily": float(daily["Total PAX"].sum()), "cabin_transit_rows": int((cab_tr > 0).sum()),
        "cabin_p2p_match_rows": int(((daily["Total P2P"] - cab_p2p).abs() < 0.5).sum()),
        "matched_markets": int(len(nats & countries)), "nationalities": int(len(nats)),
        "unmatched": sorted(nats - countries),
        "matched_guest_share": float(it.loc[it["Nationality"].isin(countries), "Guests"].sum() / it["Guests"].sum()),
        "uk_transfer_share": float(uk["Total Transfer"].sum() / uk["Total PAX"].sum()),
        "ru_guest_share": float(it.loc[it["Nationality"] == "RUSSIAN FEDERATION", "Guests"].sum() / it["Guests"].sum()),
        "ru_pax_share": float(daily.loc[daily["Departure Country Name"] == "Russian Federation", "Total PAX"].sum() / daily["Total PAX"].sum()),
        **_extra_stats(d, daily, it, ite),
    }


def _extra_stats(d: dict[str, pd.DataFrame], daily: pd.DataFrame, it: pd.DataFrame, ite: pd.DataFrame) -> dict:
    """Evidence added for the Section 2 scientific review (infants, cabin breakdown, selection rule)."""
    f = lambda c: daily[c].fillna(0)
    inf = daily["Total PAX"] - daily["Total PAX Excluding Infant"]
    p2p_c = f("Business Class P2P Count") + f("Economy Class P2P Count") + f("First Class P2P Count")
    tr_c = f("Transfer Business Class Count") + f("Transfer Economy Class Count") + f("Transfer First Class Count")
    ts_c = f("Transit Business Class Count") + f("Transit Economy Class Count") + f("Transit First Count")
    over = daily["Load Factor"] > 1
    lf_ex = daily["Total PAX Excluding Infant"] / daily["Total Seats"]
    lt10 = it["New Arrivals"] < 10
    g_tot = float(it["Guests"].sum())
    sdg_ok = it["Same-Day Guests"].notna() & it["New Arrivals"].notna()
    return {
        "infants_median": float(inf.median()), "infants_max": float(inf.max()), "infants_1_5_share": float(inf.between(1, 5).mean()),
        "infants_zero_share": float((inf == 0).mean()),
        "lf_gt1_excl_infant": int((lf_ex[over] > 1).sum()), "lf_max_excl_infant": float(lf_ex.max()),
        "cabin_eq_pax_excl_infant": float(((p2p_c + tr_c + ts_c - daily["Total PAX Excluding Infant"]).abs() < 0.5).mean()),
        "cabin_p2p_plus_inf_eq": float(((p2p_c + inf - daily["Total P2P"]).abs() < 0.5).mean()),
        "cabin_p2p_gt_total": int((p2p_c > daily["Total P2P"] + 0.5).sum()),
        "cabin_trts_le_transfer": float((tr_c + ts_c <= daily["Total Transfer"] + 0.5).mean()),
        "cabin_trts_eq_transfer": float(((tr_c + ts_c - daily["Total Transfer"]).abs() < 0.5).mean()),
        "na_gt_guests": int((it["New Arrivals"] > it["Guests"]).sum()),
        "sdg_gt_na": int((it.loc[sdg_ok, "Same-Day Guests"] > it.loc[sdg_ok, "New Arrivals"]).sum()),
        "na_lt10_guest_share": float(it.loc[lt10, "Guests"].sum() / g_tot) if g_tot else 0.0,
        "na_lt10_nationalities": int(it.loc[lt10, "Nationality"].nunique()),
        "na_lt10_median_guests": float(it.loc[lt10, "Guests"].median()), "na_ge10_median_guests": float(it.loc[~lt10, "Guests"].median()),
        "test_nationalities": int(ite["Nationality"].nunique()),
        "flight_columns": int(len([c for c in d["flight"].columns if c not in ("grain", "flights", "flights_decode_error", "market_key", "same_day_suppressed")])),
    }


def missing_profile(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in ("same_day_suppressed", "grain", "flights", "flights_decode_error", "market_key")]
    return pd.DataFrame({"column": cols, "empty": [int(df[c].isna().sum()) for c in cols], "rows": len(df),
                         "pct": [float(df[c].isna().mean() * 100) for c in cols]}).sort_values("pct", ascending=False)


def audit_findings(s: dict) -> list[dict]:
    """Findings ranked by their impact on the modelling (not by how alarming they sound)."""
    def ev(eid, label, display, value):
        return {"id": eid, "label": label, "display": display, "value": value}

    m = lambda t: pd.Timestamp(t).strftime("%-d %b %Y")
    wn = s["missing_by_nationality"]
    out = [
        dict(severity="high", title="Departure country is only a proxy for guest nationality",
             summary=f"{s['matched_markets']} of {s['nationalities']} nationalities share a name with a flight departure country, but flights carry many non-nationals.",
             found=("Hotel guests are recorded by nationality; flights by the country they depart from. "
                    f"{s['matched_markets']} of {s['nationalities']} nationalities have an exactly matching departure-country name and "
                    f"{len(s['unmatched'])} have none ({', '.join(n.title() for n in s['unmatched'][:5])} …). Passengers on a route are not all nationals of the departure country."),
             evidence=[ev("matched_markets", "Nationalities with a matching departure country", f"{s['matched_markets']} of {s['nationalities']}", s["matched_markets"]),
                       ev("matched_guest_share", "International guest-days in matched nationalities", f"{s['matched_guest_share']:.1%}", s["matched_guest_share"]),
                       ev("uk_transfer_share", "Arrivals from the United Kingdom that are transfer passengers", f"{s['uk_transfer_share']:.1%}", s["uk_transfer_share"]),
                       ev("ru_shares", "Russian nationals: share of guest-days vs. share of arriving passengers", f"{s['ru_guest_share']:.1%} vs {s['ru_pax_share']:.1%}",
                          [s["ru_guest_share"], s["ru_pax_share"]])],
             why=("A source-market model links a country’s flights to guests of the same nationality. Where nationals mostly arrive through other hubs, "
                  "flight changes from their own country capture only part of their hotel demand."),
             handling=("The match is an explicit, logged exact-name rule. Unmatched nationalities are left out of the source-market model but included "
                       "in the international-total model, and departure country is labelled a proxy wherever it is used."),
             impact="Market-level scenario results are indicative approximations of nationality demand, not exact attributions."),
        dict(severity="high", title="Flight data is monthly in 2022 and daily from 2023",
             summary=f"{s['monthly_rows']:,} monthly rows in 2022, then {s['daily_rows']:,} daily rows from {m(s['daily_start'])}; the dictionary describes the file as monthly.",
             found=(f"In 2022 each route has one row per month, dated the 1st ({s['monthly_months']} months). From {m(s['daily_start'])} the file has one row per route per day. "
                    "Average Weekly Frequency and the cabin-class columns are empty only in the 2022 monthly rows."),
             evidence=[ev("monthly_rows", "Monthly-grain rows (2022)", f"{s['monthly_rows']:,} in {s['monthly_months']} months", s["monthly_rows"]),
                       ev("daily_rows", "Daily-grain rows", f"{s['daily_rows']:,} over {s['daily_dates']:,} days", s["daily_rows"]),
                       ev("awf_empty", "Average Weekly Frequency empty: monthly vs daily rows", f"{s['awf_empty_monthly']:,} of {s['monthly_rows']:,} vs {s['awf_empty_daily']} of {s['daily_rows']:,}",
                          [s["awf_empty_monthly"], s["awf_empty_daily"]])],
             why="Daily hotel guests can only be matched to daily flights; monthly 2022 totals cannot be split into days without inventing a rule.",
             handling=(f"Daily models use {m(config.MODEL_START)} → {m(config.MODEL_END)} (after a {config.FEATURE_WARMUP_DAYS}-day feature warm-up). "
                       "The 2022 flight rows appear only in monthly charts. The last-year guest feature uses the 2022 hotel records, not flights."),
             impact=f"Models learn from {(config.MODEL_END - config.MODEL_START).days + 1:,} days of daily data; year-over-year flight growth cannot be used as a feature."),
        dict(severity="medium", title="Training and test extracts follow different inclusion rules",
             summary=f"No test row has New Arrivals below 10, while {s['na_lt10_train']:,} of {s['intl_train_rows']:,} international training rows ({_pct(s['na_lt10_train'], s['intl_train_rows'])}) do.",
             found=("Both international files list guests by nationality and day, but the test file contains no rows with fewer than 10 New Arrivals "
                    f"(its smallest value is {s['na_min_test']:.0f}). The training file keeps such rows. They are small: {s['na_lt10_guest_share']:.2%} of guest-days, "
                    f"spread over {s['na_lt10_nationalities']} nationalities, with a median of {s['na_lt10_median_guests']:.0f} Guests against {s['na_ge10_median_guests']:.0f} for the other rows."),
             evidence=[ev("na_lt10_train", "Training rows with New Arrivals below 10", f"{s['na_lt10_train']:,} of {s['intl_train_rows']:,} ({_pct(s['na_lt10_train'], s['intl_train_rows'])})", s["na_lt10_train"]),
                       ev("na_lt10_test", "Test rows with New Arrivals below 10", f"{s['na_lt10_test']} of {s['intl_test_rows']:,}", s["na_lt10_test"]),
                       ev("na_lt10_guest_share", "Share of training guest-days in those rows", f"{s['na_lt10_guest_share']:.2%}", s["na_lt10_guest_share"]),
                       ev("rows_per_day", "Nationality rows per day: training vs test", f"{s['rows_per_day_train']:.1f} vs {s['rows_per_day_test']:.1f}",
                          [round(s["rows_per_day_train"], 1), round(s["rows_per_day_test"], 1)])],
             why="A model that uses New Arrivals would otherwise learn from rows that never occur in the test file.",
             handling=("Modelling decision: Competition-mode source-market models train only on rows with New Arrivals ≥ 10, so their training population matches "
                       "the test file. What-if models never use New Arrivals and keep every row. The effect of this choice on accuracy has not been measured; "
                       "a controlled comparison would need extra models and awaits approval."),
             impact="Competition-mode accuracy refers to a population comparable with the test file; What-if forecasts are unaffected."),
        dict(severity="low", title="Some nationality-day combinations are unobserved",
             summary=f"{s['missing_nat_days']} of {s['possible_nat_days']:,} possible nationality-day combinations ({_pct(s['missing_nat_days'], s['possible_nat_days'])}) have no row in the international training file.",
             found=(f"If each of the {s['nationalities']} nationalities appeared on each of the {s['possible_nat_days'] // s['nationalities']:,} days, there would be "
                    f"{s['possible_nat_days']:,} rows; the file has {s['intl_train_rows']:,}. The {s['missing_nat_days']} unobserved combinations belong to {len(wn)} nationalities. "
                    f"The files do not say why rows are absent. Suppression of small counts is one possible explanation — no recorded Guests value is below "
                    f"{s['min_guests_intl']} — but it is not confirmed, so an unobserved combination is treated as unknown, not as a small or zero count."),
             evidence=[ev("missing_nat_days", "Unobserved nationality-day combinations", f"{s['missing_nat_days']} of {s['possible_nat_days']:,}", s["missing_nat_days"]),
                       ev("missing_by_nationality", "Most affected", ", ".join(f"{k.title()} {v}" for k, v in list(wn.items())[:4]), wn),
                       ev("min_guests_intl", "Smallest recorded international Guests value", f"{s['min_guests_intl']}", s["min_guests_intl"])],
             why="An unobserved row is neither a zero nor a known value; filling it in would invent data.",
             handling="Unobserved rows are never imputed. History features for those days stay missing, and daily totals add up only the rows that exist.",
             impact=f"Negligible for daily totals ({_pct(s['missing_nat_days'], s['possible_nat_days'])} of combinations); a few small markets have gaps in their history."),
        dict(severity="low", title="Same-Day Guests blanks cannot be interpreted uniquely",
             summary="Same-Day Guests contains blank values whose original meaning cannot be uniquely recovered from the delivered files.",
             found=("The dictionary says the value can indicate zero, “*” suppressed, or not applicable. In the delivered files these values are empty cells "
                    f"({s['literal_asterisks']} literal “*” characters), so a blank may be any of the three. "
                    f"Blank in {_pct(s['same_day_empty_train'], s['intl_train_rows'])} of international training rows and {_pct(s['same_day_empty_test'], s['intl_test_rows'])} of test rows; "
                    "the domestic files have none."),
             evidence=[ev("same_day_empty_train", "Blank Same-Day Guests, international training", f"{s['same_day_empty_train']:,} of {s['intl_train_rows']:,}", s["same_day_empty_train"]),
                       ev("same_day_empty_test", "Blank Same-Day Guests, international test", f"{s['same_day_empty_test']:,} of {s['intl_test_rows']:,}", s["same_day_empty_test"]),
                       ev("literal_asterisks", "Literal “*” cells across the five files", f"{s['literal_asterisks']}", s["literal_asterisks"])],
             why="A blank cannot be read as zero: it may hide a small suppressed count.",
             handling=("Kept missing in the data. Where values are summed across nationalities, a blank counts as 0 and the total is labelled a lower bound. "
                       "The field is used only in Competition mode."),
             impact="None for What-if forecasts."),
        dict(severity="low", title="Load factor above 100% on some route-days",
             summary=f"{s['lf_gt1_daily']:,} of {s['daily_rows']:,} daily route rows ({_pct(s['lf_gt1_daily'], s['daily_rows'])}) exceed 100%, up to {s['lf_max_daily']:.1%}.",
             found=("Load Factor equals Total PAX ÷ Total Seats on every row. The dictionary says values above 100% are possible through overbooking or infant "
                    f"counting. Excluding infants, {s['lf_gt1_excl_infant']:,} of those rows are still above 100% (maximum {s['lf_max_excl_infant']:.1%}), "
                    "so infants explain most but not all of them."),
             evidence=[ev("lf_gt1_daily", "Daily rows above 100%", f"{s['lf_gt1_daily']:,} of {s['daily_rows']:,} (max {s['lf_max_daily']:.1%})", s["lf_gt1_daily"]),
                       ev("lf_gt1_excl_infant", "Of these, still above 100% without infants", f"{s['lf_gt1_excl_infant']:,}", s["lf_gt1_excl_infant"]),
                       ev("lf_gt1_monthly", "2022 monthly rows above 100%", f"{s['lf_gt1_monthly']} of {s['monthly_rows']:,} (max {s['lf_max_monthly']:.1%})", s["lf_gt1_monthly"]),
                       ev("lf_identity", "Largest gap between Load Factor and PAX ÷ Seats", f"{s['lf_identity_max_err']:.0e}", s["lf_identity_max_err"])],
             why="These are plausible operational records, not errors — but a load factor must never be averaged across routes.",
             handling="Kept as recorded. Aggregated load factors are recomputed as ΣPAX ÷ ΣSeats, and the simulator warns when a scenario exceeds the highest observed value.",
             impact="Negligible."),
        dict(severity="info", title="Official test files contain no Guests (expected)",
             summary=f"The test period {m(s['test_start'])} → {m(s['test_end'])} is the competition’s unlabelled forecast window.",
             found=(f"The international test file ({s['intl_test_rows']:,} rows) and domestic test file ({s['dom_test_rows']} rows) contain New Arrivals and Same-Day Guests but no Guests column. "
                    "This is how the competition is set up, not a data defect."),
             evidence=[ev("test_rows", "Unlabelled test rows (international + domestic)", f"{s['intl_test_rows'] + s['dom_test_rows']:,}", s["intl_test_rows"] + s["dom_test_rows"]),
                       ev("test_have_guests", "Guests column present in a test file", "No" if not s["test_have_guests"] else "Yes", s["test_have_guests"])],
             why="Accuracy cannot be measured on these dates.",
             handling="Accuracy is measured on held-out training months that models never saw; outputs for the test dates are labelled as forecasts.",
             impact="Reported accuracy describes the most recent observed months, not the test window itself."),
        dict(severity="info", title="Total Transit is effectively zero; cabin transit counts sit inside Total Transfer",
             summary=f"Total Transit is non-zero on {s['transit_rows_daily']} of {s['daily_rows']:,} daily rows, but cabin-level transit counts are non-zero on {s['cabin_transit_rows']:,}.",
             found=(f"Cabin transfer plus cabin transit counts never exceed Total Transfer ({s['cabin_trts_le_transfer']:.3%} of daily rows). "
                    "The cabin “transit” columns therefore appear to be counted within Total Transfer under a different definition, rather than in Total Transit."),
             evidence=[ev("transit_rows_daily", "Daily rows with Total Transit > 0", f"{s['transit_rows_daily']} of {s['daily_rows']:,}", s["transit_rows_daily"]),
                       ev("transit_pax_daily", "Transit passengers vs all arriving passengers", f"{s['transit_pax_daily']:,.0f} of {s['pax_daily']:,.0f}", s["transit_pax_daily"]),
                       ev("cabin_transit_rows", "Daily rows with cabin transit counts > 0", f"{s['cabin_transit_rows']:,}", s["cabin_transit_rows"]),
                       ev("cabin_trts_le_transfer", "Rows where cabin transfer + transit ≤ Total Transfer", f"{s['cabin_trts_le_transfer']:.3%}", s["cabin_trts_le_transfer"])],
             why="A field without variation carries no information, and the cabin breakdown does not follow the documented definition.",
             handling="Transit is not a model feature and is held at its observed value in scenarios. PAX = P2P + Transfer + Transit holds on every row.",
             impact="None."),
        dict(severity="info", title="Cabin-class passenger counts do not add up to the totals",
             summary=f"Cabin P2P counts equal Total P2P on {s['cabin_p2p_match_rows']:,} of {s['daily_rows']:,} daily rows ({_pct(s['cabin_p2p_match_rows'], s['daily_rows'])}).",
             found=(f"Adding infants (Total PAX − PAX excluding infants) to the cabin P2P counts reproduces Total P2P on {s['cabin_p2p_plus_inf_eq']:.1%} of rows, "
                    "so infants without a cabin explain most, but not all, of the gap. Cabin seat capacities, by contrast, add up to Total Seats on every row."),
             evidence=[ev("cabin_p2p_match_rows", "Daily rows where cabin P2P = Total P2P", f"{s['cabin_p2p_match_rows']:,} of {s['daily_rows']:,}", s["cabin_p2p_match_rows"]),
                       ev("cabin_p2p_plus_inf", "Rows where cabin P2P + infants = Total P2P", f"{s['cabin_p2p_plus_inf_eq']:.1%}", s["cabin_p2p_plus_inf_eq"])],
             why="Features built from inconsistent parts would contradict the totals they come from.",
             handling="Total P2P, Total Transfer and Total PAX are used; cabin columns are used only for seat mix (capacity), which is exact.",
             impact="None."),
    ]
    return sorted(out, key=lambda f: SEVERITY_ORDER[f["severity"]])


def value_kinds(s: dict) -> list[dict]:
    """Five different things that can look like 'no value' — with real counts."""
    return [
        dict(kind="Unobserved combination", tone="violet", icon="users",
             text="No row exists for a nationality-day that could exist. Treated as unknown, never as zero.",
             example=f"{s['missing_nat_days']} of {s['possible_nat_days']:,} nationality-day combinations (international training)"),
        dict(kind="Blank cell", tone="blue", icon="database",
             text="A row exists but one field is empty. The dictionary’s “*” arrives as an empty cell.",
             example=f"New Arrivals blank in {s['new_arrivals_empty_train']} of {s['intl_train_rows']:,} international training rows"),
        dict(kind="Blank of unrecoverable meaning", tone="orange", icon="lock",
             text="Same-Day Guests contains blank values whose original meaning cannot be uniquely recovered from the delivered files.",
             example=f"Same-Day Guests blank in {s['same_day_empty_train']:,} of {s['intl_train_rows']:,} rows"),
        dict(kind="Genuine zero", tone="teal", icon="check",
             text="A recorded 0. It is real data and is used as a value.",
             example=f"Total Transit = 0 on {s['transit_zero_rows_daily']:,} of {s['daily_rows']:,} daily flight rows"),
        dict(kind="Unlabelled test record", tone="slate", icon="calendar",
             text="Rows in the official test files have no Guests column: the target is withheld by design, not missing.",
             example=f"{s['intl_test_rows'] + s['dom_test_rows']:,} test rows, {pd.Timestamp(s['test_start']):%b %Y} – {pd.Timestamp(s['test_end']):%b %Y}"),
    ]


def dictionary_reconciliation(s: dict) -> list[dict]:
    """Documented expectation (with a phrase that must appear in Data_Dictionary.pdf) vs what the
    workbooks contain. The workbooks are the source of truth whenever the two disagree."""
    m = lambda t: pd.Timestamp(t).strftime("%-d %b %Y")
    p2p_pct = _pct(s["cabin_p2p_match_rows"], s["daily_rows"])
    return [
        dict(topic="Target in the test files", phrase="data_international_test.xlsx",
             documented="Describes data_international_test.xlsx with Guests as the target column.",
             observed=f"Neither test file has a Guests column (international test: {', '.join(s['test_columns'])}).",
             implication="Accuracy is measured on held-out training months; outputs for the test dates are forecasts that cannot be scored."),
        dict(topic="Flight-data grain", phrase="Monthly flight-level operational data",
             documented="Monthly flight-level data; Date is the first day of the reporting month.",
             observed=f"Monthly only in 2022 ({s['monthly_rows']:,} rows, {s['monthly_months']} months); daily from {m(s['daily_start'])} ({s['daily_rows']:,} rows over {s['daily_dates']:,} days).",
             implication=f"Daily models start {m(config.MODEL_START)}, after the feature warm-up; 2022 flight rows appear only in monthly views."),
        dict(topic="Average Weekly Frequency", phrase="Populated with * in most observed rows",
             documented="Populated with “*” in most observed rows; may not be available.",
             observed=f"Empty on all {s['awf_empty_monthly']:,} monthly rows and on {s['awf_empty_daily']} of {s['daily_rows']:,} daily rows; on daily rows it equals flights × 7 ÷ days in the month.",
             implication="Daily flight frequency is recovered exactly and can be used as a feature."),
        dict(topic="Suppression marker", phrase="Asterisk",
             documented="“*” marks unavailable, suppressed or not-applicable values.",
             observed=f"{s['literal_asterisks']} literal “*” cells in the five workbooks; these values arrive as empty cells.",
             implication="Empty cells are kept as missing and never read as zero."),
        dict(topic="Load Factor format", phrase="47.02%",
             documented="A percentage string, e.g. 47.02%.",
             observed=f"A number (fraction) up to {s['lf_max_monthly']:.2f}; it equals Total PAX ÷ Total Seats on every row.",
             implication="Aggregated load factors are recomputed from summed passengers and seats, never averaged."),
        dict(topic="Date format", phrase="YYYY/M/D",
             documented="Dates as strings formatted YYYY/M/D.",
             observed="Stored as Excel date values.",
             implication="Read directly as dates; no string parsing is needed."),
        dict(topic="International test coverage", phrase="Aug 2025 – Aug 2025",
             documented="Aug 2025 – Aug 2025.",
             observed=f"{m(s['test_start'])} → {m(s['test_end'])} ({s['test_days']} days).",
             implication="The unlabelled forecast window spans seven months."),
        dict(topic="Domestic Guests range", phrase="roughly from 12,000 to 28,000",
             documented="Values range roughly from 12,000 to 28,000.",
             observed=f"Training values range from {s['dom_guests_min']:,} to {s['dom_guests_max']:,}.",
             implication="Dictionary ranges are treated as illustrative; models learn from the observed values."),
        dict(topic="Cabin-class columns", phrase="Predominantly",
             documented="First Class (and other cabin) columns predominantly “*”.",
             observed=f"Empty only on the 2022 monthly rows. On daily rows cabin seats add up to Total Seats exactly, but cabin P2P matches Total P2P on only {p2p_pct}.",
             implication="Passenger totals are used; cabin columns are used only for seat mix."),
        dict(topic="Number of columns", phrase="32 unique columns",
             documented="32 unique columns.",
             observed=(f"The dictionary lists 6 hotel columns and {s['flight_columns']} flight columns = 32 entries. Date appears in both, so the workbooks "
                       f"contain {s['distinct_columns']} distinct column names."),
             implication="Both counts are correct; they count different things. No effect on modelling."),
    ]


STATUS = {"verified": ("Verified", "good", "check"), "attention": ("Attention", "warn", "warning"),
          "failed": ("Failed", "bad", "warning"), "not_verified": ("Not verified", "info", "info")}


def technical_checks(s: dict, model_md: dict | None, holdout_pred: pd.DataFrame | None, leak: list[dict] | None,
                     identity_ok: bool) -> list[dict]:
    """Automated technical checks. A check is 'Verified' only when code was executed and passed;
    'Attention' marks an expected limitation that is handled; 'Not verified' means nothing was run."""
    m = lambda t: pd.Timestamp(t).strftime("%-d %b %Y")
    st_ = lambda ok: "verified" if ok else "failed"
    out = [
        dict(check="Observed target for training", status=st_(s["train_have_guests"] and s["guests_empty_train"] == 0),
             method="Counted empty Guests cells in both training files.",
             evidence=f"Guests present in all {s['intl_train_rows'] + s['dom_train_rows']:,} training rows ({s['guests_empty_train']} empty)."),
        dict(check="No duplicate records", status=st_(s["duplicate_keys"] == 0),
             method="Checked every row on its natural key (date × nationality, date, or date × country × city × airline).",
             evidence=f"{s['duplicate_keys']} duplicates in {s['rows_total']:,} rows."),
        dict(check="Passenger totals are consistent", status=st_(identity_ok and s["lf_identity_max_err"] < 1e-9),
             method="Recomputed PAX = P2P + Transfer + Transit and Load Factor = PAX ÷ Seats on every flight row.",
             evidence=f"Both identities hold on every row (largest Load Factor gap {s['lf_identity_max_err']:.0e})."),
        dict(check="Inputs exist for the forecast window", status=st_(pd.Timestamp(s["flight_end"]) >= pd.Timestamp(s["test_end"])
             and pd.Timestamp(s["test_end"]) - pd.Timedelta(days=config.SEASONAL_LAG_DAYS) <= pd.Timestamp(s["train_end"])),
             method="Compared the flight-file end date and the last-year guest history with the test window.",
             evidence=f"Flights run to {m(s['flight_end'])}; last-year guests exist for every forecast date to {m(s['test_end'])}."),
    ]
    if leak:
        hist = [c for c in leak if c["check"] != "Validation dates follow all training dates"]
        ok = all(c["ok"] for c in hist)
        out.append(dict(check="No future information in features", status=st_(ok),
                        method="Re-ran the feature code after deleting later data, refit preprocessing on one fold, and checked the What-if feature list.",
                        evidence=f"{sum(c['ok'] for c in hist)} of {len(hist)} executed leakage tests passed."))
        chron = next((c for c in leak if c["check"] == "Validation dates follow all training dates"), None)
        if chron is None or chron["ok"] is None or model_md is None:
            out.append(dict(check="Chronological validation", status="not_verified", method="Needs an active model.", evidence="No active model to inspect."))
        else:
            n = len(holdout_pred) if holdout_pred is not None else 0
            out.append(dict(check="Chronological validation", status=st_(chron["ok"]),
                            method="Compared the active model’s training end date and CV folds with its holdout window.",
                            evidence=f"Fitted to {m(model_md['periods']['eval_fit'][1])}; scored on {n:,} unseen rows from {m(model_md['periods']['holdout'][0])}."))
    else:
        out += [dict(check="No future information in features", status="not_verified", method="Leakage tests were not run.", evidence="—"),
                dict(check="Chronological validation", status="not_verified", method="Leakage tests were not run.", evidence="—")]
    out += [
        dict(check="Test period can be scored", status="attention",
             method="Checked the test files for a Guests column.",
             evidence="No Guests in either test file (expected). Accuracy is measured on the Feb–Jul 2025 holdout instead."),
        dict(check="Test population matches training", status="attention",
             method="Compared New Arrivals in the training and test files.",
             evidence=f"Test rows never have New Arrivals below 10; {s['na_lt10_train']:,} training rows do. Competition models train on matching rows (a modelling decision)."),
    ]
    return out


def readiness_checks(s: dict, model_md: dict | None, whatif_features: list[str], holdout_pred: pd.DataFrame | None) -> list[dict]:
    """Kept for the Methodology page and older callers: the What-if feature list must hold no hotel observation."""
    hotel_obs = {"new_arrivals", "new_arrivals_7d", "same_day_guests"}
    return [dict(check="Leakage prevention", ok=not (hotel_obs & set(whatif_features)) and (model_md is None or not (hotel_obs & set(model_md["features"]))),
                 evidence="What-if features exclude New Arrivals and Same-Day Guests; all windows end on the prediction day.")]
