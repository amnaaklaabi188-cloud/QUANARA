"""Reproducible analytical datasets at DAILY grain.

Grain decision (see reports/audit_report.md): hotel files are daily; flights are daily from
2023-01-01. 2022 flight rows are monthly totals and cannot be assigned to days without an
invented disaggregation rule, so they are excluded from daily modelling (they remain
available for monthly EDA). Training rows are restricted to dates with observed Guests.

Every step appends to a build log (rows before/after, coverage) and every merge is
validated for cardinality so no row multiplication can go unnoticed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .features import (aggregate_aviation, available_features, aviation_features, build_feature_frame,
                       daily_flight_rows, guest_history_features)


@dataclass
class BuildLog:
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, before: int | None, after: int | None, note: str = "") -> None:
        self.steps.append({"step": step, "rows_before": before, "rows_after": after, "note": note})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


@dataclass
class Analytic:
    """Everything downstream code needs; built once and cached to data/processed."""
    flight_daily_rows: pd.DataFrame        # route-day rows, daily grain only
    markets: list[str]                     # 33 matched departure-country / nationality names
    unmatched_nationalities: list[str]
    hotel_total: pd.DataFrame              # one row per date, train + test
    hotel_market: pd.DataFrame             # one row per (date, nationality), train + test
    av_total: pd.DataFrame                 # aviation features, all markets, per date
    av_market: pd.DataFrame                # aviation features per (date, market)
    log: BuildLog


# ------------------------------------------------------------------ hotel aggregates

def build_hotel_total(d: dict[str, pd.DataFrame], log: BuildLog) -> pd.DataFrame:
    it = pd.concat([d["intl_train"].assign(split="train"), d["intl_test"].assign(split="test")], ignore_index=True)
    dm = pd.concat([d["dom_train"].assign(split="train"), d["dom_test"].assign(split="test")], ignore_index=True)
    log.add("concat intl train+test", len(d["intl_train"]) + len(d["intl_test"]), len(it))
    log.add("concat domestic train+test", len(d["dom_train"]) + len(d["dom_test"]), len(dm))
    it["na_ge10"] = it["New Arrivals"].where(it["New Arrivals"] >= 10)
    intl = it.groupby("Date").agg(
        split=("split", "first"),
        intl_guests=("Guests", lambda s: s.sum(min_count=1)),
        intl_reported_nationalities=("Nationality", "nunique"),
        intl_new_arrivals_all=("New Arrivals", "sum"),
        intl_new_arrivals=("na_ge10", "sum"),  # harmonised with the test extract (NA >= 10 only)
        intl_same_day=("Same-Day Guests", "sum"),  # '*' -> 0: lower bound
    ).reset_index()
    log.add("aggregate intl to daily total (sum over nationalities)", len(it), len(intl),
            "Guests summed over the nationality rows that exist; missing nationality-day rows contribute nothing.")
    dom = dm.rename(columns={"Guests": "dom_guests", "New Arrivals": "dom_new_arrivals", "Same-Day Guests": "dom_same_day"})[
        ["Date", "split", "dom_guests", "dom_new_arrivals", "dom_same_day"]]
    n0 = len(intl)
    tot = intl.merge(dom, on=["Date", "split"], how="outer", validate="1:1", indicator=True)
    unmatched = int((tot["_merge"] != "both").sum())
    tot = tot.drop(columns="_merge")
    log.add("join intl daily ⟷ domestic daily on Date (1:1)", n0, len(tot), f"{unmatched} dates present in only one file")
    tot["overall_guests"] = tot["intl_guests"] + tot["dom_guests"]
    tot["overall_new_arrivals"] = tot["intl_new_arrivals"] + tot["dom_new_arrivals"]
    tot["overall_same_day"] = tot["intl_same_day"] + tot["dom_same_day"]
    for p in ("intl", "dom", "overall"):
        tot[f"{p}_new_arrivals_7d"] = tot[f"{p}_new_arrivals"].rolling(7, min_periods=7).sum()
    return tot.sort_values("Date").reset_index(drop=True)


def build_hotel_market(d: dict[str, pd.DataFrame], markets: list[str], log: BuildLog) -> pd.DataFrame:
    it = pd.concat([d["intl_train"].assign(split="train"), d["intl_test"].assign(split="test")], ignore_index=True)
    it = it.rename(columns={"Nationality": "market_key"})
    it["matched"] = it["market_key"].isin(markets)
    # trailing 7-day New Arrivals on a complete calendar per nationality (missing rows -> 0, lower bound)
    dates = pd.date_range(it["Date"].min(), it["Date"].max(), name="Date")
    full = it.set_index(["Date", "market_key"])["New Arrivals"].reindex(
        pd.MultiIndex.from_product([dates, sorted(it["market_key"].unique())], names=["Date", "market_key"])).fillna(0).reset_index()
    full = full.sort_values(["market_key", "Date"])
    full["new_arrivals_7d"] = full.groupby("market_key")["New Arrivals"].transform(lambda s: s.rolling(7, min_periods=7).sum())
    n0 = len(it)
    it = it.merge(full[["Date", "market_key", "new_arrivals_7d"]], on=["Date", "market_key"], how="left", validate="1:1")
    assert len(it) == n0
    log.add("intl rows by (Date, nationality), train+test", None, len(it),
            f"{int(it['matched'].sum()):,} rows in {len(markets)} matched markets; {int((~it['matched']).sum()):,} rows in unmatched nationalities")
    return it.sort_values(["market_key", "Date"]).reset_index(drop=True)


# ------------------------------------------------------------------ build

def build_analytic(data: dict[str, pd.DataFrame]) -> Analytic:
    log = BuildLog()
    fl = data["flight"]
    rows = daily_flight_rows(fl)
    log.add("flight rows → daily grain only", len(fl), len(rows), f"dropped {len(fl) - len(rows):,} monthly 2022 rows (cannot be placed on a day)")
    countries = sorted(rows["market_key"].unique())
    nats = sorted(data["intl_train"]["Nationality"].unique())
    markets = sorted(set(countries) & set(nats))
    unmatched = sorted(set(nats) - set(countries))
    log.add("match Nationality ⟷ Departure Country Name (exact upper-case name)", None, len(markets),
            f"{len(markets)}/{len(nats)} nationalities matched; unmatched: {', '.join(unmatched)}; departure countries without nationality: {', '.join(sorted(set(countries) - set(nats))) or 'none'}")

    agg_t = aggregate_aviation(rows, by_market=False)
    log.add("aggregate route-days → daily total (sum seats/pax/flights; recompute ratios)", len(rows), len(agg_t),
            f"{int((agg_t['seats'] == 0).sum())} calendar days without any flight row")
    agg_m = aggregate_aviation(rows, by_market=True, markets=markets)
    log.add("aggregate route-days → daily × market (complete calendar)", len(rows), len(agg_m),
            f"{int((agg_m['seats'] == 0).sum()):,} market-days with zero seats (no operated flight)")
    av_t = aviation_features(agg_t, by_market=False)
    av_m = aviation_features(agg_m, by_market=True)

    hotel_total = build_hotel_total(data, log)
    hotel_market = build_hotel_market(data, markets, log)

    # coverage: flight vs hotel calendars
    hd = set(hotel_total["Date"])
    fd = set(agg_t.loc[agg_t["seats"] > 0, "Date"])
    log.add("temporal alignment: hotel dates with daily flight data", len(hd), len(hd & fd),
            f"hotel dates without daily flights: {len(hd - fd)} (all in 2022, monthly flights only)")
    a = Analytic(rows, markets, unmatched, hotel_total, hotel_market, av_t, av_m, log)
    return a


# ------------------------------------------------------------------ model frames

TARGET_COL = {"international_total": "intl_guests", "domestic": "dom_guests", "overall": "overall_guests",
              "international_market": "Guests"}
_COMP_PREFIX = {"international_total": "intl", "domestic": "dom", "overall": "overall"}


def history_frame(a: Analytic, scope: str) -> pd.DataFrame:
    if scope == "international_market":
        t = a.hotel_market[(a.hotel_market["split"] == "train")][["Date", "market_key", "Guests"]]
        return guest_history_features(t, by_market=True)
    t = a.hotel_total[a.hotel_total["split"] == "train"][["Date", TARGET_COL[scope]]].rename(columns={TARGET_COL[scope]: "Guests"})
    return guest_history_features(t, by_market=False)


def competition_extra(a: Analytic, scope: str) -> pd.DataFrame:
    if scope == "international_market":
        e = a.hotel_market[["Date", "market_key", "New Arrivals", "new_arrivals_7d", "Same-Day Guests"]].rename(
            columns={"New Arrivals": "new_arrivals", "Same-Day Guests": "same_day_guests"})
        e["same_day_guests"] = e["same_day_guests"].fillna(0)
        return e
    p = _COMP_PREFIX[scope]
    return a.hotel_total[["Date", f"{p}_new_arrivals", f"{p}_new_arrivals_7d", f"{p}_same_day"]].rename(
        columns={f"{p}_new_arrivals": "new_arrivals", f"{p}_new_arrivals_7d": "new_arrivals_7d", f"{p}_same_day": "same_day_guests"})


def model_frame(a: Analytic, scope: str, mode: str) -> pd.DataFrame:
    """One row per prediction unit with every available feature, target `y` and `split`.

    What-if, market scope: the 33 matched markets only. Competition, market scope: all 45
    nationalities (aviation features NaN — not zero — where no departure country matches);
    train rows restricted to New Arrivals >= 10 to mirror the test extract.
    """
    hist = history_frame(a, scope)
    extra = competition_extra(a, scope) if mode == "competition" else None
    if scope == "international_market":
        hm = a.hotel_market
        if mode == "whatif":
            hm = hm[hm["matched"]]
        else:
            hm = hm[(hm["split"] == "test") | (hm["New Arrivals"] >= 10)]
        base = hm[["Date", "market_key", "split", "Guests", "matched"]].rename(columns={"Guests": "y"})
        base = base[base["Date"] >= config.FLIGHT_DAILY_START]
        n0 = len(base)
        df = base.merge(a.av_market, on=["Date", "market_key"], how="left", validate="1:1")
        df = df.merge(hist, on=["Date", "market_key"], how="left", validate="1:1")
        if extra is not None:
            df = df.merge(extra, on=["Date", "market_key"], how="left", validate="1:1")
        assert len(df) == n0, "row multiplication in market frame"
        from .features import time_features
        df = pd.concat([df.reset_index(drop=True), time_features(df["Date"]).reset_index(drop=True)], axis=1)
        df["market"] = df["market_key"]
    else:
        ht = a.hotel_total[["Date", "split", TARGET_COL[scope]]].rename(columns={TARGET_COL[scope]: "y"})
        ht = ht[ht["Date"] >= config.FLIGHT_DAILY_START]
        df = build_feature_frame(a.av_total[a.av_total["Date"].isin(ht["Date"])], hist, by_market=False, extra=extra)
        n0 = len(df)
        df = df.merge(ht, on="Date", how="left", validate="1:1")
        assert len(df) == n0
    keep = ["Date"] + (["market_key", "matched"] if scope == "international_market" else []) + ["split", "y"] + available_features(scope, mode)
    return df[[c for c in keep if c in df.columns]].sort_values(keep[:2] if scope != "international_market" else ["Date", "market_key"]).reset_index(drop=True)


def training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    m = (frame["split"] == "train") & frame["Date"].between(config.MODEL_START, config.MODEL_END) & frame["y"].notna()
    return frame[m].reset_index(drop=True)


# ------------------------------------------------------------------ persistence

def save_analytic(a: Analytic) -> None:
    config.ensure_dirs()
    p = config.PROCESSED_DIR
    a.hotel_total.to_parquet(p / "hotel_daily_total.parquet", index=False)
    a.hotel_market.to_parquet(p / "hotel_daily_market.parquet", index=False)
    a.av_total.to_parquet(p / "aviation_daily_total.parquet", index=False)
    a.av_market.to_parquet(p / "aviation_daily_market.parquet", index=False)
    cols = ["Date", "Departure Country Name", "Departure City", "Airline Name", "market_key", "Total Seats", "Total P2P",
            "Total PAX", "Total Transfer", "Total Transit", "flights", "Load Factor", "Business Class Seat Capacity", "First Class Seat Capacity"]
    a.flight_daily_rows[cols].to_parquet(p / "flight_route_days.parquet", index=False)
    meta = {"markets": a.markets, "unmatched_nationalities": a.unmatched_nationalities, "log": a.log.steps,
            "model_start": str(config.MODEL_START.date()), "model_end": str(config.MODEL_END.date())}
    (p / "build_meta.json").write_text(json.dumps(meta, indent=2, default=str))


def load_analytic() -> Analytic:
    p = config.PROCESSED_DIR
    meta = json.loads((p / "build_meta.json").read_text())
    log = BuildLog(meta["log"])
    rows = pd.read_parquet(p / "flight_route_days.parquet")
    return Analytic(rows, meta["markets"], meta["unmatched_nationalities"],
                    pd.read_parquet(p / "hotel_daily_total.parquet"), pd.read_parquet(p / "hotel_daily_market.parquet"),
                    pd.read_parquet(p / "aviation_daily_total.parquet"), pd.read_parquet(p / "aviation_daily_market.parquet"), log)


def get_analytic(rebuild: bool = False) -> Analytic:
    if not rebuild and (config.PROCESSED_DIR / "build_meta.json").exists():
        return load_analytic()
    from .data_loading import load_all
    a = build_analytic(load_all())
    save_analytic(a)
    return a
