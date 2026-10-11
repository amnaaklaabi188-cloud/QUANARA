"""Independent verification of every figure shown on Page 1 (Welcome & Overview) and Page 2
(Data Intelligence).

The "independent" side re-reads the five raw workbooks with openpyxl and the Data Dictionary with
pdfplumber — it does NOT use quanara.data_loading / quanara.cleaning — and recomputes each
statistic with plain Python/NumPy. The "displayed" side calls the exact functions the two pages
render from. Model metrics are recomputed from the active model's saved holdout predictions, and
every saved y_true is matched back to the raw Guests value for that date and nationality.

Usage:  python scripts/verify_pages.py      → reports/verification_pages_1_2.{json,md}
"""
from __future__ import annotations

import collections
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import openpyxl  # noqa: E402
import pandas as pd  # noqa: E402
import pdfplumber  # noqa: E402

from quanara import config, registry  # noqa: E402
from quanara.data_loading import discover_files, load_all  # noqa: E402
from quanara.intel_facts import audit_findings, dataset_cards, dictionary_reconciliation, passenger_composition, quality_overview, quality_stats  # noqa: E402
from quanara.overview_facts import data_facts, model_facts  # noqa: E402

CHECKS: list[dict] = []


def check(page: str, name: str, displayed, independent, tol: float = 0.0) -> None:
    if isinstance(displayed, (int, float, np.floating)) and isinstance(independent, (int, float, np.floating)):
        ok = abs(float(displayed) - float(independent)) <= tol
    else:
        ok = str(displayed) == str(independent)
    CHECKS.append({"page": page, "check": name, "displayed": str(displayed), "independent": str(independent), "status": "PASS" if ok else "FAIL"})


def raw(path: Path) -> tuple[list[str], list[tuple]]:
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)["Export"]
    it = ws.iter_rows(values_only=True)
    hdr = list(next(it))
    return hdr, [r for r in it if any(v is not None for v in r)]


def day(x) -> str:
    return x.strftime("%Y-%m-%d")


def main() -> None:
    files = discover_files()
    H, R = {}, {}
    for k in ("intl_train", "intl_test", "dom_train", "dom_test", "flight"):
        H[k], R[k] = raw(files[k])
    ix = {k: {c: i for i, c in enumerate(H[k])} for k in H}
    data = load_all()
    f, m = data_facts(data), model_facts()
    cards = {c["key"]: c for c in dataset_cards(data)}
    audit = json.loads((config.REPORTS_DIR / "audit.json").read_text())
    pc, q = passenger_composition(data["flight"]), quality_overview(data, audit)

    # ---------------- flights
    fl, fi = R["flight"], ix["flight"]
    dates = [r[0] for r in fl]
    months = collections.defaultdict(set)
    for d in dates:
        months[(d.year, d.month)].add(d.day)
    monthly_m = {mm for mm, ds_ in months.items() if ds_ == {1}}
    is_m = [(d.year, d.month) in monthly_m for d in dates]
    daily = [r for r, mflag in zip(fl, is_m) if not mflag]
    check("1+2", "Flight rows", f["flight_rows"], len(fl))
    check("1+2", "Flight first date", day(f["flight_start"]), day(min(dates)))
    check("1+2", "Flight last date", day(f["flight_end"]), day(max(dates)))
    check("1+2", "Flight monthly-grain rows (2022)", f["flight_rows_monthly"], sum(is_m))
    check("1+2", "Flight daily-grain rows", f["flight_rows_daily"], len(daily))
    check("1", "First daily-grain date", day(f["flight_daily_start"]), day(min(r[0] for r in daily)))
    check("1+2", "Departure countries (all rows)", f["flight_countries"], len({r[fi["Departure Country Name"]] for r in fl}))
    check("1+2", "Airlines (all rows)", f["flight_airlines"], len({r[fi["Airline Name"]] for r in fl}))
    check("1+2", "Σ monthly PAX in sparkline = Σ Total PAX", sum(f["flight_monthly_pax"]), sum(r[fi["Total PAX"]] for r in fl), 0.5)
    s = lambda rows, c: float(sum((r[fi[c]] or 0) for r in rows))
    p2p, tr, trs, pax = s(daily, "Total P2P"), s(daily, "Total Transfer"), s(daily, "Total Transit"), s(daily, "Total PAX")
    check("2", "Daily-row P2P passengers", pc["daily_totals"]["Total P2P"], p2p)
    check("2", "Daily-row Transfer passengers", pc["daily_totals"]["Total Transfer"], tr)
    check("2", "Daily-row Transit passengers", pc["daily_totals"]["Total Transit"], trs)
    check("2", "P2P share of daily PAX (shown 1 dp)", round(pc["daily_p2p_share"] * 100, 1), round(p2p / pax * 100, 1))
    check("2", "Transfer share of daily PAX (shown 1 dp)", round(pc["daily_transfer_share"] * 100, 1), round(tr / pax * 100, 1))
    check("2", "Daily rows with Transit > 0", pc["transit_rows_daily"], sum(1 for r in daily if (r[fi["Total Transit"]] or 0) > 0))
    check("2", "PAX = P2P + Transfer + Transit on every row", pc["identity_holds"],
          all(abs(r[fi["Total PAX"]] - r[fi["Total P2P"]] - r[fi["Total Transfer"]] - r[fi["Total Transit"]]) < 0.5 for r in fl))
    check("2", "Daily rows with load factor > 100% (shown 1 dp)", round(q["lf_over_100_daily_pct"], 1),
          round(sum(1 for r in daily if r[fi["Load Factor"]] > 1) / len(daily) * 100, 1))
    check("2", "Flight card: monthly rows in text", f"{sum(is_m):,} monthly rows" in cards["flight"]["secondary"], True)

    # ---------------- hotels
    it, ii = R["intl_train"], ix["intl_train"]
    dt, di = R["dom_train"], ix["dom_train"]
    check("1+2", "International train rows", f["intl_rows"], len(it))
    check("1+2", "International nationalities", f["intl_nationalities"], len({r[ii["Nationality"]] for r in it}))
    check("1", "Days with observed Guests", f["hotel_days"], len({r[0] for r in it}))
    check("1+2", "Domestic train days", f["dom_days"], len(dt))
    check("1", "Labelled records (intl + domestic)", f["labelled_records"], len(it) + len(dt))
    check("1", "Guests observed from", day(f["hotel_start"]), day(min(r[0] for r in it)))
    check("1", "Guests observed to", day(f["hotel_end"]), day(max(r[0] for r in it)))
    check("1", "Unlabelled test rows (intl + domestic)", f["test_intl_rows"] + f["test_dom_rows"], len(R["intl_test"]) + len(R["dom_test"]))
    check("1", "Test period start", day(f["test_start"]), day(min(r[0] for r in R["intl_test"] + R["dom_test"])))
    check("1", "Test period end", day(f["test_end"]), day(max(r[0] for r in R["intl_test"] + R["dom_test"])))
    check("1+2", "Test files contain a Guests column", f["test_files_have_guests"], "Guests" in H["intl_test"] or "Guests" in H["dom_test"])
    by_day = collections.defaultdict(float)
    for r in it:
        by_day[r[0]] += r[ii["Guests"]]
    check("1", "Average international Guests per day (shown 0 dp)", round(f["intl_avg_daily_total"]), round(np.mean(list(by_day.values()))))
    check("1", "Average domestic Guests per day (shown 0 dp)", round(f["dom_avg_daily"]), round(np.mean([r[di["Guests"]] for r in dt])))
    for k in ("intl_train", "intl_test", "dom_train", "dom_test"):
        c, rows = cards[k], R[k]
        check("2", f"{k}: rows", c["value"], f"{len(rows):,}")
        check("2", f"{k}: columns in text", f"{len(H[k])} columns" in c["secondary"], True)
        check("2", f"{k}: distinct days in text", f"{len({r[0] for r in rows}):,} days" in c["secondary"], True)
    keyidx = {"intl_train": (0, ii["Nationality"]), "intl_test": (0, ix["intl_test"]["Nationality"]), "dom_train": (0,), "dom_test": (0,)}
    dups = sum(len(R[k]) - len({tuple(r[j] for j in keyidx[k]) for r in R[k]}) for k in keyidx)
    dups += len(fl) - len({(r[0], r[fi["Departure Country Name"]], r[fi["Departure City"]], r[fi["Airline Name"]]) for r in fl})
    check("2", "Duplicate keys across the five files", q["duplicate_keys"], dups)
    stars = sum(1 for k in R for r in R[k] for v in r if isinstance(v, str) and v.strip() == "*")
    check("2", "Literal '*' cells across the five files", q["literal_asterisks"], stars)
    n_days, nats = len({r[0] for r in it}), {r[ii["Nationality"]] for r in it}
    check("2", "Missing nationality-day combinations", q["missing_nat_day_combinations"], n_days * len(nats) - len(it))
    per_nat = collections.Counter(r[ii["Nationality"]] for r in it)
    check("2", "Nationalities with missing days", q["nationalities_with_missing_days"], sum(1 for v in per_nat.values() if v < n_days))
    check("2", "Same-Day Guests suppressed, intl train (shown 1 dp)", round(q["same_day_suppressed_train_pct"], 1),
          round(sum(1 for r in it if r[ii["Same-Day Guests"]] is None) / len(it) * 100, 1))
    check("2", "Same-Day Guests suppressed, intl test (shown 1 dp)", round(q["same_day_suppressed_test_pct"], 1),
          round(sum(1 for r in R["intl_test"] if r[ix["intl_test"]["Same-Day Guests"]] is None) / len(R["intl_test"]) * 100, 1))
    check("2", "Train rows with New Arrivals < 10", q["na_lt10_train"], sum(1 for r in it if r[ii["New Arrivals"]] is not None and r[ii["New Arrivals"]] < 10))
    check("2", "Test rows with New Arrivals < 10", q["na_lt10_test"],
          sum(1 for r in R["intl_test"] if r[ix["intl_test"]["New Arrivals"]] is not None and r[ix["intl_test"]["New Arrivals"]] < 10))
    countries = {r[fi["Departure Country Name"]].strip().upper() for r in fl}
    check("1+2", "Nationalities matching a departure country", f["matched_markets"], len(nats & countries))
    g_m = sum(r[ii["Guests"]] for r in it if r[ii["Nationality"]] in countries)
    check("2", "Guest-days in matched nationalities (shown 1 dp)", round(audit["market_matching"]["guest_share_in_matched_nationalities_pct"], 1),
          round(g_m / sum(by_day.values()) * 100, 1))

    # ---------------- Data Quality overview cards and every audit-finding evidence value (Page 2)
    qs = quality_stats(data, stars)
    check("2", "Rows checked for duplicate keys (all five files)", qs["rows_total"], sum(len(R[k]) for k in R))
    check("2", "Duplicate keys (quality card)", qs["duplicate_keys"], dups)
    check("2", "Training rows with Guests: international", qs["intl_train_rows"], sum(1 for r in it if r[ii["Guests"]] is not None))
    check("2", "Training rows with Guests: domestic", qs["dom_train_rows"], sum(1 for r in dt if r[di["Guests"]] is not None))
    d_raw = [r for r, mflag in zip(fl, is_m) if not mflag]
    m_raw = [r for r, mflag in zip(fl, is_m) if mflag]
    cab_tr = lambda r: sum((r[fi[c]] or 0) for c in ("Transit Business Class Count", "Transit Economy Class Count", "Transit First Count"))
    cab_p2p = lambda r: sum((r[fi[c]] or 0) for c in ("Business Class P2P Count", "Economy Class P2P Count", "First Class P2P Count"))
    cab_trf = lambda r: sum((r[fi[c]] or 0) for c in ("Transfer Business Class Count", "Transfer Economy Class Count", "Transfer First Class Count"))
    uk = [r for r in d_raw if r[fi["Departure Country Name"]] == "United Kingdom"]
    ru_pax = sum(r[fi["Total PAX"]] for r in d_raw if r[fi["Departure Country Name"]] == "Russian Federation")
    tot_g = sum(r[ii["Guests"]] for r in it)
    wn_raw = {k: n_days - v for k, v in per_nat.items() if v < n_days}
    independent = {
        "matched_markets": len(nats & countries),
        "matched_guest_share": round(g_m / tot_g, 4),
        "uk_transfer_share": round(sum(r[fi["Total Transfer"]] for r in uk) / sum(r[fi["Total PAX"]] for r in uk), 4),
        "ru_shares": [round(sum(r[ii["Guests"]] for r in it if r[ii["Nationality"]] == "RUSSIAN FEDERATION") / tot_g, 4),
                      round(ru_pax / sum(r[fi["Total PAX"]] for r in d_raw), 4)],
        "monthly_rows": len(m_raw), "daily_rows": len(d_raw),
        "awf_empty": [sum(1 for r in m_raw if r[fi["Average Weekly Frequency"]] is None), sum(1 for r in d_raw if r[fi["Average Weekly Frequency"]] is None)],
        "na_lt10_train": sum(1 for r in it if r[ii["New Arrivals"]] is not None and r[ii["New Arrivals"]] < 10),
        "na_lt10_test": sum(1 for r in R["intl_test"] if r[ix["intl_test"]["New Arrivals"]] is not None and r[ix["intl_test"]["New Arrivals"]] < 10),
        "rows_per_day": [round(len(it) / n_days, 1), round(len(R["intl_test"]) / len({r[0] for r in R["intl_test"]}), 1)],
        "missing_nat_days": n_days * len(nats) - len(it),
        "missing_by_nationality": dict(sorted(wn_raw.items(), key=lambda kv: -kv[1])),
        "min_guests_intl": min(r[ii["Guests"]] for r in it),
        "same_day_empty_train": sum(1 for r in it if r[ii["Same-Day Guests"]] is None),
        "same_day_empty_test": sum(1 for r in R["intl_test"] if r[ix["intl_test"]["Same-Day Guests"]] is None),
        "literal_asterisks": stars,
        "lf_gt1_daily": sum(1 for r in d_raw if r[fi["Load Factor"]] > 1),
        "lf_gt1_monthly": sum(1 for r in m_raw if r[fi["Load Factor"]] > 1),
        "lf_identity": round(max(abs(r[fi["Load Factor"]] - r[fi["Total PAX"]] / r[fi["Total Seats"]]) for r in fl), 12),
        "test_rows": len(R["intl_test"]) + len(R["dom_test"]),
        "test_have_guests": "Guests" in H["intl_test"] or "Guests" in H["dom_test"],
        "transit_rows_daily": sum(1 for r in d_raw if (r[fi["Total Transit"]] or 0) > 0),
        "transit_pax_daily": float(sum((r[fi["Total Transit"]] or 0) for r in d_raw)),
        "cabin_transit_rows": sum(1 for r in d_raw if cab_tr(r) > 0),
        "cabin_p2p_match_rows": sum(1 for r in d_raw if abs(r[fi["Total P2P"]] - cab_p2p(r)) < 0.5),
        # evidence added in the Section 2 scientific review
        "na_lt10_guest_share": round(sum(r[ii["Guests"]] for r in it if r[ii["New Arrivals"]] is not None and r[ii["New Arrivals"]] < 10) / tot_g, 4),
        "lf_gt1_excl_infant": sum(1 for r in d_raw if r[fi["Load Factor"]] > 1 and r[fi["Total PAX Excluding Infant"]] is not None
                                  and r[fi["Total PAX Excluding Infant"]] / r[fi["Total Seats"]] > 1),
        "cabin_trts_le_transfer": round(sum(1 for r in d_raw if cab_trf(r) + cab_tr(r) <= r[fi["Total Transfer"]] + 0.5) / len(d_raw), 4),
        "cabin_p2p_plus_inf": round(sum(1 for r in d_raw if r[fi["Total PAX Excluding Infant"]] is not None
                                        and abs(cab_p2p(r) + (r[fi["Total PAX"]] - r[fi["Total PAX Excluding Infant"]]) - r[fi["Total P2P"]]) < 0.5) / len(d_raw), 4),
    }
    rnd = lambda v: round(v, 4) if isinstance(v, float) else ([round(x, 4) if isinstance(x, float) else x for x in v] if isinstance(v, list) else v)
    for f_ in audit_findings(qs):
        for e in f_["evidence"]:
            shown = round(e["value"], 12) if e["id"] == "lf_identity" else (dict(e["value"]) if isinstance(e["value"], dict) else rnd(e["value"]))
            check("2", f"Finding evidence — {e['label']}", shown, independent[e["id"]])
    check("2", "Smallest recorded international Guests value", qs["min_guests_intl"], min(r[ii["Guests"]] for r in it))

    # ---------------- "no value" kinds, readiness evidence, dictionary vs delivered data (Page 2)
    check("2", "Empty New Arrivals cells, international training", qs["new_arrivals_empty_train"], sum(1 for r in it if r[ii["New Arrivals"]] is None))
    check("2", "Daily rows with Total Transit = 0 (genuine zeros)", qs["transit_zero_rows_daily"], sum(1 for r in d_raw if r[fi["Total Transit"]] == 0))
    check("2", "Empty Guests in training files", qs["guests_empty_train"], sum(1 for r in it if r[ii["Guests"]] is None) + sum(1 for r in dt if r[di["Guests"]] is None))
    check("2", "Empty passenger/seat totals on daily rows", qs["aviation_empty_daily"],
          sum(1 for r in d_raw for c in ("Total Seats", "Total P2P", "Total PAX", "Total Transfer") if r[fi[c]] is None))
    check("2", "International test columns (no Guests)", ", ".join(qs["test_columns"]), ", ".join(H["intl_test"]))
    check("2", "Domestic test columns (no Guests)", ", ".join(qs["dom_test_columns"]), ", ".join(H["dom_test"]))
    check("2", "Domestic training Guests minimum", qs["dom_guests_min"], min(r[di["Guests"]] for r in dt))
    check("2", "Domestic training Guests maximum", qs["dom_guests_max"], max(r[di["Guests"]] for r in dt))
    check("2", "Distinct column names across the five workbooks", qs["distinct_columns"], len({c for k in H for c in H[k]}))
    check("2", "Largest load factor (2022 monthly rows)", round(qs["lf_max_monthly"], 6), round(max(r[fi["Load Factor"]] for r in m_raw), 6))
    check("2", "International test days", qs["test_days"], len({r[0] for r in R["intl_test"]}))
    check("2", "Dates stored as Excel date values (not strings)", True, all(isinstance(r[0], datetime) for k in R for r in R[k]))
    with pdfplumber.open(files["dictionary"]) as pdf:
        pdf_text = " ".join(" ".join((pg.extract_text() or "") for pg in pdf.pages).split())
    for row in dictionary_reconciliation(qs):
        check("2", f"Dictionary wording present in PDF — {row['topic']}", True, row["phrase"] in pdf_text)

    # ---------------- correlation shown in "About this project"
    p2p_day = collections.defaultdict(float)
    for r in daily:
        p2p_day[r[0]] += r[fi["Total P2P"]]
    d0, d1 = min(p2p_day), max(p2p_day)
    cal = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
    seq = [p2p_day.get(x, 0.0) for x in cal]
    roll = {cal[i]: sum(seq[i - 6:i + 1]) for i in range(6, len(cal))}
    pairs = [(by_day[x], roll[x]) for x in roll if config.MODEL_START <= pd.Timestamp(x) <= config.MODEL_END and x in by_day]
    r_ind = float(np.corrcoef(np.array(pairs).T)[0, 1])
    check("1", "Correlation: intl Guests vs trailing-7-day P2P (shown 2 dp)", round(f["p2p7_guests_r"], 2), round(r_ind, 2))

    # ---------------- Guests definition (Data Dictionary)
    with pdfplumber.open(files["dictionary"]) as pdf:
        text = " ".join((p.extract_text() or "") for p in pdf.pages)
    text = " ".join(text.split())
    check("1", "Dictionary: Guests = guests recorded for nationality on date (intl)", "guests recorded for the given nationality" in text, True)
    check("1", "Dictionary: New Arrivals are a subset of Guests (continuing stays excluded)", "Always a subset of the total Guests count" in text and "continuing stays" in text, True)
    check("1", "Dictionary: domestic Guests is a daily total", "Total number of domestic" in text and "Each date appears exactly" in text, True)

    # ---------------- active model (metrics from saved predictions; y_true against raw Guests)
    mid = registry.active_model_id()
    check("1", "Active model ID resolved", m["model_id"] if m else None, mid)
    md = registry.load_metadata(mid)
    pred = registry.load_table(mid, "predictions_holdout")
    y, p = pred["y_true"].to_numpy(float), pred["y_pred"].to_numpy(float)
    e = y - p
    mae, rmse = float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2)))
    r2 = float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))
    wmape = float(np.sum(np.abs(e)) / np.sum(np.abs(y)))
    check("1", "Model metadata ID = folder ID", md["model_id"], mid)
    check("1", "Holdout R² (shown 3 dp)", round(m["r2"], 3), round(r2, 3))
    check("1", "Holdout MAE (shown 0 dp)", round(m["mae"]), round(mae))
    check("1", "Holdout WMAPE (shown 1 dp)", round(m["wmape"] * 100, 1), round(wmape * 100, 1))
    check("1", "Holdout RMSE (metadata vs recomputed)", round(md["metrics_holdout"]["RMSE"], 6), round(rmse, 6))
    check("1", "Holdout predictions count", m["n_holdout"], len(pred))
    check("1", "Holdout start", m["holdout"][0], day(pred["Date"].min()))
    check("1", "Holdout end", m["holdout"][1], day(pred["Date"].max()))
    check("1", "Training (evaluation fit) ends the day before the holdout", m["eval_fit"][1],
          day(pred["Date"].min() - pd.Timedelta(days=1)))
    check("1", "Training starts at the modelling start", m["eval_fit"][0], day(config.MODEL_START))
    if md["scope"] == "international_market":
        raw_g = {(day(r[0]), r[ii["Nationality"]]): r[ii["Guests"]] for r in it}
        match = [raw_g.get((day(d), k)) == yt for d, k, yt in zip(pred["Date"], pred["market_key"], pred["y_true"])]
        check("1", "Every saved y_true equals the raw Guests value", f"{sum(match)}/{len(match)}", f"{len(match)}/{len(match)}")

    passed = sum(c["status"] == "PASS" for c in CHECKS)
    out = {"generated_at": datetime.now().isoformat(timespec="seconds"), "active_model": mid, "passed": passed, "total": len(CHECKS), "checks": CHECKS}
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "verification_pages_1_2.json").write_text(json.dumps(out, indent=2))
    md_lines = [f"# Verification — Pages 1 and 2", f"_{out['generated_at']} · active model `{mid}` · **{passed}/{len(CHECKS)} checks passed**_", "",
                "| page | check | displayed | independent | status |", "|---|---|---|---|---|"]
    md_lines += [f"| {c['page']} | {c['check']} | {c['displayed']} | {c['independent']} | {c['status']} |" for c in CHECKS]
    (config.REPORTS_DIR / "verification_pages_1_2.md").write_text("\n".join(md_lines) + "\n")
    for c in CHECKS:
        if c["status"] != "PASS":
            print("FAIL:", c)
    print(f"{passed}/{len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()
