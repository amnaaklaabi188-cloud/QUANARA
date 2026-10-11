"""Page 2 (Data Intelligence) facts, model-card consistency, and the independent verification."""
import json

import numpy as np
import pytest

from quanara import config, registry
from quanara.data_loading import read_raw
from quanara.intel_facts import (audit_findings, checksum_status, dataset_cards, dictionary_reconciliation, missing_profile, passenger_composition,
                                  quality_overview, quality_stats, technical_checks, value_kinds)
from quanara.overview_facts import model_facts


def test_dataset_cards_match_the_files(data):
    cards = {c["key"]: c for c in dataset_cards(data)}
    assert set(cards) == {"intl_train", "intl_test", "dom_train", "dom_test", "flight"}
    for k, c in cards.items():
        assert c["value"] == f"{len(data[k]):,}"
        assert c["caption"] and len(c["series"]) > 1
    assert f"{data['flight']['Airline Name'].nunique()} airlines" in cards["flight"]["secondary"]
    assert cards["intl_test"]["badge"][0] == "No Guests" and cards["intl_train"]["badge"][0] == "Guests observed"


def test_passenger_composition_is_consistent(data):
    pc = passenger_composition(data["flight"])
    t = pc["daily_totals"]
    assert pc["identity_holds"]
    assert np.isclose(t["Total P2P"] + t["Total Transfer"] + t["Total Transit"], t["Total PAX"])
    assert np.isclose(pc["daily_p2p_share"] + pc["daily_transfer_share"] + t["Total Transit"] / t["Total PAX"], 1)
    assert pc["transit_rows_daily"] == 3
    assert np.isclose(pc["monthly"]["Total PAX"].sum(), data["flight"]["Total PAX"].sum())


def test_quality_overview(data):
    a = json.loads((config.REPORTS_DIR / "audit.json").read_text())
    q = quality_overview(data, a)
    assert q["duplicate_keys"] == 0 and q["literal_asterisks"] == 0
    it = data["intl_train"]
    assert q["missing_nat_day_combinations"] == it["Date"].nunique() * it["Nationality"].nunique() - len(it)
    assert q["na_lt10_test"] == 0 < q["na_lt10_train"]


def test_originals_checksums_match():
    ok, n = checksum_status()
    assert ok and n == 6


@pytest.mark.skipif(not registry.list_models(), reason="reference models not trained")
def test_active_model_card_metrics_come_from_one_run():
    m = model_facts()
    assert m["consistent"], "metrics recomputed from saved predictions differ from the model's metadata"
    md = registry.load_metadata(m["model_id"])
    assert md["model_id"] == m["model_id"] and md["periods"]["holdout"] == m["holdout"]


@pytest.mark.skipif(not registry.list_models(), reason="reference models not trained")
def test_independent_verification_passes():
    from scripts import verify_pages
    verify_pages.CHECKS.clear()
    verify_pages.main()
    v = json.loads((config.REPORTS_DIR / "verification_pages_1_2.json").read_text())
    failed = [c for c in v["checks"] if c["status"] != "PASS"]
    assert not failed, failed
    assert v["total"] >= 60


def _stats(data):
    stars = sum(int((r[c].astype(str).str.strip() == "*").sum()) for k in data for r in [read_raw(k)] for c in r.columns)
    return quality_stats(data, stars)


def test_quality_stats_denominators(data):
    s = _stats(data)
    assert s["rows_total"] == sum(len(df) for df in data.values())
    assert s["possible_nat_days"] == data["intl_train"]["Date"].nunique() * data["intl_train"]["Nationality"].nunique()
    assert sum(s["missing_by_nationality"].values()) == s["missing_nat_days"] == s["possible_nat_days"] - len(data["intl_train"])
    assert s["same_day_empty_train"] == int(data["intl_train"]["Same-Day Guests"].isna().sum())
    assert s["monthly_rows"] + s["daily_rows"] == len(data["flight"])


def test_audit_findings_are_complete_and_ranked_by_impact(data):
    f = audit_findings(_stats(data))
    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    assert [order[x["severity"]] for x in f] == sorted(order[x["severity"]] for x in f)
    assert {x["severity"] for x in f} <= set(order)  # no "critical" label anywhere
    for x in f:
        for part in ("title", "summary", "found", "why", "handling", "impact"):
            assert x[part].strip(), (x["title"], part)
        assert x["evidence"] and all(e["display"] for e in x["evidence"])
    expected = next(x for x in f if "no Guests" in x["title"])
    assert expected["severity"] == "info"  # unlabelled test files are expected, not an error


def test_missing_profile_uses_row_denominator(data):
    prof = missing_profile(data["intl_train"]).set_index("column")
    assert prof.loc["Same-Day Guests", "rows"] == len(data["intl_train"])
    assert abs(prof.loc["Same-Day Guests", "pct"] - data["intl_train"]["Same-Day Guests"].isna().mean() * 100) < 1e-9


def test_missing_combinations_wording_makes_no_unconfirmed_claim(data):
    f = next(x for x in audit_findings(_stats(data)) if "nationality-day" in x["title"])
    text = " ".join([f["title"], f["summary"], f["found"], f["why"], f["handling"], f["impact"]]).lower()
    assert "unobserved" in f["title"].lower() and "withheld" not in text
    assert "possible explanation" in text and "not confirmed" in text
    assert not any("≤ 9" in e["label"] or "upper bound" in e["label"].lower() for e in f["evidence"])


def test_value_kinds_are_distinct_and_counted(data):
    kinds = value_kinds(_stats(data))
    assert [k["kind"] for k in kinds] == ["Unobserved combination", "Blank cell", "Blank of unrecoverable meaning", "Genuine zero", "Unlabelled test record"]
    sdg = next(k for k in kinds if k["kind"] == "Blank of unrecoverable meaning")
    assert "cannot be uniquely recovered" in sdg["text"]
    assert all(any(ch.isdigit() for ch in k["example"]) for k in kinds)


def test_dictionary_reconciliation_follows_the_workbooks(data):
    rows = {r["topic"]: r for r in dictionary_reconciliation(_stats(data))}
    s = _stats(data)
    assert "Guests" not in s["test_columns"] and "Guests" not in s["dom_test_columns"]
    assert ", ".join(s["test_columns"]) in rows["Target in the test files"]["observed"]
    assert "2022" in rows["Flight-data grain"]["observed"] and "daily" in rows["Flight-data grain"]["observed"]
    assert all(r["documented"] and r["observed"] and r["implication"] for r in rows.values())


def test_technical_checks_use_only_executed_evidence_for_verified(data):
    s = _stats(data)
    allowed = {"verified", "attention", "failed", "not_verified"}
    without = technical_checks(s, None, None, None, True)
    assert {c["status"] for c in without} <= allowed
    by = {c["check"]: c["status"] for c in without}
    assert by["No future information in features"] == "not_verified" and by["Chronological validation"] == "not_verified"
    assert by["Test period can be scored"] == "attention"
    leak = [{"check": "Guest-history features use only Guests from 364+ days earlier", "ok": False, "evidence": ""},
            {"check": "Validation dates follow all training dates", "ok": None, "evidence": ""}]
    assert {c["check"]: c["status"] for c in technical_checks(s, None, None, leak, True)}["No future information in features"] == "failed"
    assert {c["check"]: c["status"] for c in technical_checks(s, None, None, None, False)}["Passenger totals are consistent"] == "failed"
    assert all(c["method"] and c["evidence"] for c in without)


def test_corrected_audit_wording(data):
    f = {x["title"]: x for x in audit_findings(_stats(data))}
    grain = next(x for t, x in f.items() if "monthly in 2022" in t)
    assert "not flights" in grain["handling"]                      # 2022 flights do not feed the last-year guest feature
    sdg = next(x for t, x in f.items() if "Same-Day" in t)
    assert sdg["summary"] == "Same-Day Guests contains blank values whose original meaning cannot be uniquely recovered from the delivered files."
    sel = next(x for t, x in f.items() if "inclusion rules" in t)
    assert sel["handling"].startswith("Modelling decision")
    cols = {r["topic"]: r for r in dictionary_reconciliation(_stats(data))}["Number of columns"]
    assert "32 entries" in cols["observed"] and "31 distinct" in cols["observed"]
