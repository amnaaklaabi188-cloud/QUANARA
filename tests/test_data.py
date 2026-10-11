"""Data loading, schema validation and cleaning — checked against the real supplied files."""
import numpy as np
import pandas as pd

from quanara import config
from quanara.cleaning import detect_flight_grain
from quanara.data_loading import discover_files, read_raw, sha256
from quanara.schema import EXPECTED_COLUMNS, validate


def test_all_six_files_discovered():
    files = discover_files()
    assert set(files) == {"intl_train", "intl_test", "dom_train", "dom_test", "flight", "dictionary"}


def test_originals_unchanged_against_recorded_checksums():
    sums = dict(line.split("  ", 1)[::-1] for line in (config.RAW_DIR / "SHA256SUMS.txt").read_text().splitlines() if line.strip())
    for name, digest in sums.items():
        assert sha256(config.RAW_DIR / name) == digest, name


def test_schemas_match_and_keys_unique(data):
    for k, df in data.items():
        res = validate(k, df)
        assert res.ok, res.messages
        assert all(c in df.columns for c in EXPECTED_COLUMNS[k])


def test_test_files_have_no_target(data):
    assert "Guests" not in data["intl_test"].columns
    assert "Guests" not in data["dom_test"].columns


def test_no_literal_asterisks_and_suppression_kept_missing(data):
    for k in config.DATASET_PATTERNS:
        raw = read_raw(k)
        assert not any((raw[c].astype(str).str.strip() == "*").any() for c in raw.columns)
    # suppressed Same-Day Guests stay NaN — not converted to 0
    assert data["intl_train"]["Same-Day Guests"].isna().sum() == 35026
    assert (data["intl_train"]["Same-Day Guests"] == 0).sum() == 0


def test_date_coverage(data):
    assert data["intl_train"]["Date"].min() == config.HOTEL_TRAIN_START
    assert data["intl_train"]["Date"].max() == config.HOTEL_TRAIN_END
    assert data["dom_test"]["Date"].min() == config.HOTEL_TEST_START
    assert data["dom_test"]["Date"].max() == config.HOTEL_TEST_END


def test_flight_grain_detected_from_data(data):
    f = data["flight"]
    assert set(f.loc[f["Date"] < "2023-01-01", "grain"]) == {"monthly"}
    assert set(f.loc[f["Date"] >= "2023-01-01", "grain"]) == {"daily"}
    toy = pd.Series(pd.to_datetime(["2022-01-01", "2022-01-01", "2023-01-01", "2023-01-02"]))
    assert list(detect_flight_grain(toy)) == ["monthly", "monthly", "daily", "daily"]


def test_frequency_decodes_to_integer_daily_flights(data):
    d = data["flight"][data["flight"]["grain"] == "daily"]
    assert d["flights_decode_error"].max() < 1e-9
    assert d["flights"].min() >= 1


def test_flight_identities(data):
    f = data["flight"]
    assert np.allclose(f["Total PAX"], f["Total P2P"] + f["Total Transfer"] + f["Total Transit"])
    assert np.allclose(f["Load Factor"], f["Total PAX"] / f["Total Seats"])
    d = f[f["grain"] == "daily"]
    assert int((d["Total Transit"] > 0).sum()) == 3  # effectively zero: 3 of 116,395 daily rows
