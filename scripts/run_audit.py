"""Milestone 1: audit the five workbooks against Data_Dictionary.pdf.

Writes reports/audit_report.md, reports/audit.json and reports/field_dictionary.csv.
Usage:  python scripts/run_audit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quanara import config  # noqa: E402
from quanara.audit import render_markdown, run_audit, to_jsonable  # noqa: E402
from quanara.data_loading import file_inventory, load_all, read_raw  # noqa: E402
from quanara.dictionary import field_table  # noqa: E402


def count_literal_asterisks() -> dict[str, int]:
    out = {}
    for k in config.DATASET_PATTERNS:
        raw = read_raw(k)
        out[k] = int(sum((raw[c].astype(str).str.strip() == "*").sum() for c in raw.columns))
    return out


def main() -> None:
    config.ensure_dirs()
    inv = file_inventory()
    data = load_all()
    audit = run_audit(data, inv)
    stars = count_literal_asterisks()
    audit["flight"]["literal_asterisks"] = stars["flight"]
    audit["literal_asterisks_by_file"] = stars
    (config.REPORTS_DIR / "audit.json").write_text(json.dumps(to_jsonable(audit), indent=2))
    (config.REPORTS_DIR / "audit_report.md").write_text(render_markdown(audit))
    field_table().to_csv(config.REPORTS_DIR / "field_dictionary.csv", index=False)
    print(inv.to_string(index=False))
    print("literal '*' cells per file:", stars)
    for w in audit["warnings"]:
        print(f"[{w['level']}] {w['title']}: {w['detail']}")
    print("wrote", config.REPORTS_DIR / "audit_report.md")


if __name__ == "__main__":
    main()
