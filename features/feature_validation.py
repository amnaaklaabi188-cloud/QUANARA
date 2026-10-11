"""Executable checks on the feature pipeline.

    python features/feature_validation.py

1. The five automated leakage tests of quanara.leakage (guest history uses only t − 364 or earlier, aviation windows
   are trailing, preprocessing is fitted inside each fold, the holdout is chronological, What-if excludes same-day
   hotel fields).
2. Trailing windows include the current day: p2p_7d on a sample of dates equals the sum of daily P2P over t−6…t
   recomputed directly from the route-day rows.
3. Ratios are computed from sums: p2p_share_7d = Σ P2P ÷ Σ PAX over the same window.
4. The feature matrix is deterministic: building it twice gives the same fingerprint.
Exits non-zero if any check fails.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from quanara import leakage, registry  # noqa: E402
from quanara.feature_facts import matrix_fingerprint, training_matrix  # noqa: E402
from quanara.integration import get_analytic, model_frame  # noqa: E402

if __name__ == "__main__":
    a = get_analytic()
    results = []
    mid = registry.active_model_id()
    for c in leakage.run_checks(a, registry.load_metadata(mid) if mid else None):
        results.append((c["check"], c["ok"] is not False, c["evidence"]))
    f = model_frame(a, "international_total", "whatif").set_index("Date")
    rows = a.flight_daily_rows
    daily = rows.groupby("Date")[["Total P2P", "Total PAX"]].sum().asfreq("D", fill_value=0)
    rng = np.random.default_rng(42)
    dates = pd.to_datetime(rng.choice(f.index[(f.index >= "2023-03-01") & (f.index <= "2026-02-20")], 8, replace=False))
    ok7 = all(np.isclose(f.loc[d, "p2p_7d"], daily.loc[d - pd.Timedelta(days=6):d, "Total P2P"].sum()) for d in dates)
    results.append(("p2p_7d = Σ daily P2P over t−6…t (current day included)", ok7, f"{len(dates)} random dates recomputed from route-day rows"))
    oks = all(np.isclose(f.loc[d, "p2p_share_7d"], daily.loc[d - pd.Timedelta(days=6):d, "Total P2P"].sum() / daily.loc[d - pd.Timedelta(days=6):d, "Total PAX"].sum())
              for d in dates)
    results.append(("p2p_share_7d = Σ P2P ÷ Σ PAX over the same window", oks, "ratio of sums, never an average of daily ratios"))
    fp1, fp2 = matrix_fingerprint(training_matrix(a, "international_total", "whatif")), matrix_fingerprint(training_matrix(a, "international_total", "whatif"))
    results.append(("feature matrix is deterministic", fp1 == fp2, f"fingerprint {fp1}"))
    for name, passed, ev in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name}\n      {ev}")
    sys.exit(0 if all(p for _, p, _ in results) else 1)
