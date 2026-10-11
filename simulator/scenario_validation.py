"""Validate the What-if engine against the saved models.

    python simulator/scenario_validation.py

1. Zero change reproduces the baseline for every simulator-ready model (one market for market-level models, all markets
   otherwise, over a window that includes a route-day with transit passengers). Identical within 1e-6 guests counts as
   reproduced: Random Forest averages its trees in parallel threads, so even the same input differs by ~1e-11 between calls.
2. The Judges' Demo scenario reproduces its saved reference result (artifacts/judges_demo.json).
3. A +20% seat change passes every consistency check and moves seats, flights, PAX and P2P by exactly +20%.
Writes results/scenario_examples/validation.csv and exits non-zero if any check fails.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from quanara import config, registry  # noqa: E402
from quanara import model_identity as MI  # noqa: E402
from quanara import scenario as S  # noqa: E402
from quanara.integration import get_analytic  # noqa: E402

if __name__ == "__main__":
    a = get_analytic()
    rows = []
    for m in [x for x in registry.list_models() if MI.scenario_ready(x)]:
        mk = "INDIA" if m["scope"] == "international_market" else "ALL"
        r = S.run(S.ScenarioConfig(model_id=m["model_id"], market=mk, start="2025-09-01", end="2025-09-14"), a, save=False)
        gap = float((r.daily["scenario"] - r.daily["baseline"]).abs().max())
        rows.append({"check": "zero change reproduces the baseline", "model_id": m["model_id"], "market": mk, "passed": gap <= 1e-6,
                     "detail": "exact" if gap == 0 else f"within floating-point precision (max |scenario − baseline| = {gap:.1e} guests)"})
    demo = json.loads((config.ARTIFACTS_DIR / "judges_demo.json").read_text())
    r = S.run(S.ScenarioConfig(model_id=demo["model_id"], **demo["scenario"]), a, save=False)
    same = bool(np.isclose(r.summary["baseline_avg_daily_guests"], demo["summary"]["baseline_avg_daily_guests"])
                and np.isclose(r.summary["scenario_avg_daily_guests"], demo["summary"]["scenario_avg_daily_guests"]))
    rows.append({"check": "Judges' Demo reproduces the saved reference", "model_id": demo["model_id"], "market": demo["scenario"]["market"], "passed": same,
                 "detail": f"{r.summary['baseline_avg_daily_guests']:,.1f} → {r.summary['scenario_avg_daily_guests']:,.1f} guests/day ({r.summary['pct_change']:+.2f}%)"})
    av = r.aviation_changes.set_index("quantity")
    pct = {q: av.loc[q, "change_%"] for q in ("Seats", "Flights", "PAX", "P2P passengers")}
    rows.append({"check": "+20% seats: consistency checks and proportional quantities", "model_id": demo["model_id"], "market": demo["scenario"]["market"],
                 "passed": all(c["passed"] for c in r.checks[:5]) and all(abs(v - 20) < 1e-6 for v in pct.values()),
                 "detail": "; ".join(f"{k} {v:+.4f}%" for k, v in pct.items())})
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "scenario_examples" / "validation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.groupby("check")["passed"].agg(["sum", "count"]).to_string())
    print(f"wrote {out.relative_to(ROOT)}")
    sys.exit(0 if df["passed"].all() else 1)
