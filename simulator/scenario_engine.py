"""Run one What-if scenario with a saved model — the same engine as the app's What-if Simulator.

    python simulator/scenario_engine.py --market "RUSSIAN FEDERATION" --month 2025-09 --seats 20
    python simulator/scenario_engine.py --model-id MKT-WI-catboost-20261010-123103-7d603b --market ALL --start 2025-12-01 --end 2025-12-31 --load-factor-pp 5

Steps (quanara.scenario.run): copy the route-day flight rows → apply the change to the selected departure country
(sustained over the 56-day feature look-back unless --not-sustained) → check consistency (PAX = P2P + transfer +
transit, load factor limits, no negative counts) → rebuild every feature with the training code → predict baseline
and scenario with the saved deployment pipeline on identical dates. Outputs are model-based sensitivity estimates,
not causal effects. Writes a JSON summary to results/scenario_examples/.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from quanara import registry  # noqa: E402
from quanara import model_identity as MI  # noqa: E402
from quanara import scenario as S  # noqa: E402
from quanara.integration import get_analytic  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", help="a simulator-ready What-if model (default: the active model)")
    ap.add_argument("--market", default="ALL", help="departure country as in the flight file (upper case), or ALL")
    ap.add_argument("--month", help="YYYY-MM (alternative to --start/--end)")
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--seats", type=float, default=0.0, help="seat capacity change in %% (frequency follows unless --frequency is given)")
    ap.add_argument("--frequency", type=float, help="flight frequency change in %%; seats per flight absorb the difference")
    ap.add_argument("--load-factor-pp", type=float, default=0.0, help="load factor shift in percentage points")
    ap.add_argument("--p2p-share-pp", type=float, default=0.0, help="P2P share shift in percentage points")
    ap.add_argument("--not-sustained", action="store_true", help="start the change on the first day of the period")
    a_ = ap.parse_args()
    mid = a_.model_id or registry.active_model_id()
    md = registry.load_metadata(mid)
    if not MI.scenario_ready(md):
        sys.exit(f"{mid} cannot drive scenarios: {MI.not_ready_reason(md)}")
    if a_.month:
        p = pd.Period(a_.month, "M"); start, end = str(p.start_time.date()), str(p.end_time.date())
    else:
        start, end = a_.start, a_.end
    cfg = S.ScenarioConfig(model_id=mid, market=a_.market, start=start, end=end, seats_change_pct=a_.seats,
                           frequency_mode="independent" if a_.frequency is not None else "follow_seats", frequency_change_pct=a_.frequency or 0.0,
                           load_factor_mode="delta" if a_.load_factor_pp else "baseline", load_factor_value=a_.load_factor_pp,
                           p2p_share_mode="delta" if a_.p2p_share_pp else "baseline", p2p_share_value=a_.p2p_share_pp, sustained=not a_.not_sustained)
    res = S.run(cfg, get_analytic(), save=False)
    s = res.summary
    print(f"Model     {MI.name(md)} ({mid})\nScenario  {a_.market} · {start} → {end} · seats {a_.seats:+.0f}%"
          + (f" · frequency {a_.frequency:+.0f}%" if a_.frequency is not None else "") + (f" · load factor {a_.load_factor_pp:+.1f} pp" if a_.load_factor_pp else "")
          + (f" · P2P share {a_.p2p_share_pp:+.1f} pp" if a_.p2p_share_pp else ""))
    print(f"Baseline  {s['baseline_avg_daily_guests']:,.1f} guests/day\nScenario  {s['scenario_avg_daily_guests']:,.1f} guests/day\n"
          f"Change    {s['abs_change_avg_daily']:+,.1f} guests/day ({s['pct_change']:+.2f}%) · {s['abs_change_guest_days']:+,.0f} guest-days over {s['days']} days")
    print(f"Checks    {sum(c['passed'] for c in res.checks)}/{len(res.checks)} consistency checks passed")
    for w in res.warnings:
        print("Warning  ", w)
    out = ROOT / "results" / "scenario_examples" / f"{mid}__{a_.market.replace(' ', '_')}__{start}_{end}__seats{a_.seats:+.0f}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model_id": mid, "model": MI.name(md), "config": res.config, "summary": s,
                               "aviation_changes": res.aviation_changes.to_dict("records"), "checks": res.checks, "warnings": res.warnings,
                               "note": "Model-based sensitivity estimate under changed aviation inputs; not a causal effect."}, indent=2, default=str))
    print(f"wrote {out.relative_to(ROOT)}")
