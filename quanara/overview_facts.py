"""Facts shown on the Welcome & Overview page — computed from the competition files and the
active model's saved artefacts at run time. Nothing here is hard-coded."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, registry
from .data_loading import load_all
from .evaluation import metrics


def data_facts(data: dict[str, pd.DataFrame] | None = None) -> dict:
    d = data or load_all()
    fl, it, dt, ite, dte = d["flight"], d["intl_train"], d["dom_train"], d["intl_test"], d["dom_test"]
    daily = fl[fl["grain"] == "daily"]
    # monthly totals are the only grain at which 2022 (monthly) and 2023+ (daily) flight rows are comparable
    pax_m = fl.groupby(fl["Date"].dt.to_period("M"))["Total PAX"].sum()
    guests_m = it.groupby(it["Date"].dt.to_period("M"))["Guests"].sum()
    daily_intl = it.groupby("Date")["Guests"].sum()
    countries = set(fl["market_key"].unique())
    nats = set(it["Nationality"].unique())
    # daily international guests vs. P2P arrivals over the trailing 7 days, inside the modelling window
    p2p7 = daily.groupby("Date")["Total P2P"].sum().asfreq("D", fill_value=0).rolling(7, min_periods=7).sum()
    joined = pd.concat([daily_intl.rename("g"), p2p7.rename("p")], axis=1).loc[config.MODEL_START:config.MODEL_END].dropna()
    r = float(joined["g"].corr(joined["p"])) if len(joined) > 30 else float("nan")
    return {
        "n_datasets": int(len(d)),
        "p2p7_guests_r": r,
        # flights — all rows unless stated
        "flight_rows": int(len(fl)),
        "flight_rows_monthly": int((fl["grain"] == "monthly").sum()),
        "flight_rows_daily": int(len(daily)),
        "flight_start": fl["Date"].min(), "flight_end": fl["Date"].max(),
        "flight_daily_start": daily["Date"].min(),
        "flight_countries": int(fl["Departure Country Name"].nunique()),
        "flight_airlines": int(fl["Airline Name"].nunique()),
        "flight_cities": int(fl["Departure City"].nunique()),
        "flight_monthly_pax": [float(v) for v in pax_m.to_numpy()],
        # hotels
        "intl_rows": int(len(it)),
        "intl_nationalities": int(len(nats)),
        "dom_days": int(len(dt)),
        "labelled_records": int(len(it) + len(dt)),
        "hotel_days": int(it["Date"].nunique()),
        "hotel_start": it["Date"].min(), "hotel_end": it["Date"].max(),
        "test_intl_rows": int(len(ite)), "test_dom_rows": int(len(dte)),
        "test_start": min(ite["Date"].min(), dte["Date"].min()), "test_end": max(ite["Date"].max(), dte["Date"].max()),
        "test_files_have_guests": bool("Guests" in ite.columns or "Guests" in dte.columns),
        "intl_monthly_guests": [float(v) for v in guests_m.to_numpy()],
        "intl_avg_daily_total": float(daily_intl.mean()),
        "dom_avg_daily": float(dt["Guests"].mean()),
        "matched_markets": int(len(nats & countries)),
    }


def model_facts() -> dict | None:
    """The ACTIVE model only. Metrics are recomputed from that model's saved holdout predictions and
    cross-checked against its metadata, so nothing from another run can leak in."""
    mid = registry.active_model_id()
    if mid is None:
        return None
    md = registry.load_metadata(mid)
    pred = registry.load_table(mid, "predictions_holdout")
    m = metrics(pred["y_true"], pred["y_pred"])
    saved = md["metrics_holdout"]
    consistent = (md.get("model_id") == mid and m["n"] == saved["n"]
                  and all(np.isclose(m[k], saved[k], rtol=1e-9, atol=1e-9) for k in ("MAE", "RMSE", "R2", "WMAPE"))
                  and str(pred["Date"].min().date()) == md["periods"]["holdout"][0]
                  and str(pred["Date"].max().date()) == md["periods"]["holdout"][1])
    return {
        "model_id": mid,
        "algorithm": md["algorithm_label"],
        "scope": md["scope"],
        "scope_label": md["scope_label"],
        "grain": md["grain"],
        "n_markets": len(md.get("markets") or []),
        "mae": m["MAE"], "rmse": m["RMSE"], "r2": m["R2"], "wmape": m["WMAPE"], "n_holdout": m["n"],
        "holdout": [str(pred["Date"].min().date()), str(pred["Date"].max().date())],
        "eval_fit": md["periods"]["eval_fit"],
        "deployment_fit": md["periods"]["deployment_fit"],
        "mode": md["mode"],
        "consistent": bool(consistent),
    }
