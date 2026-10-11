"""Diagnostics that decide how far the correlations can be trusted.

    python analysis/statistical_diagnostics.py

For every scope: lag-1 autocorrelation of each feature and of Guests, the effective sample size
n_eff = n(1 − ρx·ρy)/(1 + ρx·ρy), Pearson on levels versus day-over-day and week-over-week changes (a correlation
that disappears after differencing is driven by shared trend or season), and the lagged correlation between
7-day P2P arrivals ending k days earlier and Guests. Writes results/correlations/<scope>_diagnostics.csv,
<scope>_lag_p2p_7d.csv and figures/correlations/lag_p2p_7d.png.
"""
import pandas as pd

from correlations import FIGURES, LEVELS, analytic, features_of, frame, write
from plotstyle import CHARCOAL, EMERALD, GREY, plt
from quanara import config
from quanara import correlation as C

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for (scope, lv), col in zip(LEVELS.items(), (EMERALD, CHARCOAL, GREY)):
        df = frame(scope)
        obs = df["Guests"].notna()
        rows = []
        for f in features_of(df):
            lev = C.pair_stats(df.loc[obs, f], df.loc[obs, "Guests"])
            d1 = C.pair_stats(C.transform(df[f].astype(float), "diff1")[obs], C.transform(df["Guests"].astype(float), "diff1")[obs])
            d7 = C.pair_stats(C.transform(df[f].astype(float), "diff7")[obs], C.transform(df["Guests"].astype(float), "diff7")[obs])
            rows.append({"feature": f, "label": C.label(f), "n": lev["n"], "pearson_levels": lev["pearson"], "pearson_day_over_day": d1["pearson"],
                         "pearson_week_over_week": d7["pearson"], "lag1_autocorr_feature": lev["rho_x"], "lag1_autocorr_guests": lev["rho_y"],
                         "effective_n": lev["n_eff"]})
        diag = pd.DataFrame(rows)
        write(diag, f"{scope}_diagnostics.csv")
        full = C.eda_frame(analytic(), LEVELS[scope])
        lag = C.lag_correlation(full, "p2p", 7, "Guests", 42, "pearson", config.MODEL_START, config.HOTEL_TRAIN_END)
        write(lag, f"{scope}_lag_p2p_7d.csv")
        ax.plot(lag["lag"], lag["r"], color=col, linewidth=2, marker="o", markersize=3, label=lv)
        best = lag.loc[lag["r"].abs().idxmax()]
        print(f"{lv}: median effective n {diag['effective_n'].median():.0f} of {int(diag['n'].max())} days; "
              f"strongest 7-day P2P lag {int(best['lag'])} d (r = {best['r']:.3f}); same-day r = {lag.loc[0, 'r']:.3f}")
    ax.axhline(0, color="#c9cfcc", linewidth=1)
    ax.set_xlabel("Lag k (days): 7-day P2P window ends k days before the guest date"); ax.set_ylabel("Pearson r")
    ax.set_title("Lagged association between P2P arrivals and Guests", loc="left"); ax.legend(frameon=False)
    p = FIGURES / "correlations" / "lag_p2p_7d.png"; p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p); plt.close(fig)
