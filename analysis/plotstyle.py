"""Matplotlib style shared by the figure scripts (QuanAra emerald, charcoal and grey)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EMERALD, DEEP, CHARCOAL, GREY, LIGHT = "#059669", "#047857", "#1f2937", "#9ca3af", "#e5e7eb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c9cfcc", "axes.labelcolor": CHARCOAL, "xtick.color": "#4b5563", "ytick.color": "#4b5563",
                     "axes.grid": True, "grid.color": "#e9efec", "grid.linewidth": 0.8, "figure.dpi": 110, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 12})
