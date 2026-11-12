"""Shared matplotlib style for role B's figures (vector PDF, >= 8pt text)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GRAY,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,
        "figure.dpi": 150,
    }
)
