#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig 6 (Part A folder) — COVID-19 counterfactual VLW time series, by disease.

Rebuilds the Part B CF_F1 aggregate time series in the Part A house style:
(a) Depression, (b) Anxiety; bold "(a)"/"(b)" panel labels and disease-specific
y-axis labels. Actual VLW (GBD 2023) vs ETS counterfactual (trained 1990-2019);
the wedge between them is the COVID-19 excess (red = actual>CF, blue = below).

Source data: Fig6_COVID_counterfactual_timeseries.csv (= Part B
  COVID_counterfactual_IE1.0/CF_lmic_aggregate_series.csv).
Outputs (next to the other Part A figures):
  Fig6_COVID_counterfactual_timeseries.pdf  (CSV already present, unchanged)
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "结果V1", "PartA_描述性流行病学", "figures")
CSV = os.path.join(FIG, "Fig6_COVID_counterfactual_timeseries.csv")

DISEASES = ["Depressive disorders", "Anxiety disorders"]
DLAB = {"Depressive disorders": "Depression", "Anxiety disorders": "Anxiety"}
DCOL = {"Depressive disorders": "#1B4F72", "Anxiety disorders": "#C0392B"}
EXCESS_POS = "#D9534F"   # red: actual above counterfactual
EXCESS_NEG = "#5B9BD5"   # blue: actual below counterfactual
ONSET = 2019.5

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.grid": True, "grid.alpha": .25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42,
    "ps.fonttype": 42, "axes.linewidth": .8, "lines.linewidth": 1.6,
})


def kfmt(x, _):
    return f"{x/1000:.1f}K" if x >= 1000 else f"{x:.0f}"


df = pd.read_csv(CSV)
fig, axes = plt.subplots(2, 1, figsize=(8.8, 9.2))

for k, cause in enumerate(DISEASES):
    ax = axes[k]
    d = df[df.disease == cause].sort_values("year")
    col = DCOL[cause]

    # actual VLW + its 95% UI band (full period)
    ax.fill_between(d.year, d.actual_lo, d.actual_hi, color=col, alpha=.12, lw=0)
    ax.plot(d.year, d.actual, color=col, lw=2.0, zorder=5,
            label="Actual VLW (GBD 2023)")

    # counterfactual: anchor at last historical year (2019 actual), then forecast
    fc = d[d.cf_point.notna()].copy()
    anchor = d[d.year == fc.year.min() - 1][["year", "actual"]]
    cf_x = np.concatenate([anchor.year.values, fc.year.values])
    cf_y = np.concatenate([anchor.actual.values, fc.cf_point.values])
    ax.plot(cf_x, cf_y, color="0.35", lw=1.8, ls="--", zorder=4,
            label="Counterfactual (ETS, no COVID-19)")
    # counterfactual prediction interval
    ax.fill_between(fc.year, fc.cf_pi_lo, fc.cf_pi_hi, color="0.5",
                    alpha=.18, lw=0)

    # excess wedge (actual vs CF), red where above, blue where below
    ax.fill_between(fc.year, fc.cf_point, fc.actual,
                    where=(fc.actual >= fc.cf_point),
                    color=EXCESS_POS, alpha=.45, lw=0, interpolate=True)
    ax.fill_between(fc.year, fc.cf_point, fc.actual,
                    where=(fc.actual < fc.cf_point),
                    color=EXCESS_NEG, alpha=.45, lw=0, interpolate=True)

    ax.axvline(ONSET, color="0.4", lw=.9, ls=":", zorder=3)
    ax.text(ONSET + .15, ax.get_ylim()[1], "COVID-19\nonset", fontsize=7.5,
            color="0.4", va="top", ha="left")

    # cumulative 2020-2023 excess annotation (sum of annual point + bounds)
    cum, lo, hi = fc.excess.sum(), fc.excess_lo.sum(), fc.excess_hi.sum()
    ax.text(0.985, 0.04,
            f"Cumulative excess 2020–2023:\n+{cum:,.1f} bn USD\n"
            f"({lo:,.1f}–{hi:,.1f})",
            transform=ax.transAxes, fontsize=8, color=EXCESS_POS,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=EXCESS_POS,
                      lw=.7, alpha=.9))

    ax.set_xticks([1990, 1995, 2000, 2005, 2010, 2015, 2019, 2023])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(kfmt))
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{DLAB[cause]} VLW (billion USD)")
    ax.text(-0.085, 1.02, f"({'ab'[k]})", transform=ax.transAxes,
            fontweight="bold", va="bottom", ha="left", fontsize=13)

# shared legend (top panel)
leg_h = [
    Line2D([0], [0], color="0.2", lw=2.0, label="Actual VLW (GBD 2023)"),
    Line2D([0], [0], color="0.35", lw=1.8, ls="--",
           label="Counterfactual (ETS, no COVID-19)"),
    Patch(fc=EXCESS_POS, alpha=.45, label="COVID-19 excess VLW (actual > CF)"),
    Patch(fc=EXCESS_NEG, alpha=.45, label="Actual below CF"),
]
axes[0].legend(handles=leg_h, frameon=False, loc="upper left", fontsize=8)

fig.tight_layout(h_pad=2.0)
fig.savefig(os.path.join(FIG, "Fig6_COVID_counterfactual_timeseries.pdf"),
            bbox_inches="tight")
plt.close(fig)
print("   wrote Fig6_COVID_counterfactual_timeseries.pdf")
