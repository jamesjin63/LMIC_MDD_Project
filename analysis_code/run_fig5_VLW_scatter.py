#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig 5 (Part A folder) — GDP per capita vs VLW/GDP, by disease, 2023.

Per-disease counterpart of Part B's combined F5 scatter: one panel per disease
(a = Depression, b = Anxiety), x = GDP per capita (PPP, log scale), y = VLW/GDP
(%), points coloured by income group and sized by DALYs, with a LOWESS trend
curve (+95% UI band) added to each panel.

Source: Part B VLW table (0-19 years, Both sexes):
  结果V1/PartB_经济负担/VLW_2disease_IE1.0/T1_country_disease_2023.csv
Outputs (next to the other Part A figures):
  Fig5_scatter_GDPpc_VLW_by_disease.pdf  +  .csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "结果V1", "PartA_描述性流行病学", "figures")
T1 = os.path.join(ROOT, "结果V1", "PartB_经济负担", "VLW_2disease_IE1.0",
                  "T1_country_disease_2023.csv")

DISEASES = ["Depressive disorders", "Anxiety disorders"]
DLAB = {"Depressive disorders": "Depression", "Anxiety disorders": "Anxiety"}
INCOME = ["Low income", "Lower middle income", "Upper middle income"]
ILAB = {"Low income": "Low", "Lower middle income": "Lower-mid",
        "Upper middle income": "Upper-mid"}
ICOL = {"Low income": "#9467bd", "Lower middle income": "#E67E22",
        "Upper middle income": "#16A085"}

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


def pt(s):  # point estimate before " ("; strip thousands commas
    return float(str(s).split("(")[0].strip().replace(",", ""))


def ui(s):  # (lower, upper) from "pt (lo–hi)"; en-dash separated
    inside = str(s).split("(", 1)[1].rstrip(")").replace(",", "")
    lo, hi = inside.split("–")
    return float(lo), float(hi)


# --- load VLW (0-19y, Both sexes), per country per disease ---
t1 = pd.read_csv(T1)
assert set(t1["Age"].unique()) == {"<20 years"}, "T1 is not 0-19 only!"
t1 = t1[t1["Sex"] == "Both"].copy()
t1["VLW_GDP_pct"] = t1["VLW/GDP (%)"].map(pt)
t1[["VLW_GDP_pct_lo", "VLW_GDP_pct_hi"]] = t1["VLW/GDP (%)"].apply(
    lambda s: pd.Series(ui(s)))
t1["DALYs_n"] = t1["DALYs"].map(pt)
t1[["DALYs_n_lo", "DALYs_n_hi"]] = t1["DALYs"].apply(lambda s: pd.Series(ui(s)))
t1["GDP_pc"] = t1["GDP_pc_PPP"].astype(float)

out = t1[["Disease", "Country", "Income_Group", "GDP_pc",
          "DALYs_n", "DALYs_n_lo", "DALYs_n_hi",
          "VLW_GDP_pct", "VLW_GDP_pct_lo", "VLW_GDP_pct_hi"]].sort_values(
          ["Disease", "GDP_pc"])
out.to_csv(os.path.join(FIG, "Fig5_scatter_GDPpc_VLW_by_disease.csv"), index=False)
print("   wrote Fig5_scatter_GDPpc_VLW_by_disease.csv")

# size legend (DALYs) — shared scale across panels
dmax = t1["DALYs_n"].max()
size_ref = [1e4, 1e5, 5e5, 2e6]
size_lab = ["10K", "100K", "500K", "2M"]
def msize(d):  # area in pts^2, sqrt scaling so area ~ DALYs
    return 12 + 230 * np.sqrt(d / dmax)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
for k, cause in enumerate(DISEASES):
    ax = axes[k]
    d = t1[t1.Disease == cause].dropna(subset=["GDP_pc", "VLW_GDP_pct"])
    for g in INCOME:
        gg = d[d.Income_Group == g]
        ax.scatter(gg["GDP_pc"], gg["VLW_GDP_pct"], s=msize(gg["DALYs_n"]),
                   c=ICOL[g], alpha=.6, edgecolor="white", linewidth=.4,
                   zorder=3, label=ILAB[g])
    # LOWESS trend on log10(GDP_pc)
    lx = np.log10(d["GDP_pc"].values)
    sm = lowess(d["VLW_GDP_pct"].values, lx, frac=.7, return_sorted=True)
    ax.plot(10 ** sm[:, 0], sm[:, 1], color="0.15", lw=2.0, zorder=4)
    # bootstrap 95% band for the LOWESS curve
    rng = np.random.default_rng(42)
    grid = sm[:, 0]
    boot = np.full((400, len(grid)), np.nan)
    n = len(lx)
    for b in range(400):
        idx = rng.integers(0, n, n)
        try:
            sb = lowess(d["VLW_GDP_pct"].values[idx], lx[idx],
                        frac=.7, xvals=grid)
            boot[b] = sb
        except Exception:
            pass
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    ax.fill_between(10 ** grid, lo, hi, color="0.15", alpha=.12, zorder=2)

    # Pearson correlation (log10 GDP per capita vs VLW/GDP) — top-left
    r, p = pearsonr(lx, d["VLW_GDP_pct"].values)
    pstr = "P < 0.001" if p < 1e-3 else f"P = {p:.3f}"
    ax.text(0.03, 0.97, f"r = {r:.2f}\n{pstr}", transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=.6,
                      alpha=.85))

    ax.set_xscale("log")
    xmin, xmax = d["GDP_pc"].min(), d["GDP_pc"].max()
    ax.set_xlim(xmin * 0.9, xmax * 1.08)
    ax.set_xticks([2000, 5000, 10000, 25000])
    ax.set_xticklabels(["2k", "5k", "10k", "25k"])
    ax.set_xlabel("GDP per capita (PPP, USD)")
    if k == 0:
        ax.set_ylabel("VLW / GDP (%)")
    ax.set_title(f"({'ab'[k]}) {DLAB[cause]}", loc="left",
                 fontweight="bold", fontsize=12)

# shared legends: income (colour) + DALYs (size)
inc_h = [Line2D([0], [0], marker="o", ls="", mfc=ICOL[g], mec="white",
                ms=8, label=ILAB[g]) for g in INCOME]
leg1 = axes[1].legend(handles=inc_h, title="Income group", frameon=False,
                      loc="upper right", fontsize=8, title_fontsize=8)
axes[1].add_artist(leg1)
sz_h = [Line2D([0], [0], marker="o", ls="", mfc="0.6", mec="white",
               ms=np.sqrt(msize(s)) / 1.6, label=l)
        for s, l in zip(size_ref, size_lab)]
axes[0].legend(handles=sz_h, title="DALYs", frameon=False, loc="upper right",
               fontsize=8, title_fontsize=8, labelspacing=1.3,
               borderpad=1.0, handletextpad=1.2)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "Fig5_scatter_GDPpc_VLW_by_disease.pdf"),
            bbox_inches="tight")
plt.close(fig)
print("   wrote Fig5_scatter_GDPpc_VLW_by_disease.pdf")
print(f"   countries/panel: {t1[t1.Disease==DISEASES[0]].shape[0]} | "
      f"VLW/GDP range {t1.VLW_GDP_pct.min():.2f}-{t1.VLW_GDP_pct.max():.2f}%")
