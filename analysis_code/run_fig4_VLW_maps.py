#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig 4 (Part A folder) — Economic burden (VLW) world maps, 2x2, 2023.
Merges the two per-disease F6b maps into one 2x2 in the same compact style as
Fig3: rows = disease (Depression, Anxiety); cols = (VLW absolute, VLW/GDP).

Source: Part B VLW table (0-19 years, Both sexes):
  结果V1/PartB_经济负担/VLW_2disease_IE1.0/T1_country_disease_2023.csv
Country -> location_id via data/204_with_LMIC.csv; polygons from df_world2.geojson.
Outputs (next to the other Part A figures):
  Fig4_VLW_maps_2x2.pdf  +  Fig4_VLW_maps_2x2.csv
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import FuncNorm, LinearSegmentedColormap

# Nature Cities-style sequentials, both distinct from Fig3's warm YlOrRd.
# VLW absolute -> cool teal->green->deep indigo (mako-like).
# VLW/GDP      -> cool pink->magenta->deep plum (lipari-like), to read the two
#                 columns apart. Light = low, dark = high.
NC_VLW = LinearSegmentedColormap.from_list("nc_mako", [
    "#DEF5E5", "#A8DBC4", "#74C7AC", "#46BEAD", "#35A1AB",
    "#3487A6", "#366FA2", "#3D5296", "#403A75", "#2B1B45", "#160E26",
])
NC_GDP = LinearSegmentedColormap.from_list("nc_plum", [
    "#FCEEF5", "#F4C9E0", "#E89BC8", "#D96BAE", "#C43E93",
    "#A52878", "#7E1F5C", "#581743", "#330E29", "#160611",
])

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "结果V1", "PartA_描述性流行病学", "figures")
T1 = os.path.join(ROOT, "结果V1", "PartB_经济负担", "VLW_2disease_IE1.0",
                  "T1_country_disease_2023.csv")
DISEASES = ["Depressive disorders", "Anxiety disorders"]
DLAB = {"Depressive disorders": "Depression", "Anxiety disorders": "Anxiety"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False, "font.size": 9, "axes.titlesize": 10,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})


def pt(s):  # point estimate before " ("; strip thousands commas
    return float(str(s).split("(")[0].strip().replace(",", ""))


def ui(s):  # (lower, upper) from "pt (lo–hi)"; en-dash separated
    inside = str(s).split("(", 1)[1].rstrip(")").replace(",", "")
    lo, hi = inside.split("–")
    return float(lo), float(hi)


# --- load VLW (0-19y, Both sexes), map Country -> location_id ---
t1 = pd.read_csv(T1)
assert set(t1["Age"].unique()) == {"<20 years"}, "T1 is not 0-19 only!"
t1 = t1[t1["Sex"] == "Both"].copy()
t1["VLW_bn"] = t1["VLW (billion USD)"].map(pt)
t1[["VLW_bn_lo", "VLW_bn_hi"]] = t1["VLW (billion USD)"].apply(
    lambda s: pd.Series(ui(s)))
t1["VLW_GDP_pct"] = t1["VLW/GDP (%)"].map(pt)
t1[["VLW_GDP_pct_lo", "VLW_GDP_pct_hi"]] = t1["VLW/GDP (%)"].apply(
    lambda s: pd.Series(ui(s)))
names = pd.read_csv(os.path.join(DATA, "204_with_LMIC.csv"))[["location_name", "location_id"]]
t1 = t1.merge(names, left_on="Country", right_on="location_name", how="left")
miss = t1[t1.location_id.isna()].Country.unique()
if len(miss):
    print("   [warn] no location_id for:", list(miss))

world = gpd.read_file(os.path.join(DATA, "df_world2.geojson"))
world["location_id"] = pd.to_numeric(world["location_id"], errors="coerce")

# --- save source CSV next to the figure ---
out_csv = t1[["Disease", "location_id", "Country", "Income_Group",
              "VLW_bn", "VLW_bn_lo", "VLW_bn_hi",
              "VLW_GDP_pct", "VLW_GDP_pct_lo", "VLW_GDP_pct_hi"]].sort_values(
              ["Disease", "VLW_bn"], ascending=[True, False])
out_csv.to_csv(os.path.join(FIG, "Fig4_VLW_maps_2x2.csv"), index=False)
print("   wrote 结果V1/PartA_描述性流行病学/figures/Fig4_VLW_maps_2x2.csv")

# --- 2x2 maps: rows = disease; cols = (VLW absolute [log], VLW/GDP [linear]) ---
fig, axes = plt.subplots(2, 2, figsize=(15, 5.6))
lab = "abcd"
cbar = dict(shrink=.85, pad=0.008, fraction=0.022, aspect=22)
cols = [("VLW_bn", "VLW (billion USD)", True, NC_VLW),
        ("VLW_GDP_pct", "VLW/GDP (%)", False, NC_GDP)]
for i, cause in enumerate(DISEASES):
    d = t1[t1.Disease == cause]
    for j, (col, clab, logscale, cmap) in enumerate(cols):
        ax = axes[i, j]
        m = world.merge(d[["location_id", col]], on="location_id", how="left")
        kw = dict(column=col, cmap=cmap, legend=True, ax=ax, edgecolor="0.5",
                  linewidth=.3,
                  missing_kwds={"color": "white", "edgecolor": "0.75", "linewidth": .15},
                  legend_kwds={**cbar, "label": clab})
        if logscale:
            vmax = float(np.nanmax(d[col].values))
            kw["norm"] = FuncNorm((np.log1p, np.expm1), vmin=0, vmax=vmax)
        m.plot(**kw)
        ax.set_title(f"({lab[i*2 + j]}) {DLAB[cause]} {clab.split(' (')[0]}, 2023",
                     fontsize=9)
for ax in axes.flat:
    ax.set_xlim(-180, 180); ax.set_ylim(-57, 85); ax.margins(0); ax.axis("off")
fig.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.01,
                    wspace=0.02, hspace=0.08)
fig.savefig(os.path.join(FIG, "Fig4_VLW_maps_2x2.pdf"), bbox_inches="tight")
plt.close(fig)
print("   wrote 结果V1/PartA_描述性流行病学/figures/Fig4_VLW_maps_2x2.pdf")
print(f"   countries mapped: {t1.location_id.notna().sum()} | "
      f"combined VLW = {t1.VLW_bn.sum():,.0f} bn USD (Both, 0-19, 2023)")
