#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part A — Descriptive epidemiology of childhood/adolescent (0-19y) Depressive &
Anxiety disorders in LMICs (GBD 2023), DALYs-based.

Figure style follows the GBD main-text house style learned from
NTD/344.../final_main_figures/code (gbd-figure-style skill):
  no figure titles; bold (a)/(b) panel labels top-left; no top/right spines;
  grid alpha 0.25; frameless legends; 95% UI ribbons; twin-axis bars+rate-lines;
  YlOrRd level maps + RdBu_r EAPC maps; A/B concentration-curve + frontier;
  TrueType-embedded PDF at 300 dpi.

NOTE: Prevalence/Incidence not yet extracted (data on disk = DALYs only).
Outputs: 结果V1/PartA_描述性流行病学/{tables,figures}
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "结果V1", "PartA_描述性流行病学")
TBL  = os.path.join(OUT, "tables")
FIG  = os.path.join(OUT, "figures")
for d in (TBL, FIG):
    os.makedirs(d, exist_ok=True)

DISEASES = ["Depressive disorders", "Anxiety disorders"]
DLAB     = {"Depressive disorders": "Depression", "Anxiety disorders": "Anxiety"}
DCOL     = {"Depressive disorders": "#C44E52", "Anxiety disorders": "#4C72B0"}
INCOME   = ["Low income", "Lower middle income", "Upper middle income"]
ILAB     = {"Low income": "Low", "Lower middle income": "Lower-mid",
            "Upper middle income": "Upper-mid"}
ICOL     = {"Low income": "#9467bd", "Lower middle income": "#E67E22",
            "Upper middle income": "#16A085"}
AGE_BANDS = ["<5 years", "5-9 years", "10-14 years", "15-19 years"]
AGE_SHORT = {"<5 years": "<5", "5-9 years": "5-9",
             "10-14 years": "10-14", "15-19 years": "15-19"}
PER = "per 100,000"


def set_style():
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


def panel(ax, k, off=-0.12, fs=13):
    ax.text(off, 1.02, f"({'abcd'[k]})", transform=ax.transAxes,
            fontweight="bold", va="bottom", ha="left", fontsize=fs)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("   wrote", os.path.relpath(os.path.join(FIG, name), ROOT))


def save_csv(df, name):
    """write a figure's source data next to the figure (figures/ folder)."""
    df.to_csv(os.path.join(FIG, name), index=False)
    print("   wrote", os.path.relpath(os.path.join(FIG, name), ROOT))


set_style()

# ---------------------------------------------------------------- load
print("[1/5] Loading data ...")
lmic = pd.read_csv(os.path.join(DATA, "204_with_LMIC.csv"))
lmic = lmic[lmic["LMIC"] == 1][["location_id", "location_name", "LMIC_group"]]
lmic_ids = set(lmic["location_id"])

cols = ["location_id", "sex_name", "age_name", "cause_name",
        "metric_name", "year", "val", "lower", "upper"]
df = pd.read_csv(os.path.join(DATA, "merged_gbd_data.csv"), usecols=cols)
df = df[df["cause_name"].isin(DISEASES) & df["location_id"].isin(lmic_ids)].copy()

num = df[df["metric_name"] == "Number"].rename(
    columns={"val": "DALY_n", "lower": "n_lo", "upper": "n_hi"})
rate = df[df["metric_name"] == "Rate"][
    ["location_id", "sex_name", "age_name", "cause_name", "year",
     "val", "lower", "upper"]
].rename(columns={"val": "DALY_rate", "lower": "DALY_rate_lo",
                  "upper": "DALY_rate_hi"})
d = num.merge(rate, on=["location_id", "sex_name", "age_name", "cause_name", "year"],
              how="left").merge(lmic, on="location_id", how="left")
d["pop"] = np.where(d["DALY_rate"] > 0, d["DALY_n"] / d["DALY_rate"] * 1e5, np.nan)
print(f"   rows: {len(d):,} | LMIC countries: {d.location_id.nunique()}")

# GDP per capita (2023) by location_id, via ISO3 from the geojson
import geopandas as gpd
_gj = gpd.read_file(os.path.join(DATA, "df_world2.geojson"))[["location_id", "brk_a3"]].dropna()
_gj["location_id"] = _gj.location_id.astype(int)
_gdp = pd.read_csv(os.path.join(DATA, "gdp.csv"))
_gdp = _gdp[_gdp.year == 2023][["iso3c", "NY.GDP.PCAP.PP.CD"]].dropna()
GDPPC = (_gj.merge(_gdp, left_on="brk_a3", right_on="iso3c")
         .set_index("location_id")["NY.GDP.PCAP.PP.CD"].to_dict())


def agg_rate(g):
    pop = g["pop"].sum()
    f = (lambda c: g[c].sum() / pop * 1e5 if pop > 0 else np.nan)
    return pd.Series({"DALY_n": g["DALY_n"].sum(), "n_lo": g["n_lo"].sum(),
                      "n_hi": g["n_hi"].sum(), "pop": pop,
                      "rate": f("DALY_n"), "rate_lo": f("n_lo"), "rate_hi": f("n_hi")})


def eapc(years, rates):
    m = (rates > 0) & np.isfinite(rates)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan
    y, r = np.asarray(years)[m], np.log(np.asarray(rates)[m])
    n = len(y)
    b, a = np.polyfit(y, r, 1)
    resid = r - (a + b * y)
    se = np.sqrt((resid @ resid) / (n - 2) / np.sum((y - y.mean()) ** 2))
    return ((np.exp(b) - 1) * 100, (np.exp(b - 1.96 * se) - 1) * 100,
            (np.exp(b + 1.96 * se) - 1) * 100)


agg = d[d["age_name"] == "<20 years"]

# ============================================================ tables D1-D7
print("[2/5] Tables D1-D7 ...")
d1 = (agg[agg.year == 2023].groupby(["cause_name", "LMIC_group", "sex_name"],
      observed=True).apply(agg_rate, include_groups=False).reset_index())
d1.to_csv(os.path.join(TBL, "D1_burden_2023_by_disease_income.csv"), index=False)

both = agg[agg.sex_name == "Both"]
d2_all = both.groupby(["cause_name", "year"], observed=True).apply(
    agg_rate, include_groups=False).reset_index()
d2_all.insert(1, "LMIC_group", "All LMIC")
d2_inc = both.groupby(["cause_name", "LMIC_group", "year"], observed=True).apply(
    agg_rate, include_groups=False).reset_index()
pd.concat([d2_all, d2_inc], ignore_index=True).to_csv(
    os.path.join(TBL, "D2_trend_1990_2023.csv"), index=False)

rows = []
for cause in DISEASES:
    groups = [("All LMIC", d2_all[d2_all.cause_name == cause])] + \
        [(g, d2_inc[(d2_inc.cause_name == cause) & (d2_inc.LMIC_group == g)])
         for g in INCOME]
    for grp, sub in groups:
        sub = sub.sort_values("year")
        full = eapc(sub.year, sub.rate)
        pre = eapc(sub[sub.year <= 2019].year, sub[sub.year <= 2019].rate)
        post = eapc(sub[sub.year >= 2019].year, sub[sub.year >= 2019].rate)
        rows.append({"cause_name": cause, "group": grp, "EAPC_1990_2023": full[0],
                     "EAPC_lo": full[1], "EAPC_hi": full[2],
                     "EAPC_pre2019": pre[0], "EAPC_post2019": post[0]})
d3 = pd.DataFrame(rows)
d3.to_csv(os.path.join(TBL, "D3_EAPC_by_disease_income.csv"), index=False)

ages = d[d.age_name.isin(AGE_BANDS) & (d.year == 2023)]
d4 = (ages.groupby(["cause_name", "age_name", "sex_name"], observed=True)
      .apply(agg_rate, include_groups=False).reset_index())
d4.to_csv(os.path.join(TBL, "D4_age_sex_distribution_2023.csv"), index=False)

c23 = agg[(agg.year == 2023) & (agg.sex_name == "Both")]
piv = c23.pivot_table(index=["location_id", "location_name", "LMIC_group"],
                      columns="cause_name", values="DALY_n", aggfunc="sum")
popc = c23.groupby(["location_id", "location_name", "LMIC_group"])["pop"].mean()
d5 = piv.join(popc).reset_index()
d5.columns = [c.replace(" disorders", "") if c in DISEASES else c for c in d5.columns]
d5["Combined_n"] = d5["Depressive"] + d5["Anxiety"]
for c in ["Depressive", "Anxiety", "Combined"]:
    d5[f"{c}_rate"] = d5[c if c != "Combined" else "Combined_n"] / d5["pop"] * 1e5
d5["GDP_pc"] = d5["location_id"].map(GDPPC)
d5 = d5.sort_values("Combined_rate", ascending=False)
d5.to_csv(os.path.join(TBL, "D5_country_burden_2023.csv"), index=False)

sr = d4.pivot_table(index=["cause_name", "age_name"], columns="sex_name",
                    values="rate", observed=True).reset_index()
sr["F_to_M_ratio"] = sr["Female"] / sr["Male"]
sr.to_csv(os.path.join(TBL, "D6_sex_ratio_by_age_2023.csv"), index=False)

# ============================================================ figures
print("[3/5] Fig1-3 ...")
x_inc = np.arange(len(INCOME))
w = 0.38

# ---- Fig 1: REMOVED (was redundant with Fig 2; deleted per request) -------

# ---- Fig 2: 2x2 trends by income group — INCIDENCE (top) + DALYs (bottom) --
#  Rows: (top) Incidence ASIR, (bottom) DALYs ASDR.  Cols: (a/c) Depression,
#  (b/d) Anxiety.  Each cell = 3 income groups + All-LMIC overall, 95% UI bands.
#  Requires 0-19 incidence at data/incidence_0_19.csv (see README spec). Until
#  that file exists, fall back to a DALYs-only 1x2 (Depression | Anxiety).
INC_FILE = os.path.join(DATA, "incidence_0_19.csv")
HAS_INC = os.path.exists(INC_FILE)
i4 = None  # incidence age x sex (2023), filled below if data present
strata = [(g, ICOL[g], "-", 1.7, .08, ILAB[g]) for g in INCOME] + \
         [("All LMIC", "#000000", "--", 2.4, .05, "All LMIC")]


def trend_panel(ax, src_all, src_inc, cause, ylab, k):
    for grp, col, ls, lw, al, lab in strata:
        if grp == "All LMIC":
            s = src_all[src_all.cause_name == cause].sort_values("year")
        else:
            s = src_inc[(src_inc.cause_name == cause) &
                        (src_inc.LMIC_group == grp)].sort_values("year")
        ax.fill_between(s.year, s.rate_lo, s.rate_hi, color=col, alpha=al)
        ax.plot(s.year, s.rate, color=col, ls=ls, lw=lw, label=lab)
    ax.axvline(2019, ls=":", color="0.5", lw=.9)
    ax.set_xlabel("Year"); ax.set_ylabel(ylab)
    panel(ax, k)


if HAS_INC:
    # build incidence aggregates the same way as DALYs
    inc = pd.read_csv(INC_FILE)
    inc = inc[inc.cause_name.isin(DISEASES) & inc.location_id.isin(lmic_ids)]
    inum = inc[inc.metric_name == "Number"].rename(
        columns={"val": "DALY_n", "lower": "n_lo", "upper": "n_hi"})
    irate = inc[inc.metric_name == "Rate"][["location_id", "sex_name", "age_name",
              "cause_name", "year", "val"]].rename(columns={"val": "DALY_rate"})
    di = inum.merge(irate, on=["location_id", "sex_name", "age_name", "cause_name",
                   "year"], how="left").merge(lmic, on="location_id", how="left")
    di["pop"] = np.where(di.DALY_rate > 0, di.DALY_n / di.DALY_rate * 1e5, np.nan)
    i4 = (di[di.age_name.isin(AGE_BANDS) & (di.year == 2023)]
          .groupby(["cause_name", "age_name", "sex_name"], observed=True)
          .apply(agg_rate, include_groups=False).reset_index())
    ia = di[(di.age_name == "<20 years") & (di.sex_name == "Both")]
    i_all = ia.groupby(["cause_name", "year"], observed=True).apply(
        agg_rate, include_groups=False).reset_index()
    i_inc = ia.groupby(["cause_name", "LMIC_group", "year"], observed=True).apply(
        agg_rate, include_groups=False).reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
    for j, cause in enumerate(DISEASES):
        trend_panel(axes[0, j], i_all, i_inc, cause,
                    f"{DLAB[cause]} incidence rate ({PER})", j)
        trend_panel(axes[1, j], d2_all, d2_inc, cause,
                    f"{DLAB[cause]} DALY rate ({PER})", j + 2)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=7, loc="upper left",
                      title="income group (shaded = 95% UI; dotted = 2019)",
                      title_fontsize=7)
    save(fig, "Fig1_incidence_DALYs_trend_2x2.pdf")
    f1 = pd.concat([
        d2_all.assign(measure="DALYs", group="All LMIC"),
        d2_inc.assign(measure="DALYs").rename(columns={"LMIC_group": "group"}),
        i_all.assign(measure="Incidence", group="All LMIC"),
        i_inc.assign(measure="Incidence").rename(columns={"LMIC_group": "group"}),
    ], ignore_index=True).rename(
        columns={"DALY_n": "number", "n_lo": "number_lo", "n_hi": "number_hi"})
    save_csv(f1[["measure", "cause_name", "group", "year", "number", "number_lo",
                 "number_hi", "rate", "rate_lo", "rate_hi"]],
             "Fig1_incidence_DALYs_trend_2x2.csv")
else:
    print("   [Fig2] incidence_0_19.csv not found -> DALYs-only 1x2 (interim)")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for j, cause in enumerate(DISEASES):
        trend_panel(axes[j], d2_all, d2_inc, cause,
                    f"{DLAB[cause]} DALY rate ({PER})", j)
    axes[0].legend(frameon=False, ncol=2, fontsize=7, loc="upper left",
                   title="income group (shaded = 95% UI; dotted = 2019)",
                   title_fontsize=7)
    save(fig, "Fig1_trend_1990_2023.pdf")

# ---- Fig 3: 2x2 age x sex — INCIDENCE (top) + DALYs (bottom), 2023 --------
#  bars = sex-specific counts (left axis, 95% UI whiskers); lines = sex-specific
#  rates (right axis, 95% UI shaded band). Cols: Depression / Anxiety.
xb = np.arange(len(AGE_BANDS))
SEXCOL = {"Male": "#4C72B0", "Female": "#C44E52"}
SEXLN = {"Male": "#1f3b73", "Female": "#7a1f2b"}


def yerr_n(s):   # asymmetric 95% UI for counts (millions)
    return np.array([(s.DALY_n - s.n_lo) / 1e6, (s.n_hi - s.DALY_n) / 1e6])


def age_sex_panel(ax, data4, cause, count_lab, rate_lab):
    dd = data4[data4.cause_name == cause]
    mm = dd[dd.sex_name == "Male"].set_index("age_name").reindex(AGE_BANDS)
    ff = dd[dd.sex_name == "Female"].set_index("age_name").reindex(AGE_BANDS)
    ekw = dict(capsize=2, error_kw=dict(elinewidth=.7, ecolor="0.35"))
    b1 = ax.bar(xb - w/2, mm.DALY_n/1e6, w, color=SEXCOL["Male"], yerr=yerr_n(mm), **ekw)
    b2 = ax.bar(xb + w/2, ff.DALY_n/1e6, w, color=SEXCOL["Female"], yerr=yerr_n(ff), **ekw)
    ax.set_ylabel(count_lab)
    ax.set_xticks(xb); ax.set_xticklabels([AGE_SHORT[a] for a in AGE_BANDS])
    ax2 = ax.twinx(); ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax2.fill_between(xb, mm.rate_lo, mm.rate_hi, color=SEXLN["Male"], alpha=.15)
    ax2.fill_between(xb, ff.rate_lo, ff.rate_hi, color=SEXLN["Female"], alpha=.15)
    l1, = ax2.plot(xb, mm.rate, color=SEXLN["Male"], marker="o", ms=3.5, lw=1.4)
    l2, = ax2.plot(xb, ff.rate, color=SEXLN["Female"], marker="s", ms=3.5, lw=1.4)
    ax2.set_ylabel(rate_lab)
    return (b1, b2, l1, l2)


if HAS_INC and i4 is not None:
    rows = [("Incidence", i4, "Incident cases (millions)", "incidence rate"),
            ("DALYs", d4, "DALYs (millions)", "DALY rate")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    handles = None
    for ri, (mlab, data4, clab, rlab) in enumerate(rows):
        for ci, cause in enumerate(DISEASES):
            ax = axes[ri, ci]
            H = age_sex_panel(ax, data4, cause, clab, f"{rlab} ({PER})")
            if ri == 1:
                ax.set_xlabel("Age group (years)")
            ax.set_title(f"{DLAB[cause]} — {mlab}", fontsize=9, fontweight="bold")
            panel(ax, ri * 2 + ci)
            if ri == 0 and ci == 0:
                handles = H
    fig.legend(handles, ["Male (count)", "Female (count)", "Male (rate)", "Female (rate)"],
               loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.03))
    save(fig, "Fig2_age_sex_incidence_DALYs_2x2.pdf")
    f2 = pd.concat([i4.assign(measure="Incidence"), d4.assign(measure="DALYs")],
                   ignore_index=True).rename(
        columns={"DALY_n": "number", "n_lo": "number_lo", "n_hi": "number_hi"})
    save_csv(f2[["measure", "cause_name", "age_name", "sex_name", "number",
                 "number_lo", "number_hi", "rate", "rate_lo", "rate_hi"]],
             "Fig2_age_sex_incidence_DALYs_2x2.csv")
else:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    handles = None
    for k, cause in enumerate(DISEASES):
        H = age_sex_panel(axes[k], d4, cause, "DALYs (millions)", f"DALY rate ({PER})")
        axes[k].set_xlabel("Age group (years)")
        axes[k].set_title(DLAB[cause], fontsize=9, fontweight="bold")
        panel(axes[k], k)
        if k == 0:
            handles = H
    fig.legend(handles, ["Male (DALYs)", "Female (DALYs)", "Male (rate)", "Female (rate)"],
               loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.04))
    save(fig, "Fig2_age_sex_2023.pdf")

# ---- Fig 4: per-disease maps (a) DALY rate 2023, (b) EAPC -----------------
print("[4/5] Fig4 maps (level + EAPC) ...")
from matplotlib.colors import TwoSlopeNorm
world = gpd.read_file(os.path.join(DATA, "df_world2.geojson"))
world["location_id"] = pd.to_numeric(world["location_id"], errors="coerce")
# per-country rate 2023 + EAPC (Both, <20)
cb = agg[agg.sex_name == "Both"]
rate23 = cb[cb.year == 2023][["location_id", "cause_name", "DALY_rate",
                             "DALY_rate_lo", "DALY_rate_hi"]]
ec = (cb.sort_values("year").groupby(["location_id", "cause_name"])
      .apply(lambda g: pd.Series(eapc(g.year, g.DALY_rate),
                                 index=["EAPC", "EAPC_lo", "EAPC_hi"]),
             include_groups=False)
      .reset_index())
# 2x2: rows = diseases (Depression, Anxiety); cols = (DALY rate 2023, EAPC)
# Compact layout: short figure (matches the wide world-map aspect) + thin,
# close colorbars + minimal subplot spacing -> little wasted whitespace.
fig, axes = plt.subplots(2, 2, figsize=(15, 5.6))
lab = "abcd"
cbar_rate = dict(shrink=.85, pad=0.008, fraction=0.022, aspect=22)
for i, cause in enumerate(DISEASES):
    r = rate23[rate23.cause_name == cause][["location_id", "DALY_rate"]]
    e = ec[ec.cause_name == cause][["location_id", "EAPC"]]
    # (col 0) DALY rate level
    axr = axes[i, 0]
    world.merge(r, on="location_id", how="left").plot(
        column="DALY_rate", cmap="YlOrRd", legend=True, ax=axr, edgecolor="0.5",
        linewidth=.3, missing_kwds={"color": "white", "edgecolor": "0.75", "linewidth": .15},
        legend_kwds={**cbar_rate, "label": f"DALYs rate ({PER})"})
    axr.set_title(f"({lab[i*2]}) {DLAB[cause]} DALY rate, 2023", fontsize=9)
    # (col 1) EAPC
    axe = axes[i, 1]
    vmax = float(np.nanmax(np.abs(e["EAPC"]))) if e["EAPC"].notna().any() else 1
    world.merge(e, on="location_id", how="left").plot(
        column="EAPC", cmap="RdBu_r", legend=True, ax=axe, edgecolor="0.5",
        linewidth=.3, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        missing_kwds={"color": "white", "edgecolor": "0.75", "linewidth": .15},
        legend_kwds={**cbar_rate, "label": "EAPC (% per year)"})
    axe.set_title(f"({lab[i*2+1]}) {DLAB[cause]} EAPC, 1990-2023", fontsize=9)
for ax in axes.flat:
    ax.set_xlim(-180, 180); ax.set_ylim(-57, 85); ax.margins(0); ax.axis("off")
fig.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.01,
                    wspace=0.02, hspace=0.08)
fig.savefig(os.path.join(FIG, "Fig3_maps_rate_EAPC_2x2.pdf"), bbox_inches="tight")
plt.close(fig)
print("   wrote 结果V1/PartA_描述性流行病学/figures/Fig3_maps_rate_EAPC_2x2.pdf")
f3 = (rate23.rename(columns={"DALY_rate": "DALY_rate_2023",
                             "DALY_rate_lo": "DALY_rate_2023_lo",
                             "DALY_rate_hi": "DALY_rate_2023_hi"})
      .merge(ec, on=["location_id", "cause_name"], how="outer")
      .merge(lmic[["location_id", "location_name"]], on="location_id", how="left"))
save_csv(f3[["cause_name", "location_id", "location_name",
             "DALY_rate_2023", "DALY_rate_2023_lo", "DALY_rate_2023_hi",
             "EAPC", "EAPC_lo", "EAPC_hi"]]
         .sort_values(["cause_name", "DALY_rate_2023"], ascending=[True, False]),
         "Fig3_maps_rate_EAPC_2x2.csv")

# ---- (EAPC-forest & inequality A/B figures removed per request) -----------
# EAPC values remain in D3 and the Fig3 EAPC maps; here we keep the inequality
# summary as a TABLE only (concentration index + SII), no figure.
print("[4/4] Inequality summary table (D7) ...")
g6 = d5[d5["GDP_pc"].notna() & (d5["pop"] > 0)].copy().sort_values("GDP_pc")
rng = np.random.default_rng(42)
B = 500
pop = g6["pop"].values
xx = np.r_[0, np.cumsum(pop) / pop.sum()]
yy = np.r_[0, np.cumsum(g6["Combined_n"].values) / g6["Combined_n"].sum()]
CI = 1 - 2 * np.trapz(yy, xx)                      # <0 => burden on poorer
fr = (np.cumsum(pop) - pop / 2) / pop.sum()
SII = np.polyfit(fr, g6["Combined_rate"].values, 1, w=pop / pop.sum())[0]
sii_bs = []
for _ in range(B):
    s = g6.iloc[rng.integers(0, len(g6), len(g6))].sort_values("GDP_pc")
    p = s["pop"].values
    frb = (np.cumsum(p) - p / 2) / p.sum()
    sii_bs.append(np.polyfit(frb, s["Combined_rate"].values, 1, w=p / p.sum())[0])
sii_lo, sii_hi = np.percentile(sii_bs, [2.5, 97.5])
pd.DataFrame([{"metric": "Concentration_index", "value": CI},
             {"metric": "SII", "value": SII, "lo": sii_lo, "hi": sii_hi}]).to_csv(
    os.path.join(TBL, "D7_inequality_2023.csv"), index=False)
print(f"   Concentration index={CI:.3f} | SII={SII:,.0f} ({sii_lo:,.0f}..{sii_hi:,.0f})")
print("\nDONE. Tables + figures in", os.path.relpath(OUT, ROOT))
