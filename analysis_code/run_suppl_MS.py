#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Appendix figures + tables for the JAMA Pediatrics submission
(0-19y Depressive & Anxiety disorders in LMICs, GBD 2023).

Implements Supplementary_Appendix_JAMA_Pediatrics.md (v2): Fig1-Fig6 are ALL
main-text figures, so NOTHING here duplicates a main figure. Every eFigure/
eTable is an extension result (finer sex/age/income/country breakdowns, country
rankings, COVID-19 uncertainty, source dictionary, sensitivity).

House style follows the Part A descriptive figures (run_descriptive_epi_MDD.py):
  no titles; bold (a)/(b) panel labels; no top/right spines; grid alpha .25;
  frameless legends; 95% UI/CI/PI error bars; TrueType-embedded PDF @300dpi.
  Income: Low #9467bd / Lower-mid #E67E22 / Upper-mid #16A085.
  Disease: Depression #C44E52 / Anxiety #4C72B0.

Outputs -> MS_result/Suppl/{tables,figures}
  tables/   eTable1-3 (copies), eTable4-10 (new)
  figures/  eFigure1-12 (new; each PDF + same-named source CSV)
"""
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MS   = os.path.join(ROOT, "MS_result")
FIGS = os.path.join(MS, "figures")
SUP  = os.path.join(MS, "Suppl")
STAB = os.path.join(SUP, "tables")
SFIG = os.path.join(SUP, "figures")
SENS = os.path.join(ROOT, "results", "results_VLW_MDD")
# fresh start (Suppl holds only generated files)
if os.path.isdir(SUP):
    shutil.rmtree(SUP)
for d in (STAB, SFIG):
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------- style
INCOME = ["Low income", "Lower middle income", "Upper middle income"]
ILAB   = {"Low income": "Low", "Lower middle income": "Lower-mid",
          "Upper middle income": "Upper-mid"}
ICOL   = {"Low income": "#9467bd", "Lower middle income": "#E67E22",
          "Upper middle income": "#16A085"}
AGE_BANDS = ["<5 years", "5-9 years", "10-14 years", "15-19 years"]
AGE_SHORT = {"<5 years": "<5", "5-9 years": "5-9",
             "10-14 years": "10-14", "15-19 years": "15-19"}
DISEASES_L = ["Depressive disorders", "Anxiety disorders"]   # long names
DLAB = {"Depressive disorders": "Depression", "Anxiety disorders": "Anxiety"}
DLONG = {"Depression": "Depressive disorders", "Anxiety": "Anxiety disorders"}
DCOL = {"Depressive disorders": "#C44E52", "Anxiety disorders": "#4C72B0"}
DZ3 = ["Depressive disorders", "Anxiety disorders", "Combined"]
DZ3COL = {"Depressive disorders": "#C44E52", "Anxiety disorders": "#4C72B0",
          "Combined": "#555555"}
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


def panel(ax, k, off=-0.10, fs=13):
    ax.text(off, 1.02, f"({'abcdefgh'[k]})", transform=ax.transAxes,
            fontweight="bold", va="bottom", ha="left", fontsize=fs)


def savef(fig, name):
    fig.savefig(os.path.join(SFIG, name), bbox_inches="tight")
    plt.close(fig)
    print("   wrote figures/" + name)


def fcsv(df, name):
    df.to_csv(os.path.join(SFIG, name), index=False)
    print("   wrote figures/" + name)


def tcsv(df, name):
    df.to_csv(os.path.join(STAB, name), index=False)
    print("   wrote tables/" + name)


set_style()

# --------------------------------------------- MS_result main-table parsers
# "30 962.4 (24 267.1 to 39 420.3)"  (space thousands, ' to ', may be '—'/neg)
def _f(x):
    s = str(x).strip().replace(" ", "").replace(",", "")
    if s in ("—", "-", "nan", "", "None"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def pt(s):
    return _f(str(s).split("(")[0])


def ui(s):
    s = str(s)
    if "(" not in s:
        return np.nan, np.nan
    lo, hi = s.split("(", 1)[1].rstrip(")").split(" to ")
    return _f(lo), _f(hi)


# ----------------------------------------------------------------- load
t1 = pd.read_csv(os.path.join(MS, "Table1_incidence_DALYs.csv"))
t2 = pd.read_csv(os.path.join(MS, "Table2_VLW.csv"))
t3 = pd.read_csv(os.path.join(MS, "Table3_COVID_excess.csv"))
f1 = pd.read_csv(os.path.join(FIGS, "Fig1_incidence_DALYs_trend_2x2.csv"))
f2 = pd.read_csv(os.path.join(FIGS, "Fig2_age_sex_incidence_DALYs_2x2.csv"))
f3 = pd.read_csv(os.path.join(FIGS, "Fig3_maps_rate_EAPC_2x2.csv"))
fig5 = pd.read_csv(os.path.join(FIGS, "Fig5_scatter_GDPpc_VLW_by_disease.csv"))

VLW_COL = "VLW (billion USD)"
VLWGDP_COL = "VLW/GDP (%)"
EXC_COL = "COVID-19 excess VLW 2020–2023 (billion USD)"
EXCGDP_COL = "Excess/GDP (%)"


def err2(p, lo, hi):  # asymmetric, clipped error array for bar/errorbar
    p, lo, hi = np.asarray(p, float), np.asarray(lo, float), np.asarray(hi, float)
    return np.array([np.maximum(p - lo, 0), np.maximum(hi - p, 0)])


# ====================================================== eTable 1-3 (copies)
print("[1/13] eTable 1-3 (full main tables, copied) ...")
for src, dst in [("Table1_incidence_DALYs.csv", "eTable1_Full_incidence_DALYs.csv"),
                 ("Table2_VLW.csv", "eTable2_Full_VLW_2023.csv"),
                 ("Table3_COVID_excess.csv", "eTable3_Full_COVID_excess_VLW.csv")]:
    shutil.copy(os.path.join(MS, src), os.path.join(STAB, dst))
    print("   wrote tables/" + dst)

# ====================================================== sex ratios (T4,F1,F2)
print("[2/13] Sex rate ratios -> eTable4, eFigure1, eFigure2 ...")
sx = f2[f2.sex_name.isin(["Male", "Female"])].copy()
rate_p = sx.pivot_table(index=["measure", "cause_name", "age_name"],
                        columns="sex_name", values="rate").reset_index()
rate_p["F_to_M_ratio"] = rate_p["Female"] / rate_p["Male"]
rate_p["abs_diff"] = rate_p["Female"] - rate_p["Male"]
rate_p["age_name"] = pd.Categorical(rate_p["age_name"], AGE_BANDS, ordered=True)
rate_p = rate_p.sort_values(["measure", "cause_name", "age_name"])
tcsv(rate_p.rename(columns={"Female": "Female_rate", "Male": "Male_rate"})[
     ["cause_name", "measure", "age_name", "Female_rate", "Male_rate",
      "F_to_M_ratio", "abs_diff"]], "eTable4_sex_rate_ratios_2023.csv")

MEAS = ["Incidence", "DALYs"]


def sex_grid(value, ylab, hline, fname, title):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    x = np.arange(len(AGE_BANDS))
    for ri, meas in enumerate(MEAS):
        for ci, cause in enumerate(DISEASES_L):
            ax = axes[ri, ci]
            d = rate_p[(rate_p.measure == meas) & (rate_p.cause_name == cause)]
            d = d.set_index("age_name").reindex(AGE_BANDS)
            ax.bar(x, d[value], 0.6, color=DCOL[cause], alpha=.85)
            if hline is not None:
                ax.axhline(hline, color="0.4", lw=.9, ls="--")
            ax.set_xticks(x); ax.set_xticklabels([AGE_SHORT[a] for a in AGE_BANDS])
            if ri == 1:
                ax.set_xlabel("Age group (years)")
            if ci == 0:
                ax.set_ylabel(ylab)
            ax.set_title(f"{DLAB[cause]} — {meas}", fontsize=9, fontweight="bold")
            ax.grid(axis="x", visible=False)
            panel(ax, ri * 2 + ci)
    savef(fig, fname)


sex_grid("F_to_M_ratio", "Female-to-male rate ratio", 1.0,
         "eFigure1_female_male_rate_ratio.pdf", None)
fcsv(rate_p[["cause_name", "measure", "age_name", "F_to_M_ratio"]],
     "eFigure1_female_male_rate_ratio.csv")
sex_grid("abs_diff", f"Female − male rate ({PER})", 0.0,
         "eFigure2_female_male_rate_difference.pdf", None)
fcsv(rate_p[["cause_name", "measure", "age_name", "abs_diff"]],
     "eFigure2_female_male_rate_difference.csv")

# ====================================================== eFigure3 number trends
print("[3/13] eFigure3 — income-group trends in counts ...")
strata = [(g, ICOL[g], "-", 1.7, .08, ILAB[g]) for g in INCOME] + \
         [("All LMIC", "#000000", "--", 2.4, .05, "All LMIC")]
fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
for ri, meas in enumerate(MEAS):
    for ci, cause in enumerate(DISEASES_L):
        ax = axes[ri, ci]
        for grp, col, ls, lw, al, lab in strata:
            s = f1[(f1.measure == meas) & (f1.cause_name == cause) &
                   (f1.group == grp)].sort_values("year")
            ax.fill_between(s.year, s.number_lo / 1e6, s.number_hi / 1e6,
                            color=col, alpha=al)
            ax.plot(s.year, s.number / 1e6, color=col, ls=ls, lw=lw, label=lab)
        ax.axvline(2019, ls=":", color="0.5", lw=.9)
        if ri == 1:
            ax.set_xlabel("Year")
        unit = "Incident cases" if meas == "Incidence" else "DALYs"
        ax.set_ylabel(f"{DLAB[cause]} {unit} (millions)")
        panel(ax, ri * 2 + ci)
axes[0, 0].legend(frameon=False, ncol=2, fontsize=7, loc="upper left",
                  title="income group (shaded = 95% UI; dotted = 2019)",
                  title_fontsize=7)
savef(fig, "eFigure3_income_group_number_trends.pdf")
fcsv(f1[["measure", "cause_name", "group", "year", "number",
         "number_lo", "number_hi"]], "eFigure3_income_group_number_trends.csv")

# ====================================================== country rank: DALY/EAPC
print("[4/13] eTable5, eFigure4 (DALY rate), eFigure5 (EAPC) ...")
fr = f3.dropna(subset=["location_name"]).copy()


def hbar_by_disease(metric, lo, hi, xlabel, fname, ascending_top=False,
                    zero_line=False, n=20):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.6))
    for k, cause in enumerate(DISEASES_L):
        ax = axes[k]
        d = (fr[fr.cause_name == cause].dropna(subset=[metric])
             .sort_values(metric, ascending=False).head(n).iloc[::-1])
        y = np.arange(len(d))
        ax.barh(y, d[metric], color=DCOL[cause], alpha=.85,
                xerr=err2(d[metric].values, d[lo].values, d[hi].values),
                error_kw=dict(elinewidth=.7, ecolor="0.4", capsize=2))
        if zero_line:
            ax.axvline(0, color="0.4", lw=.9)
        ax.set_yticks(y); ax.set_yticklabels(d.location_name, fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_title(f"({'ab'[k]}) {DLAB[cause]}", loc="left",
                     fontweight="bold", fontsize=12)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    savef(fig, fname)


hbar_by_disease("DALY_rate_2023", "DALY_rate_2023_lo", "DALY_rate_2023_hi",
                f"DALY rate, 2023 ({PER})", "eFigure4_top20_DALY_rate_2023.pdf")
fcsv(fr[["cause_name", "location_name", "DALY_rate_2023",
         "DALY_rate_2023_lo", "DALY_rate_2023_hi"]]
     .sort_values(["cause_name", "DALY_rate_2023"], ascending=[True, False]),
     "eFigure4_top20_DALY_rate_2023.csv")
hbar_by_disease("EAPC", "EAPC_lo", "EAPC_hi", "EAPC, 1990–2023 (% per year)",
                "eFigure5_top20_EAPC_1990_2023.pdf", zero_line=True)
fcsv(fr[["cause_name", "location_name", "EAPC", "EAPC_lo", "EAPC_hi"]]
     .sort_values(["cause_name", "EAPC"], ascending=[True, False]),
     "eFigure5_top20_EAPC_1990_2023.csv")

# eTable5: top-20 DALY rate, top-20 EAPC, and negative-EAPC countries, per disease
rows = []
for cause in DISEASES_L:
    d = fr[fr.cause_name == cause]
    top_rate = d.sort_values("DALY_rate_2023", ascending=False).head(20)
    for i, (_, r) in enumerate(top_rate.iterrows(), 1):
        rows.append([cause, "Top 20 DALY rate 2023", i, r.location_name,
                     round(r.DALY_rate_2023, 1), round(r.DALY_rate_2023_lo, 1),
                     round(r.DALY_rate_2023_hi, 1)])
    top_eapc = d.sort_values("EAPC", ascending=False).head(20)
    for i, (_, r) in enumerate(top_eapc.iterrows(), 1):
        rows.append([cause, "Top 20 EAPC 1990-2023", i, r.location_name,
                     round(r.EAPC, 3), round(r.EAPC_lo, 3), round(r.EAPC_hi, 3)])
    neg = d[d.EAPC < 0].sort_values("EAPC")
    for i, (_, r) in enumerate(neg.iterrows(), 1):
        rows.append([cause, "Negative EAPC (declining)", i, r.location_name,
                     round(r.EAPC, 3), round(r.EAPC_lo, 3), round(r.EAPC_hi, 3)])
tcsv(pd.DataFrame(rows, columns=["Disease", "Ranking", "Rank", "Country",
                                 "Value", "Lower", "Upper"]),
     "eTable5_country_rank_DALY_EAPC.csv")

# ====================================================== country rank: VLW
print("[5/13] eTable6, eFigure6 (VLW), eFigure7 (VLW/GDP) ...")
cc = t2[t2.Type == "Country"].copy()
for col, p in [(VLW_COL, "vlw"), (VLWGDP_COL, "vlwgdp")]:
    cc[p] = cc[col].map(pt)
    cc[[p + "_lo", p + "_hi"]] = cc[col].apply(lambda s: pd.Series(ui(s)))


def hbar_top(df, val, lo, hi, color, xlabel, fname, zero=False, note=None,
             n=20):
    d = df.sort_values(val, ascending=False).head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 7.6))
    y = np.arange(len(d))
    ax.barh(y, d[val], color=color, alpha=.85,
            xerr=err2(d[val].values, d[lo].values, d[hi].values),
            error_kw=dict(elinewidth=.8, ecolor="0.4", capsize=2))
    if zero:
        ax.axvline(0, color="0.4", lw=.9)
    ax.set_yticks(y); ax.set_yticklabels(d.Location, fontsize=8)
    ax.set_xlabel(xlabel); ax.grid(axis="y", visible=False)
    panel(ax, 0, off=-0.34)
    if note:
        ax.text(0.98, 0.04, note, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=7, style="italic", color="0.4")
    savef(fig, fname)
    return d


comb = cc[cc.Disease == "Combined"]
hbar_top(comb, "vlw", "vlw_lo", "vlw_hi", "#3D5296",
         "Combined VLW, 2023 (billion USD, PPP)",
         "eFigure6_top20_VLW_2023.pdf")
fcsv(comb[["Location", "vlw", "vlw_lo", "vlw_hi"]].rename(columns={
     "vlw": "VLW_bn", "vlw_lo": "VLW_bn_lo", "vlw_hi": "VLW_bn_hi"})
     .sort_values("VLW_bn", ascending=False), "eFigure6_top20_VLW_2023.csv")
hbar_top(comb, "vlwgdp", "vlwgdp_lo", "vlwgdp_hi", "#A52878",
         "Combined VLW / GDP, 2023 (%)", "eFigure7_top20_VLW_GDP_2023.pdf")
fcsv(comb[["Location", "vlwgdp", "vlwgdp_lo", "vlwgdp_hi"]].rename(columns={
     "vlwgdp": "VLW_GDP_pct", "vlwgdp_lo": "VLW_GDP_pct_lo",
     "vlwgdp_hi": "VLW_GDP_pct_hi"}).sort_values("VLW_GDP_pct", ascending=False),
     "eFigure7_top20_VLW_GDP_2023.csv")

rows = []
for rk, (val, lo, hi, lab, dz) in enumerate([
        ("vlw", "vlw_lo", "vlw_hi", "Top 20 combined VLW", "Combined"),
        ("vlwgdp", "vlwgdp_lo", "vlwgdp_hi", "Top 20 combined VLW/GDP", "Combined"),
        ("vlw", "vlw_lo", "vlw_hi", "Top 20 VLW", "Depressive disorders"),
        ("vlw", "vlw_lo", "vlw_hi", "Top 20 VLW", "Anxiety disorders")]):
    d = cc[cc.Disease == dz].sort_values(val, ascending=False).head(20)
    unit = "VLW/GDP %" if "GDP" in lab else "VLW bn"
    for i, (_, r) in enumerate(d.iterrows(), 1):
        rows.append([dz, lab, i, r.Location, round(r[val], 3),
                     round(r[lo], 3), round(r[hi], 3), unit])
tcsv(pd.DataFrame(rows, columns=["Disease", "Ranking", "Rank", "Country",
                                 "Value", "Lower", "Upper", "Unit"]),
     "eTable6_country_rank_VLW_VLWGDP.csv")

# ====================================================== country rank: COVID
print("[6/13] eTable7, eFigure9 (excess VLW), eFigure10 (Excess/GDP) ...")
ce = t3[t3.Type == "Country"].copy()
for col, p in [(EXC_COL, "exc"), (EXCGDP_COL, "excgdp")]:
    ce[p] = ce[col].map(pt)
    ce[[p + "_lo", p + "_hi"]] = ce[col].apply(lambda s: pd.Series(ui(s)))
ce_comb = ce[ce.Disease == "Combined"]
PINOTE = "Error bars: 95% PI (wide; may cross 0)"
hbar_top(ce_comb, "exc", "exc_lo", "exc_hi", "#C0392B",
         "Cumulative COVID-19 excess VLW, 2020–2023 (billion USD)",
         "eFigure9_top20_COVID_excess_VLW.pdf", zero=True, note=PINOTE)
fcsv(ce_comb[["Location", "exc", "exc_lo", "exc_hi"]].rename(columns={
     "exc": "excess_bn", "exc_lo": "excess_bn_lo", "exc_hi": "excess_bn_hi"})
     .sort_values("excess_bn", ascending=False),
     "eFigure9_top20_COVID_excess_VLW.csv")
hbar_top(ce_comb, "excgdp", "excgdp_lo", "excgdp_hi", "#C0392B",
         "Cumulative excess VLW / GDP, 2020–2023 (%)",
         "eFigure10_top20_COVID_excess_GDP.pdf", zero=True, note=PINOTE)
fcsv(ce_comb[["Location", "excgdp", "excgdp_lo", "excgdp_hi"]].rename(columns={
     "excgdp": "excess_GDP_pct", "excgdp_lo": "excess_GDP_pct_lo",
     "excgdp_hi": "excess_GDP_pct_hi"}).sort_values("excess_GDP_pct",
     ascending=False), "eFigure10_top20_COVID_excess_GDP.csv")

rows = []
for val, lo, hi, lab in [("exc", "exc_lo", "exc_hi", "Top 20 combined excess VLW"),
                         ("excgdp", "excgdp_lo", "excgdp_hi",
                          "Top 20 combined Excess/GDP")]:
    d = ce_comb.sort_values(val, ascending=False).head(20)
    unit = "Excess/GDP %" if "GDP" in lab else "excess bn"
    for i, (_, r) in enumerate(d.iterrows(), 1):
        crosses = "yes" if (r[lo] < 0 < r[hi]) else "no"
        rows.append([lab, i, r.Location, round(r[val], 3), round(r[lo], 3),
                     round(r[hi], 3), unit, crosses])
tcsv(pd.DataFrame(rows, columns=["Ranking", "Rank", "Country", "Value",
                                 "PI_lower", "PI_upper", "Unit", "PI_crosses_0"]),
     "eTable7_country_rank_COVID_excess.csv")

# ====================================================== income decomposition
print("[7/13] eFigure8 (income VLW), eFigure11 (income COVID) ...")
ig2 = t2[t2.Type == "Income group"].copy()
for col, p in [(VLW_COL, "vlw"), (VLWGDP_COL, "vlwgdp")]:
    ig2[p] = ig2[col].map(pt)
    ig2[[p + "_lo", p + "_hi"]] = ig2[col].apply(lambda s: pd.Series(ui(s)))
ig3 = t3[t3.Type == "Income group"].copy()
for col, p in [(EXC_COL, "exc"), (EXCGDP_COL, "excgdp")]:
    ig3[p] = ig3[col].map(pt)
    ig3[[p + "_lo", p + "_hi"]] = ig3[col].apply(lambda s: pd.Series(ui(s)))


def grouped_income(df, specs, fname, hline0=False, note=None):
    """specs = [(value, lo, hi, ylabel), ...] -> one panel each, grouped by DZ3."""
    fig, axes = plt.subplots(1, len(specs), figsize=(5.5 * len(specs), 4.8))
    if len(specs) == 1:
        axes = [axes]
    x = np.arange(len(INCOME)); w = 0.26
    for pi, (val, lo, hi, ylab) in enumerate(specs):
        ax = axes[pi]
        for i, dz in enumerate(DZ3):
            sub = df[df.Disease == dz].set_index("Location").reindex(INCOME)
            off = (i - 1) * w
            ax.bar(x + off, sub[val], w, color=DZ3COL[dz], label=DLAB.get(dz, dz),
                   yerr=err2(sub[val].values, sub[lo].values, sub[hi].values),
                   error_kw=dict(elinewidth=.7, ecolor="0.45", capsize=1.5))
        if hline0:
            ax.axhline(0, color="0.4", lw=.9)
        ax.set_xticks(x); ax.set_xticklabels([ILAB[g] for g in INCOME])
        ax.set_xlabel("Income group"); ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
        panel(ax, pi)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    if note:
        axes[-1].text(0.98, 0.96, note, transform=axes[-1].transAxes, ha="right",
                      va="top", fontsize=7, style="italic", color="0.4")
    savef(fig, fname)


grouped_income(ig2, [("vlw", "vlw_lo", "vlw_hi", "VLW, 2023 (billion USD)"),
                     ("vlwgdp", "vlwgdp_lo", "vlwgdp_hi", "VLW / GDP (%)")],
               "eFigure8_income_group_VLW_decomposition.pdf")
fcsv(ig2[["Location", "Disease", "vlw", "vlw_lo", "vlw_hi",
          "vlwgdp", "vlwgdp_lo", "vlwgdp_hi"]].rename(columns={
     "Location": "income_group"}), "eFigure8_income_group_VLW_decomposition.csv")
grouped_income(ig3, [("exc", "exc_lo", "exc_hi",
                      "Cumulative excess VLW, 2020–2023 (billion USD)"),
                     ("excgdp", "excgdp_lo", "excgdp_hi", "Excess VLW / GDP (%)")],
               "eFigure11_income_group_COVID_excess.pdf", hline0=True,
               note="95% PI (wide; cross 0)")
fcsv(ig3[["Location", "Disease", "exc", "exc_lo", "exc_hi",
          "excgdp", "excgdp_lo", "excgdp_hi"]].rename(columns={
     "Location": "income_group"}), "eFigure11_income_group_COVID_excess.csv")

# ====================================================== eFigure12 scatter
print("[8/13] eFigure12 — burden growth vs relative economic burden ...")
ch = t1[t1.Type == "Country"][["Location", "Disease", "DALYs: Rate %change"]].copy()
ch = ch.rename(columns={"Location": "Country",
                        "DALYs: Rate %change": "rate_change"})
ch["rate_change"] = pd.to_numeric(ch["rate_change"], errors="coerce")
ch["Disease"] = ch.Disease.map(DLONG)
m = fig5.merge(ch[["Country", "Disease", "rate_change"]],
               on=["Country", "Disease"], how="inner")
print(f"   merged: {len(m)} country-disease rows")
dmax = m["DALYs_n"].max()


def msize(d):
    return 12 + 230 * np.sqrt(d / dmax)


fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for k, cause in enumerate(DISEASES_L):
    ax = axes[k]
    d = m[m.Disease == cause].dropna(subset=["rate_change", "VLW_GDP_pct"])
    for g in INCOME:
        gg = d[d.Income_Group == g]
        ax.scatter(gg.rate_change, gg.VLW_GDP_pct, s=msize(gg.DALYs_n),
                   c=ICOL[g], alpha=.6, edgecolor="white", linewidth=.4,
                   zorder=3, label=ILAB[g])
    ax.axvline(0, color="0.6", lw=.8, ls=":")
    ax.set_xlabel("DALY rate change, 1990–2023 (%)")
    if k == 0:
        ax.set_ylabel("VLW / GDP, 2023 (%)")
    ax.set_title(f"({'ab'[k]}) {DLAB[cause]}", loc="left", fontweight="bold",
                 fontsize=12)
inc_h = [Line2D([0], [0], marker="o", ls="", mfc=ICOL[g], mec="white", ms=8,
                label=ILAB[g]) for g in INCOME]
axes[1].legend(handles=inc_h, title="Income group", frameon=False,
               loc="upper left", fontsize=8, title_fontsize=8)
sz = [1e4, 1e5, 5e5, 2e6]; szl = ["10K", "100K", "500K", "2M"]
sz_h = [Line2D([0], [0], marker="o", ls="", mfc="0.6", mec="white",
               ms=np.sqrt(msize(s)) / 1.6, label=l) for s, l in zip(sz, szl)]
axes[0].legend(handles=sz_h, title="DALYs", frameon=False, loc="upper left",
               fontsize=8, title_fontsize=8, labelspacing=1.3, borderpad=1.0,
               handletextpad=1.2)
savef(fig, "eFigure12_DALY_change_vs_VLW_GDP.pdf")
fcsv(m[["Disease", "Country", "Income_Group", "rate_change", "DALYs_n",
        "VLW_GDP_pct", "VLW_GDP_pct_lo", "VLW_GDP_pct_hi"]].sort_values(
        ["Disease", "rate_change"]), "eFigure12_DALY_change_vs_VLW_GDP.csv")

# ====================================================== eTable8 income summary
print("[9/13] eTable8 — income-group combined summary ...")
b1 = t1[t1.Type == "Income group"].copy()
b1["Disease"] = b1.Disease.map(DLONG)
b1 = b1.rename(columns={"Location": "Income group"})
b1["Incidence_2023_k"] = b1["Incident cases: N_2023(k)"].map(pt)
b1["Incidence_change_pct"] = pd.to_numeric(b1["Incident cases: N %change"],
                                           errors="coerce")
b1["DALYs_2023_k"] = b1["DALYs: N_2023(k)"].map(pt)
b1["DALYs_change_pct"] = pd.to_numeric(b1["DALYs: N %change"], errors="coerce")
b1["DALY_rate_2023"] = b1["DALYs: Rate_2023"].map(pt)
b2 = ig2.rename(columns={"Location": "Income group"})[
    ["Income group", "Disease", "vlw", "vlwgdp"]].rename(
    columns={"vlw": "VLW_bn_2023", "vlwgdp": "VLW_GDP_pct_2023"})
b3 = ig3.rename(columns={"Location": "Income group"})[
    ["Income group", "Disease", "exc", "excgdp"]].rename(
    columns={"exc": "COVID_excess_bn", "excgdp": "COVID_excess_GDP_pct"})
e8 = (b1[["Income group", "Disease", "Incidence_2023_k", "Incidence_change_pct",
          "DALYs_2023_k", "DALYs_change_pct", "DALY_rate_2023"]]
      .merge(b2, on=["Income group", "Disease"], how="left")
      .merge(b3, on=["Income group", "Disease"], how="left"))
e8["_o"] = e8["Income group"].map({g: i for i, g in enumerate(INCOME)})
e8 = e8.sort_values(["_o", "Disease"]).drop(columns="_o").round(3)
tcsv(e8, "eTable8_income_group_summary.csv")

# ====================================================== eTable9 dictionary
print("[10/13] eTable9 — figure source-data dictionary ...")
DEFS = {
    "measure": ("Outcome measure (Incidence or DALYs)", "category"),
    "cause_name": ("GBD cause (Depressive / Anxiety disorders)", "category"),
    "group": ("Aggregation group (income group or All LMIC)", "category"),
    "year": ("Calendar year", "year"),
    "age_name": ("Age band (<5/5-9/10-14/15-19 years)", "category"),
    "sex_name": ("Sex (Male/Female/Both)", "category"),
    "number": ("Count (incident cases or DALYs)", "n"),
    "rate": ("Crude rate per 100,000 (0-19 population)", "per 100,000"),
    "location_id": ("GBD location identifier", "id"),
    "location_name": ("GBD location name", "category"),
    "DALY_rate_2023": ("DALY rate in 2023", "per 100,000"),
    "EAPC": ("Estimated annual percentage change, 1990-2023", "% per year"),
    "Disease": ("Disease (Depressive/Anxiety disorders)", "category"),
    "Country": ("Country name", "category"),
    "Income_Group": ("World Bank income group", "category"),
    "GDP_pc": ("GDP per capita", "USD (PPP)"),
    "DALYs_n": ("DALYs (count)", "n"),
    "VLW_GDP_pct": ("VLW as a share of national GDP", "%"),
    "actual": ("Actual VLW (GBD 2023)", "billion USD"),
    "cf_point": ("Counterfactual VLW (ETS, trained 1990-2019)", "billion USD"),
    "cf_pi_lo": ("Counterfactual VLW 95% PI lower", "billion USD"),
    "cf_pi_hi": ("Counterfactual VLW 95% PI upper", "billion USD"),
    "excess": ("Excess VLW (actual - counterfactual)", "billion USD"),
    "period": ("Period flag (historical vs forecast)", "category"),
}


def describe(col):
    for suf, kind in [("_lo", "95% lower bound"), ("_hi", "95% upper bound")]:
        if col.endswith(suf) and col[:-len(suf)] in DEFS:
            d, u = DEFS[col[:-len(suf)]]
            return f"{d} ({kind})", u
    d, u = DEFS.get(col, (col.replace("_", " "), ""))
    return d, u


rows = []
for fn in sorted(f for f in os.listdir(FIGS) if f.endswith(".csv")):
    for col in pd.read_csv(os.path.join(FIGS, fn), nrows=0).columns:
        desc, unit = describe(col)
        rows.append({"Main figure source file": fn, "Variable": col,
                     "Definition": desc, "Unit": unit})
tcsv(pd.DataFrame(rows), "eTable9_figure_source_dictionary.csv")

# ====================================================== eTable10 sensitivity
print("[11/13] eTable10 — economic valuation sensitivity (epsilon) ...")
EPS = {"0.5": "2disease_IE0.5", "1.0": "2disease_IE1", "1.5": "2disease_IE1.5"}
rows = []
for eps, sub in EPS.items():
    a = pd.read_csv(os.path.join(SENS, sub, "T3_disease_summary_allLMIC_2023.csv"))
    for _, r in a.iterrows():
        rows.append({"Scope": "All LMIC", "Disease": r["Disease"], "epsilon": eps,
                     "VLW (billion USD)": r["VLW (billion USD)"],
                     "VLW discounted (bn USD)": r["VLW discounted (bn USD)"],
                     "VLW/GDP (%)": r["VLW/GDP (%)"]})
    b = pd.read_csv(os.path.join(SENS, sub, "T2_disease_income_summary_2023.csv"))
    for _, r in b.iterrows():
        rows.append({"Scope": r["Income Group"], "Disease": r["Disease"],
                     "epsilon": eps, "VLW (billion USD)": r["VLW (billion USD)"],
                     "VLW discounted (bn USD)": r["VLW discounted (bn USD)"],
                     "VLW/GDP (%)": r["VLW/GDP (%)"]})
e10 = pd.DataFrame(rows)
order = {"All LMIC": 0, "Low income": 1, "Lower middle income": 2,
         "Upper middle income": 3}
e10["_o"] = e10.Scope.map(order)
e10 = e10.sort_values(["_o", "Disease", "epsilon"]).drop(columns="_o")
tcsv(e10, "eTable10_economic_valuation_sensitivity.csv")

# ====================================================== manifest
print("[12/13] writing Suppl/README index ...")
manifest = """# MS_result/Suppl — JAMA Pediatrics Supplementary Appendix outputs

Generated by `code/run_suppl_MS.py` from `MS_result/` (tables + figure CSVs)
and `results/results_VLW_MDD/` (epsilon sensitivity). Fig1-Fig6 are MAIN-text
figures and are NOT duplicated here; every item below is an extension result.

## tables/
- eTable1  Full country-level incidence & DALYs, 1990 & 2023 (copy)
- eTable2  Country-level economic burden (VLW), 2023 (copy)
- eTable3  Country-level COVID-19 excess VLW, 2020-2023 (copy)
- eTable4  Female-to-male rate ratios by age, disease, measure, 2023
- eTable5  Country ranking by DALY rate (top 20) and EAPC (top 20 / negative)
- eTable6  Country ranking by VLW and VLW/GDP, 2023
- eTable7  Country ranking by COVID-19 excess VLW and Excess/GDP (PI-crosses-0 flag)
- eTable8  Income-group combined disease + economic burden summary
- eTable9  Main-figure source-data dictionary
- eTable10 Economic valuation sensitivity (epsilon = 0.5 / 1.0 / 1.5)

## figures/  (each PDF has a same-named source CSV)
- eFigure1  Female-to-male rate ratios by age, disease, measure
- eFigure2  Absolute female-male rate differences by age
- eFigure3  Income-group trends in incident cases and DALYs (counts), 1990-2023
- eFigure4  Top 20 countries by DALY rate, 2023 (by disease)
- eFigure5  Top 20 countries by EAPC, 1990-2023 (by disease)
- eFigure6  Top 20 countries by combined VLW, 2023
- eFigure7  Top 20 countries by combined VLW/GDP, 2023
- eFigure8  Income-group decomposition of VLW and VLW/GDP, 2023
- eFigure9  Top 20 countries by COVID-19 excess VLW, 2020-2023
- eFigure10 Top 20 countries by COVID-19 Excess/GDP, 2020-2023
- eFigure11 Income-group decomposition of COVID-19 excess VLW and Excess/GDP
- eFigure12 Country-level DALY-rate growth vs relative economic burden (by disease)
"""
with open(os.path.join(SUP, "README.md"), "w") as fh:
    fh.write(manifest)
print("   wrote Suppl/README.md")

print("[13/13] DONE. Outputs in", os.path.relpath(SUP, ROOT))
