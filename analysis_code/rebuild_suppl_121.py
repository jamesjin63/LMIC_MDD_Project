#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild MS_result/Suppl figures (eFigure 1-12) and the two consolidated
long-format tables (eTable1, eTable2) at the corrected 121-country economic
sample (Somalia + Viet Nam added). Reads the updated MS_result tables/figure
CSVs and the re-run epsilon sensitivity (results/results_VLW_MDD, now 121).

Does NOT delete README/SI/data — only overwrites Suppl/figures/*.pdf+csv and
Suppl/tables/eTable1*, eTable2*. Matches the user's reorganized Suppl schema.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MS   = os.path.join(ROOT, "MS_result")
FIGS = os.path.join(MS, "figures")
SUP  = os.path.join(MS, "Suppl")
STAB = os.path.join(SUP, "tables")
SFIG = os.path.join(SUP, "figures")
SENS = os.path.join(ROOT, "results", "results_VLW_MDD")
for d in (STAB, SFIG):
    os.makedirs(d, exist_ok=True)

INCOME = ["Low income", "Lower middle income", "Upper middle income"]
ILAB = {"Low income": "Low", "Lower middle income": "Lower-mid",
        "Upper middle income": "Upper-mid"}
ICOL = {"Low income": "#9467bd", "Lower middle income": "#E67E22",
        "Upper middle income": "#16A085"}
AGE_BANDS = ["<5 years", "5-9 years", "10-14 years", "15-19 years"]
AGE_SHORT = {"<5 years": "<5", "5-9 years": "5-9", "10-14 years": "10-14",
             "15-19 years": "15-19"}
DISEASES_L = ["Depressive disorders", "Anxiety disorders"]
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
        "ps.fonttype": 42, "axes.linewidth": .8, "lines.linewidth": 1.6})


def panel(ax, k, off=-0.10, fs=13):
    ax.text(off, 1.02, f"({'abcdefgh'[k]})", transform=ax.transAxes,
            fontweight="bold", va="bottom", ha="left", fontsize=fs)


def savef(fig, name):
    fig.savefig(os.path.join(SFIG, name), bbox_inches="tight")
    plt.close(fig); print("   fig", name)


def fcsv(df, name):
    df.to_csv(os.path.join(SFIG, name), index=False)


set_style()

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


def err2(p, lo, hi):
    p, lo, hi = np.asarray(p, float), np.asarray(lo, float), np.asarray(hi, float)
    return np.array([np.maximum(p - lo, 0), np.maximum(hi - p, 0)])


# ----------------------------------------------------------------- load
t1 = pd.read_csv(os.path.join(MS, "Table1_incidence_DALYs.csv"))
t2 = pd.read_csv(os.path.join(MS, "Table2_VLW.csv"))
t3 = pd.read_csv(os.path.join(MS, "Table3_COVID_excess.csv"))
f1 = pd.read_csv(os.path.join(FIGS, "Fig1_incidence_DALYs_trend_2x2.csv"))
f2 = pd.read_csv(os.path.join(FIGS, "Fig2_age_sex_incidence_DALYs_2x2.csv"))
f3 = pd.read_csv(os.path.join(FIGS, "Fig3_maps_rate_EAPC_2x2.csv"))
fig5 = pd.read_csv(os.path.join(FIGS, "Fig5_scatter_GDPpc_VLW_by_disease.csv"))
VLW_COL, VLWGDP_COL = "VLW (billion USD)", "VLW/GDP (%)"
EXC_COL = "COVID-19 excess VLW 2020–2023 (billion USD)"
EXCGDP_COL = "Excess/GDP (%)"

# =====================================================================
# FIGURES eFigure 1-12
# =====================================================================
print("[figures] eFigure 1-12 ...")
# --- sex ratios (eFig1/2) ---
sx = f2[f2.sex_name.isin(["Male", "Female"])].copy()
rate_p = sx.pivot_table(index=["measure", "cause_name", "age_name"],
                        columns="sex_name", values="rate").reset_index()
rate_p["F_to_M_ratio"] = rate_p["Female"] / rate_p["Male"]
rate_p["abs_diff"] = rate_p["Female"] - rate_p["Male"]
rate_p["age_name"] = pd.Categorical(rate_p["age_name"], AGE_BANDS, ordered=True)
rate_p = rate_p.sort_values(["measure", "cause_name", "age_name"])
MEAS = ["Incidence", "DALYs"]


def sex_grid(value, ylab, hline, fname):
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
         "eFigure1_female_male_rate_ratio.pdf")
fcsv(rate_p[["cause_name", "measure", "age_name", "F_to_M_ratio"]],
     "eFigure1_female_male_rate_ratio.csv")
sex_grid("abs_diff", f"Female − male rate ({PER})", 0.0,
         "eFigure2_female_male_rate_difference.pdf")
fcsv(rate_p[["cause_name", "measure", "age_name", "abs_diff"]],
     "eFigure2_female_male_rate_difference.csv")

# --- eFig3 income number trends ---
strata = [(g, ICOL[g], "-", 1.7, .08, ILAB[g]) for g in INCOME] + \
         [("All LMIC", "#000000", "--", 2.4, .05, "All LMIC")]
fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
for ri, meas in enumerate(MEAS):
    for ci, cause in enumerate(DISEASES_L):
        ax = axes[ri, ci]
        for grp, col, ls, lw, al, lab in strata:
            s = f1[(f1.measure == meas) & (f1.cause_name == cause) &
                   (f1.group == grp)].sort_values("year")
            ax.fill_between(s.year, s.number_lo / 1e6, s.number_hi / 1e6, color=col, alpha=al)
            ax.plot(s.year, s.number / 1e6, color=col, ls=ls, lw=lw, label=lab)
        ax.axvline(2019, ls=":", color="0.5", lw=.9)
        if ri == 1:
            ax.set_xlabel("Year")
        unit = "Incident cases" if meas == "Incidence" else "DALYs"
        ax.set_ylabel(f"{DLAB[cause]} {unit} (millions)")
        panel(ax, ri * 2 + ci)
axes[0, 0].legend(frameon=False, ncol=2, fontsize=7, loc="upper left",
                  title="income group (shaded = 95% UI; dotted = 2019)", title_fontsize=7)
savef(fig, "eFigure3_income_group_number_trends.pdf")
fcsv(f1[["measure", "cause_name", "group", "year", "number", "number_lo", "number_hi"]],
     "eFigure3_income_group_number_trends.csv")

# --- eFig4/5 country rank DALY rate / EAPC ---
fr = f3.dropna(subset=["location_name"]).copy()


def hbar_by_disease(metric, lo, hi, xlabel, fname, zero_line=False, n=20):
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
        ax.set_title(f"({'ab'[k]}) {DLAB[cause]}", loc="left", fontweight="bold", fontsize=12)
        ax.grid(axis="y", visible=False)
    fig.tight_layout(); savef(fig, fname)


hbar_by_disease("DALY_rate_2023", "DALY_rate_2023_lo", "DALY_rate_2023_hi",
                f"DALY rate, 2023 ({PER})", "eFigure4_top20_DALY_rate_2023.pdf")
fcsv(fr[["cause_name", "location_name", "DALY_rate_2023", "DALY_rate_2023_lo",
         "DALY_rate_2023_hi"]].sort_values(["cause_name", "DALY_rate_2023"],
         ascending=[True, False]), "eFigure4_top20_DALY_rate_2023.csv")
hbar_by_disease("EAPC", "EAPC_lo", "EAPC_hi", "EAPC, 1990–2023 (% per year)",
                "eFigure5_top20_EAPC_1990_2023.pdf", zero_line=True)
fcsv(fr[["cause_name", "location_name", "EAPC", "EAPC_lo", "EAPC_hi"]].sort_values(
     ["cause_name", "EAPC"], ascending=[True, False]), "eFigure5_top20_EAPC_1990_2023.csv")

# --- economic country tables parsed (121) ---
cc = t2[t2.Type == "Country"].copy()
for col, p in [(VLW_COL, "vlw"), (VLWGDP_COL, "vlwgdp")]:
    cc[p] = cc[col].map(pt)
    cc[[p + "_lo", p + "_hi"]] = cc[col].apply(lambda s: pd.Series(ui(s)))
ce = t3[t3.Type == "Country"].copy()
for col, p in [(EXC_COL, "exc"), (EXCGDP_COL, "excgdp")]:
    ce[p] = ce[col].map(pt)
    ce[[p + "_lo", p + "_hi"]] = ce[col].apply(lambda s: pd.Series(ui(s)))
comb = cc[cc.Disease == "Combined"]
ce_comb = ce[ce.Disease == "Combined"]


def hbar_top(df, val, lo, hi, color, xlabel, fname, zero=False, note=None, n=20):
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
        ax.text(0.98, 0.04, note, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, style="italic", color="0.4")
    savef(fig, fname)


hbar_top(comb, "vlw", "vlw_lo", "vlw_hi", "#3D5296",
         "Combined VLW, 2023 (billion USD, PPP)", "eFigure6_top20_VLW_2023.pdf")
fcsv(comb[["Location", "vlw", "vlw_lo", "vlw_hi"]].rename(columns={
     "vlw": "VLW_bn", "vlw_lo": "VLW_bn_lo", "vlw_hi": "VLW_bn_hi"}).sort_values(
     "VLW_bn", ascending=False), "eFigure6_top20_VLW_2023.csv")
hbar_top(comb, "vlwgdp", "vlwgdp_lo", "vlwgdp_hi", "#A52878",
         "Combined VLW / GDP, 2023 (%)", "eFigure7_top20_VLW_GDP_2023.pdf")
fcsv(comb[["Location", "vlwgdp", "vlwgdp_lo", "vlwgdp_hi"]].rename(columns={
     "vlwgdp": "VLW_GDP_pct", "vlwgdp_lo": "VLW_GDP_pct_lo", "vlwgdp_hi": "VLW_GDP_pct_hi"})
     .sort_values("VLW_GDP_pct", ascending=False), "eFigure7_top20_VLW_GDP_2023.csv")

# --- eFig8 income VLW decomposition ---
ig2 = t2[t2.Type == "Income group"].copy()
for col, p in [(VLW_COL, "vlw"), (VLWGDP_COL, "vlwgdp")]:
    ig2[p] = ig2[col].map(pt)
    ig2[[p + "_lo", p + "_hi"]] = ig2[col].apply(lambda s: pd.Series(ui(s)))
ig3 = t3[t3.Type == "Income group"].copy()
for col, p in [(EXC_COL, "exc"), (EXCGDP_COL, "excgdp")]:
    ig3[p] = ig3[col].map(pt)
    ig3[[p + "_lo", p + "_hi"]] = ig3[col].apply(lambda s: pd.Series(ui(s)))


def grouped_income(df, specs, fname, hline0=False, note=None):
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
        ax.set_xlabel("Income group"); ax.set_ylabel(ylab); ax.grid(axis="x", visible=False)
        panel(ax, pi)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    if note:
        axes[-1].text(0.98, 0.96, note, transform=axes[-1].transAxes, ha="right",
                      va="top", fontsize=7, style="italic", color="0.4")
    savef(fig, fname)


grouped_income(ig2, [("vlw", "vlw_lo", "vlw_hi", "VLW, 2023 (billion USD)"),
                     ("vlwgdp", "vlwgdp_lo", "vlwgdp_hi", "VLW / GDP (%)")],
               "eFigure8_income_group_VLW_decomposition.pdf")
fcsv(ig2[["Location", "Disease", "vlw", "vlw_lo", "vlw_hi", "vlwgdp", "vlwgdp_lo",
          "vlwgdp_hi"]].rename(columns={"Location": "income_group"}),
     "eFigure8_income_group_VLW_decomposition.csv")

# --- eFig9/10 COVID rankings ---
PINOTE = "Error bars: 95% PI (wide; may cross 0)"
hbar_top(ce_comb, "exc", "exc_lo", "exc_hi", "#C0392B",
         "Cumulative COVID-19 excess VLW, 2020–2023 (billion USD)",
         "eFigure9_top20_COVID_excess_VLW.pdf", zero=True, note=PINOTE)
fcsv(ce_comb[["Location", "exc", "exc_lo", "exc_hi"]].rename(columns={
     "exc": "excess_bn", "exc_lo": "excess_bn_lo", "exc_hi": "excess_bn_hi"}).sort_values(
     "excess_bn", ascending=False), "eFigure9_top20_COVID_excess_VLW.csv")
hbar_top(ce_comb, "excgdp", "excgdp_lo", "excgdp_hi", "#C0392B",
         "Cumulative excess VLW / GDP, 2020–2023 (%)",
         "eFigure10_top20_COVID_excess_GDP.pdf", zero=True, note=PINOTE)
fcsv(ce_comb[["Location", "excgdp", "excgdp_lo", "excgdp_hi"]].rename(columns={
     "excgdp": "excess_GDP_pct", "excgdp_lo": "excess_GDP_pct_lo",
     "excgdp_hi": "excess_GDP_pct_hi"}).sort_values("excess_GDP_pct", ascending=False),
     "eFigure10_top20_COVID_excess_GDP.csv")

# --- eFig11 income COVID ---
grouped_income(ig3, [("exc", "exc_lo", "exc_hi",
                      "Cumulative excess VLW, 2020–2023 (billion USD)"),
                     ("excgdp", "excgdp_lo", "excgdp_hi", "Excess VLW / GDP (%)")],
               "eFigure11_income_group_COVID_excess.pdf", hline0=True,
               note="95% PI (wide; cross 0)")
fcsv(ig3[["Location", "Disease", "exc", "exc_lo", "exc_hi", "excgdp", "excgdp_lo",
          "excgdp_hi"]].rename(columns={"Location": "income_group"}),
     "eFigure11_income_group_COVID_excess.csv")

# --- eFig12 scatter (121) ---
ch = t1[t1.Type == "Country"][["Location", "Disease", "DALYs: Rate %change"]].copy()
ch = ch.rename(columns={"Location": "Country", "DALYs: Rate %change": "rate_change"})
ch["rate_change"] = pd.to_numeric(ch["rate_change"], errors="coerce")
ch["Disease"] = ch.Disease.map(DLONG)
m = fig5.merge(ch[["Country", "Disease", "rate_change"]], on=["Country", "Disease"], how="inner")
dmax = m["DALYs_n"].max()
msize = lambda d: 12 + 230 * np.sqrt(d / dmax)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for k, cause in enumerate(DISEASES_L):
    ax = axes[k]
    d = m[m.Disease == cause].dropna(subset=["rate_change", "VLW_GDP_pct"])
    for g in INCOME:
        gg = d[d.Income_Group == g]
        ax.scatter(gg.rate_change, gg.VLW_GDP_pct, s=msize(gg.DALYs_n), c=ICOL[g],
                   alpha=.6, edgecolor="white", linewidth=.4, zorder=3, label=ILAB[g])
    ax.axvline(0, color="0.6", lw=.8, ls=":")
    ax.set_xlabel("DALY rate change, 1990–2023 (%)")
    if k == 0:
        ax.set_ylabel("VLW / GDP, 2023 (%)")
    ax.set_title(f"({'ab'[k]}) {DLAB[cause]}", loc="left", fontweight="bold", fontsize=12)
inc_h = [Line2D([0], [0], marker="o", ls="", mfc=ICOL[g], mec="white", ms=8, label=ILAB[g])
         for g in INCOME]
axes[1].legend(handles=inc_h, title="Income group", frameon=False, loc="upper left",
               fontsize=8, title_fontsize=8)
sz = [1e4, 1e5, 5e5, 2e6]; szl = ["10K", "100K", "500K", "2M"]
sz_h = [Line2D([0], [0], marker="o", ls="", mfc="0.6", mec="white",
               ms=np.sqrt(msize(s)) / 1.6, label=l) for s, l in zip(sz, szl)]
axes[0].legend(handles=sz_h, title="DALYs", frameon=False, loc="upper left", fontsize=8,
               title_fontsize=8, labelspacing=1.3, borderpad=1.0, handletextpad=1.2)
savef(fig, "eFigure12_DALY_change_vs_VLW_GDP.pdf")
fcsv(m[["Disease", "Country", "Income_Group", "rate_change", "DALYs_n", "VLW_GDP_pct",
        "VLW_GDP_pct_lo", "VLW_GDP_pct_hi"]].sort_values(["Disease", "rate_change"]),
     "eFigure12_DALY_change_vs_VLW_GDP.csv")

# =====================================================================
# eTable 1 (long): sex differences + income summary + sensitivity
# =====================================================================
print("[tables] eTable1 ...")
E1 = ["Section", "Subsection", "Disease", "Measure", "Age group", "Income group",
      "Scope", "Parameter", "Metric", "Estimate", "Lower", "Upper", "Unit", "Notes"]
rows = []
# sex differences
for _, r in rate_p.iterrows():
    base = dict(Disease=r.cause_name, Measure=r.measure, **{"Age group": str(r.age_name)},
                **{"Income group": np.nan}, Scope="All LMIC", Parameter=np.nan,
                Lower=np.nan, Upper=np.nan)
    rows.append({**base, "Section": "Sex differences",
                 "Subsection": "Female-to-male rate ratio",
                 "Metric": "Female-to-male rate ratio", "Estimate": r.F_to_M_ratio,
                 "Unit": "ratio", "Notes": "Calculated as female rate divided by male rate."})
    rows.append({**base, "Section": "Sex differences",
                 "Subsection": "Absolute rate difference",
                 "Metric": "Female minus male rate", "Estimate": r.abs_diff,
                 "Unit": "per 100000", "Notes": "Calculated as female rate minus male rate."})
# income-group summary (121 economic; descriptive burden unchanged)
b1 = t1[t1.Type == "Income group"].copy()
b1["Disease"] = b1.Disease.map(DLONG)
b1 = b1.rename(columns={"Location": "IG"})
b1m = {"Incidence_2023_k": ("Incident cases: N_2023(k)", "thousand cases"),
       "Incidence_change_pct": ("Incident cases: N %change", "%"),
       "DALYs_2023_k": ("DALYs: N_2023(k)", "thousand DALYs"),
       "DALYs_change_pct": ("DALYs: N %change", "%"),
       "DALY_rate_2023": ("DALYs: Rate_2023", "per 100000")}
b2 = ig2.rename(columns={"Location": "IG"})
b3 = ig3.rename(columns={"Location": "IG"})
ig_note = "Income-group summary derived from main MS_result tables."
for ig in INCOME:
    for dz in DISEASES_L:
        base = dict(Section="Income group summary",
                    Subsection="Disease burden and economic burden", Disease=dz,
                    Measure=np.nan, **{"Age group": "0-19 years"},
                    **{"Income group": ig}, Scope=ig, Parameter=np.nan,
                    Lower=np.nan, Upper=np.nan)
        r1 = b1[(b1.IG == ig) & (b1.Disease == dz)]
        for metric, (col, unit) in b1m.items():
            val = pt(r1[col].iloc[0]) if len(r1) else np.nan
            rows.append({**base, "Metric": metric, "Estimate": round(val, 3),
                         "Unit": unit, "Notes": ig_note})
        r2 = b2[(b2.IG == ig) & (b2.Disease == dz)]
        rows.append({**base, "Metric": "VLW_bn_2023",
                     "Estimate": round(r2["vlw"].iloc[0], 3) if len(r2) else np.nan,
                     "Unit": "billion USD", "Notes": ig_note})
        rows.append({**base, "Metric": "VLW_GDP_pct_2023",
                     "Estimate": round(r2["vlwgdp"].iloc[0], 3) if len(r2) else np.nan,
                     "Unit": "% GDP", "Notes": ig_note})
        r3 = b3[(b3.IG == ig) & (b3.Disease == dz)]
        rows.append({**base, "Metric": "COVID_excess_bn",
                     "Estimate": round(r3["exc"].iloc[0], 3) if len(r3) else np.nan,
                     "Unit": "billion USD", "Notes": ig_note})
        rows.append({**base, "Metric": "COVID_excess_GDP_pct",
                     "Estimate": round(r3["excgdp"].iloc[0], 3) if len(r3) else np.nan,
                     "Unit": "% GDP", "Notes": ig_note})
# economic valuation sensitivity (epsilon = 0.5/1.0/1.5), 121 countries
EPS = {"epsilon=0.5": "2disease_IE0.5", "epsilon=1.0": "2disease_IE1",
       "epsilon=1.5": "2disease_IE1.5"}
sens_metrics = [("VLW (billion USD)", "billion USD"),
                ("VLW discounted (bn USD)", "billion USD"), ("VLW/GDP (%)", "% GDP")]
snote = "Sensitivity analysis for value of lost welfare estimation."
for param, sub in EPS.items():
    a = pd.read_csv(os.path.join(SENS, sub, "T3_disease_summary_allLMIC_2023.csv"))
    for _, r in a.iterrows():
        for col, unit in sens_metrics:
            rows.append(dict(Section="Economic valuation sensitivity",
                             Subsection="Income elasticity epsilon", Disease=r["Disease"],
                             Measure="VLW", **{"Age group": "0-19 years"},
                             **{"Income group": np.nan}, Scope="All LMIC", Parameter=param,
                             Metric=col, Estimate=r[col], Lower=np.nan, Upper=np.nan,
                             Unit=unit, Notes=snote))
    b = pd.read_csv(os.path.join(SENS, sub, "T2_disease_income_summary_2023.csv"))
    for _, r in b.iterrows():
        for col, unit in sens_metrics:
            rows.append(dict(Section="Economic valuation sensitivity",
                             Subsection="Income elasticity epsilon", Disease=r["Disease"],
                             Measure="VLW", **{"Age group": "0-19 years"},
                             **{"Income group": r["Income Group"]}, Scope=r["Income Group"],
                             Parameter=param, Metric=col, Estimate=r[col],
                             Lower=np.nan, Upper=np.nan, Unit=unit, Notes=snote))
e1 = pd.DataFrame(rows)[E1]
e1.to_csv(os.path.join(STAB, "eTable1_extended_group_sex_sensitivity_results.csv"), index=False)
print("   eTable1 rows:", len(e1))

# =====================================================================
# eTable 2 (long): country rankings
# =====================================================================
print("[tables] eTable2 ...")
E2 = ["Section", "Ranking", "Disease", "Rank", "Country", "Metric", "Estimate",
      "Lower", "Upper", "Unit", "Uncertainty type", "Notes"]
rows = []
DUNC = "95% UI for DALY rate; 95% CI for EAPC"
DUNIT = "per 100000 or % per year"
for cause in DISEASES_L:
    d = fr[fr.cause_name == cause]
    for rk, (mcol, lo, hi, nd) in [("Top 20 DALY rate 2023",
            ("DALY_rate_2023", "DALY_rate_2023_lo", "DALY_rate_2023_hi", 1)),
            ("Top 20 EAPC 1990-2023", ("EAPC", "EAPC_lo", "EAPC_hi", 3))]:
        col, l, h, nd = mcol[0] if isinstance(mcol, tuple) else mcol, lo, hi, nd
        sub = d.sort_values(col, ascending=False).head(20)
        for i, (_, r) in enumerate(sub.iterrows(), 1):
            rows.append(dict(Section="Disease burden country ranking", Ranking=rk,
                             Disease=cause, Rank=i, Country=r.location_name, Metric=rk,
                             Estimate=round(r[col], nd), Lower=round(r[l], nd),
                             Upper=round(r[h], nd), Unit=DUNIT, **{"Uncertainty type": DUNC},
                             Notes=np.nan))
    neg = d[d.EAPC < 0].sort_values("EAPC")
    for i, (_, r) in enumerate(neg.iterrows(), 1):
        rows.append(dict(Section="Disease burden country ranking",
                         Ranking="Negative EAPC (declining)", Disease=cause, Rank=i,
                         Country=r.location_name, Metric="Negative EAPC (declining)",
                         Estimate=round(r.EAPC, 3), Lower=round(r.EAPC_lo, 3),
                         Upper=round(r.EAPC_hi, 3), Unit=DUNIT,
                         **{"Uncertainty type": DUNC}, Notes=np.nan))
# economic rankings
for rk, dz, val, lo, hi, unit in [
        ("Top 20 combined VLW", "Combined", "vlw", "vlw_lo", "vlw_hi", "VLW bn"),
        ("Top 20 combined VLW/GDP", "Combined", "vlwgdp", "vlwgdp_lo", "vlwgdp_hi", "VLW/GDP %"),
        ("Top 20 VLW", "Depressive disorders", "vlw", "vlw_lo", "vlw_hi", "VLW bn"),
        ("Top 20 VLW", "Anxiety disorders", "vlw", "vlw_lo", "vlw_hi", "VLW bn")]:
    sub = cc[cc.Disease == dz].sort_values(val, ascending=False).head(20)
    for i, (_, r) in enumerate(sub.iterrows(), 1):
        rows.append(dict(Section="Economic burden country ranking", Ranking=rk,
                         Disease=dz, Rank=i, Country=r.Location, Metric=rk,
                         Estimate=round(r[val], 3), Lower=round(r[lo], 3),
                         Upper=round(r[hi], 3), Unit=unit, **{"Uncertainty type": "95% UI"},
                         Notes=np.nan))
# COVID rankings
for rk, val, lo, hi, unit in [
        ("Top 20 combined excess VLW", "exc", "exc_lo", "exc_hi", "excess bn"),
        ("Top 20 combined Excess/GDP", "excgdp", "excgdp_lo", "excgdp_hi", "Excess/GDP %")]:
    sub = ce_comb.sort_values(val, ascending=False).head(20)
    for i, (_, r) in enumerate(sub.iterrows(), 1):
        crosses = "yes" if (r[lo] < 0 < r[hi]) else "no"
        rows.append(dict(Section="COVID-19 excess burden country ranking", Ranking=rk,
                         Disease="Combined", Rank=i, Country=r.Location, Metric=rk,
                         Estimate=round(r[val], 3), Lower=round(r[lo], 3),
                         Upper=round(r[hi], 3), Unit=unit, **{"Uncertainty type": "95% PI"},
                         Notes=f"PI crosses 0: {crosses}"))
e2 = pd.DataFrame(rows)[E2]
e2.to_csv(os.path.join(STAB, "eTable2_country_rankings.csv"), index=False)
print("   eTable2 rows:", len(e2))
print("DONE — Suppl figures + eTable1/2 rebuilt at 121.")
