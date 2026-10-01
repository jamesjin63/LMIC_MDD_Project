#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Part A summary tables, learning the NTD Table1 layout
(NTD/344.../final_main_figures/Table1_incidence_DALYs.csv).

Table1_incidence_DALYs.csv — descriptive burden, 0-19y, Both sexes:
  per Location (All LMIC overall / income group / country) x Disease,
  Incident cases & DALYs, each as N_1990(k), N_2023(k), N %change,
  Rate_1990, Rate_2023, Rate %change (crude rate per 100,000; 95% UI).
Table2_VLW.csv — economic burden (VLW, IE=1.0, 0-19y, Both, 2023):
  per Location x Disease (Depression / Anxiety / Combined),
  GDP per capita, DALYs, VLW (bn USD), VLW discounted, VLW/GDP (%); 95% UI.

Both span the total region (All LMIC) and each country (+ income groups).
Note: incidence of "Depressive disorders" is GBD MDD (cause 568); DALYs use
Depressive disorders (567) — consistent with the rest of Part A.

Outputs (结果V1/PartA_描述性流行病学/):
  Table1_incidence_DALYs.csv   Table2_VLW.csv
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "结果V1", "PartA_描述性流行病学")
T1V = os.path.join(OUT, "..", "PartB_经济负担", "VLW_2disease_IE1.0",
                   "T1_country_disease_2023.csv")
CFDIR = os.path.join(OUT, "..", "PartB_经济负担", "COVID_counterfactual_IE1.0")

DISEASES = ["Depressive disorders", "Anxiety disorders"]
DLAB = {"Depressive disorders": "Depression", "Anxiety disorders": "Anxiety"}
INCOME = ["Low income", "Lower middle income", "Upper middle income"]


def sp(x, dec):
    """thousands-separated by space (NTD style); NaN -> '—'."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:,.{dec}f}".replace(",", " ")


def ui(p, l, h, dec, scale=1.0):
    return f"{sp(p/scale, dec)} ({sp(l/scale, dec)} to {sp(h/scale, dec)})"


def pct(a, b):
    return (b - a) / a * 100 if (a and np.isfinite(a) and a != 0) else np.nan


def pt(s):  # point estimate before " ("
    return float(str(s).split("(")[0].strip().replace(",", ""))


def bnd(s):  # (lo, hi) from "pt (lo–hi)"
    inside = str(s).split("(", 1)[1].rstrip(")").replace(",", "")
    lo, hi = inside.split("–")
    return float(lo), float(hi)


# ============================================================ load & classify
lmic = pd.read_csv(os.path.join(DATA, "204_with_LMIC.csv"))
lmic = lmic[lmic["LMIC"] == 1][["location_id", "location_name", "LMIC_group"]]
grp = dict(zip(lmic.location_id, lmic.LMIC_group))
lmic_ids = set(lmic.location_id)


def load_measure(path, measure_filter=None):
    """return long df: location_id, cause_name, year, num/num_lo/num_hi,
    rate/rate_lo/rate_hi (Both, age <20, LMIC only)."""
    df = pd.read_csv(path)
    if measure_filter:
        df = df[df["measure_name"] == measure_filter]
    df = df[(df.sex_name == "Both") & (df.age_name == "<20 years")
            & df.location_id.isin(lmic_ids) & df.cause_name.isin(DISEASES)]
    num = df[df.metric_name == "Number"].rename(
        columns={"val": "num", "lower": "num_lo", "upper": "num_hi"})
    rate = df[df.metric_name == "Rate"].rename(
        columns={"val": "rate", "lower": "rate_lo", "upper": "rate_hi"})
    key = ["location_id", "cause_name", "year"]
    m = num[key + ["num", "num_lo", "num_hi"]].merge(
        rate[key + ["rate", "rate_lo", "rate_hi"]], on=key, how="inner")
    m["pop"] = np.where(m.rate > 0, m.num / m.rate * 1e5, np.nan)
    return m


inc = load_measure(os.path.join(DATA, "incidence_0_19.csv"))
dal = load_measure(os.path.join(DATA, "merged_gbd_data.csv"),
                   "DALYs (Disability-Adjusted Life Years)")


def agg(sub):
    """aggregate a set of country rows -> num/rate with crude-derived rate UI."""
    n, nl, nh = sub.num.sum(), sub.num_lo.sum(), sub.num_hi.sum()
    pop = sub["pop"].sum()
    r = (lambda x: x / pop * 1e5 if pop > 0 else np.nan)
    return dict(num=n, num_lo=nl, num_hi=nh,
                rate=r(n), rate_lo=r(nl), rate_hi=r(nh))


def row_for(m, loc_ids, cause):
    """one location x disease: 1990 & 2023 number+rate dict, or None."""
    s90 = m[(m.location_id.isin(loc_ids)) & (m.cause_name == cause)
            & (m.year == 1990)]
    s23 = m[(m.location_id.isin(loc_ids)) & (m.cause_name == cause)
            & (m.year == 2023)]
    if s90.empty or s23.empty:
        return None
    if len(loc_ids) == 1:  # single country -> use official rate UI
        a = s90.iloc[0]; b = s23.iloc[0]
        d90 = dict(num=a.num, num_lo=a.num_lo, num_hi=a.num_hi,
                   rate=a.rate, rate_lo=a.rate_lo, rate_hi=a.rate_hi)
        d23 = dict(num=b.num, num_lo=b.num_lo, num_hi=b.num_hi,
                   rate=b.rate, rate_lo=b.rate_lo, rate_hi=b.rate_hi)
    else:
        d90, d23 = agg(s90), agg(s23)
    return d90, d23


# ============================================================ Table 1
ALL = sorted(lmic_ids)
locs = [("All LMIC", "Overall", ALL)]
for g in INCOME:
    locs.append((g, "Income group", [i for i in ALL if grp.get(i) == g]))
ctry = lmic.sort_values("location_name")
for _, rr in ctry.iterrows():
    locs.append((rr.location_name, "Country", [rr.location_id]))

rows = []
for name, typ, ids in locs:
    for cause in DISEASES:
        ri = row_for(inc, ids, cause)
        rd = row_for(dal, ids, cause)
        if ri is None and rd is None:
            continue
        rec = {"Location": name, "Type": typ, "Disease": DLAB[cause]}
        for tag, r in (("Incident cases", ri), ("DALYs", rd)):
            if r is None:
                for c in ("N_1990(k)", "N_2023(k)", "N %change",
                          "Rate_1990", "Rate_2023", "Rate %change"):
                    rec[f"{tag}: {c}"] = "—"
                continue
            a, b = r
            rec[f"{tag}: N_1990(k)"] = ui(a["num"], a["num_lo"], a["num_hi"], 1, 1e3)
            rec[f"{tag}: N_2023(k)"] = ui(b["num"], b["num_lo"], b["num_hi"], 1, 1e3)
            rec[f"{tag}: N %change"] = sp(pct(a["num"], b["num"]), 1)
            rec[f"{tag}: Rate_1990"] = ui(a["rate"], a["rate_lo"], a["rate_hi"], 1)
            rec[f"{tag}: Rate_2023"] = ui(b["rate"], b["rate_lo"], b["rate_hi"], 1)
            rec[f"{tag}: Rate %change"] = sp(pct(a["rate"], b["rate"]), 1)
        rows.append(rec)

cols1 = ["Location", "Type", "Disease"] + [
    f"{t}: {c}" for t in ("Incident cases", "DALYs")
    for c in ("N_1990(k)", "N_2023(k)", "N %change",
              "Rate_1990", "Rate_2023", "Rate %change")]
t1 = pd.DataFrame(rows)[cols1]
t1.to_csv(os.path.join(OUT, "Table1_incidence_DALYs.csv"), index=False)
print(f"   Table1_incidence_DALYs.csv  ({len(t1)} rows, "
      f"{t1.Location.nunique()} locations x {len(DISEASES)} diseases)")

# ============================================================ Table 2 (VLW)
v = pd.read_csv(T1V)
v = v[v.Sex == "Both"].copy()
for col, base in [("DALY", "DALYs"), ("VLW", "VLW (billion USD)"),
                  ("VLWd", "VLW discounted (billion USD)"),
                  ("GDPpct", "VLW/GDP (%)")]:
    v[col] = v[base].map(pt)
    v[[col + "_lo", col + "_hi"]] = v[base].apply(lambda s: pd.Series(bnd(s)))
v["lid"] = v.Country.map(dict(zip(lmic.location_name, lmic.location_id)))
v["GDPtot"] = v["GDP_total_bn"].astype(float)
v["GDPpc"] = v["GDP_pc_PPP"].astype(float)


def vagg(sub):
    """sum VLW/DALY across rows; VLW/GDP recomputed = ΣVLW/ΣGDP*100."""
    out = {}
    for c in ("DALY", "VLW", "VLWd"):
        out[c] = sub[c].sum(); out[c + "_lo"] = sub[c + "_lo"].sum()
        out[c + "_hi"] = sub[c + "_hi"].sum()
    g = sub.drop_duplicates("Country").GDPtot.sum()  # each country GDP once
    out["GDPpct"] = out["VLW"] / g * 100
    out["GDPpct_lo"] = out["VLW_lo"] / g * 100
    out["GDPpct_hi"] = out["VLW_hi"] / g * 100
    return out


def vrec(name, typ, sub, disease, gdppc=None):
    if disease == "Combined":
        d = vagg(sub)
    elif typ == "Country":
        r = sub[sub.Disease == disease].iloc[0]
        d = {c: r[c] for c in ("DALY", "DALY_lo", "DALY_hi", "VLW", "VLW_lo",
             "VLW_hi", "VLWd", "VLWd_lo", "VLWd_hi", "GDPpct", "GDPpct_lo",
             "GDPpct_hi")}
    else:
        d = vagg(sub[sub.Disease == disease])
    return {
        "Location": name, "Type": typ, "Disease": disease,
        "GDP per capita (PPP USD)": sp(gdppc, 0) if gdppc else "—",
        "DALYs": ui(d["DALY"], d["DALY_lo"], d["DALY_hi"], 0),
        "VLW (billion USD)": ui(d["VLW"], d["VLW_lo"], d["VLW_hi"], 3),
        "VLW discounted (billion USD)": ui(d["VLWd"], d["VLWd_lo"], d["VLWd_hi"], 3),
        "VLW/GDP (%)": ui(d["GDPpct"], d["GDPpct_lo"], d["GDPpct_hi"], 3),
    }


DSEQ = ["Depressive disorders", "Anxiety disorders", "Combined"]
rows2 = []
# overall
for dis in DSEQ:
    rows2.append(vrec("All LMIC", "Overall", v, dis))
# income groups
for g in INCOME:
    sg = v[v.Income_Group == g]
    for dis in DSEQ:
        rows2.append(vrec(g, "Income group", sg, dis))
# countries (sorted by combined VLW desc)
order = (v.groupby("Country").VLW.sum().sort_values(ascending=False).index)
for c in order:
    sc = v[v.Country == c]
    pc = sc.GDPpc.iloc[0]
    for dis in DSEQ:
        rows2.append(vrec(c, "Country", sc, dis, gdppc=pc))

t2 = pd.DataFrame(rows2)
t2.to_csv(os.path.join(OUT, "Table2_VLW.csv"), index=False)
print(f"   Table2_VLW.csv  ({len(t2)} rows, {v.Country.nunique()} countries "
      f"x 3 disease rows + overall/income)")
# headline check
chk = vagg(v)
print(f"   check: All-LMIC combined VLW = {chk['VLW']:,.1f} bn "
      f"({chk['GDPpct']:.2f}% GDP)")

# ============================================================ Table 3 (COVID-19)
# Country-level cumulative COVID-19 excess VLW, 2020-2023 (the country detail
# behind Fig6); 95% prediction interval (lo/hi). Excess/GDP uses the cumulative
# 2020-2023 excess over single-year 2023 GDP (same convention as CF_summary).
cfc = pd.read_csv(os.path.join(CFDIR, "CF_country_cumulative_2020_2023.csv"))
cfk = pd.read_csv(os.path.join(CFDIR, "CF_country_combined_cumulative.csv"))
gdp_lid = v.drop_duplicates("lid").set_index("lid")["GDPtot"].to_dict()
DMAP3 = {"Depressive disorders": cfc[cfc.disease == "Depressive disorders"],
         "Anxiety disorders": cfc[cfc.disease == "Anxiety disorders"],
         "Combined": cfk}


def t3_rows(name, typ, ids, gdppc=None):
    gdp = sum(gdp_lid.get(i, np.nan) for i in ids if i in gdp_lid)
    out = []
    for dis, src in DMAP3.items():
        s = src[src.location_id.isin(ids)]
        if s.empty:
            continue
        p, l, h = (s.excess_cum.sum(), s.excess_lo_cum.sum(), s.excess_hi_cum.sum())
        eg = (ui(p / gdp * 100, l / gdp * 100, h / gdp * 100, 4)
              if gdp and np.isfinite(gdp) and gdp > 0 else "—")
        out.append({
            "Location": name, "Type": typ, "Disease": dis,
            "GDP per capita (PPP USD)": sp(gdppc, 0) if gdppc else "—",
            "COVID-19 excess VLW 2020–2023 (billion USD)": ui(p, l, h, 3),
            "Excess/GDP (%)": eg,
        })
    return out


all_ids = cfk.location_id.tolist()
rows3 = t3_rows("All LMIC", "Overall", all_ids)
for g in INCOME:
    rows3 += t3_rows(g, "Income group", cfk[cfk.LMIC_group == g].location_id.tolist())
for _, rr in cfk.sort_values("excess_cum", ascending=False).iterrows():
    pc = v[v.lid == rr.location_id].GDPpc.iloc[0] if (v.lid == rr.location_id).any() else None
    rows3 += t3_rows(rr.location_name, "Country", [rr.location_id], gdppc=pc)

t3 = pd.DataFrame(rows3)
t3.to_csv(os.path.join(OUT, "Table3_COVID_excess.csv"), index=False)
print(f"   Table3_COVID_excess.csv  ({len(t3)} rows, {cfk.location_id.nunique()} "
      f"countries x 3 disease rows + overall/income)")
agg3 = t3_rows("All LMIC", "Overall", all_ids)
for r in agg3:
    print(f"      All-LMIC {r['Disease']}: "
          f"{r['COVID-19 excess VLW 2020–2023 (billion USD)']}")
