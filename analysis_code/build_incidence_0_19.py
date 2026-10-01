#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert the per-year Incidence files in
   data/{568_Depressive_disorders,571_Anxiety_disorders}/measure6_Incidence_year*.csv
into data/incidence_0_19.csv with the SAME schema as merged_gbd_data.csv, so
run_descriptive_epi_MDD.py auto-builds the Fig2 2x2 (Incidence + DALYs).

NOTE: depression incidence is GBD cause 568 (Major depressive disorder), mapped
to "Depressive disorders" so it pairs with the 0-19 DALYs (cause 567). MDD
dominates depressive-disorder incidence; this mismatch is documented.
"""
import os
import glob
import pandas as pd

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
AGE = {1: "<5 years", 6: "5-9 years", 7: "10-14 years", 8: "15-19 years", 158: "<20 years"}
SEX = {1: "Male", 2: "Female", 3: "Both"}
METRIC = {1: "Number", 3: "Rate"}
CAUSE = {568: "Depressive disorders", 571: "Anxiety disorders"}

frames = []
for fol in ["568_Depressive_disorders", "571_Anxiety_disorders"]:
    for f in sorted(glob.glob(os.path.join(DATA, fol, "measure6_Incidence_year*.csv"))):
        frames.append(pd.read_csv(f))
inc = pd.concat(frames, ignore_index=True)
inc = inc[inc.age.isin(AGE) & inc.cause.isin(CAUSE)].copy()

# --- sanity: 4 child bands should sum to the <20 aggregate ---
chk = inc[(inc.metric == 1) & (inc.sex == 3) & (inc.year == 2023) & (inc.cause == 571)]
bands = chk[chk.age.isin([1, 6, 7, 8])].groupby("location").val.sum()
agg = chk[chk.age == 158].set_index("location").val
ratio = (bands / agg).dropna()
print(f"check sum(<5..15-19)/<20 (anxiety 2023, Both, Number): "
      f"mean={ratio.mean():.4f} min={ratio.min():.4f} max={ratio.max():.4f} (expect ~1.0)")

names = pd.read_csv(os.path.join(DATA, "204_with_LMIC.csv"))[["location_id", "location_name"]]
out = pd.DataFrame({
    "location_id": inc.location,
    "sex_name": inc.sex.map(SEX),
    "age_name": inc.age.map(AGE),
    "cause_name": inc.cause.map(CAUSE),
    "metric_name": inc.metric.map(METRIC),
    "year": inc.year,
    "val": inc.val, "upper": inc.upper, "lower": inc.lower,
}).merge(names, on="location_id", how="left")
out = out[["location_id", "location_name", "sex_name", "age_name", "cause_name",
           "metric_name", "year", "val", "upper", "lower"]]
dst = os.path.join(DATA, "incidence_0_19.csv")
out.to_csv(dst, index=False)
print(f"wrote {dst}: {len(out):,} rows | years {out.year.min()}-{out.year.max()} | "
      f"causes {sorted(out.cause_name.unique())} | locations {out.location_id.nunique()}")
