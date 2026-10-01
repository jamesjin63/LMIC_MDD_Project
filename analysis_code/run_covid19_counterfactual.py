#!/usr/bin/env python3
"""
COVID-19 Counterfactual VLW Analysis
Depressive disorders & Anxiety disorders | LMIC | 0-19 years | IE = 1.0

反事实逻辑：
  1. 用 1990–2019 VLW 时间序列拟合 ETS 模型（自动选最优 AIC）
  2. 向前预测 2020–2023（反事实：假设无 COVID-19）
  3. Excess = Actual − CF_forecast（阴影面积 = COVID-19 超额经济负担）
"""

import os
import sys
import warnings
import unicodedata
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# 0. PARAMETERS
# =============================================================================
IE          = 1.0
VSL_US      = 13.2e6       # USD (EPA 2023)
GDP_PC_USA  = 82304.62     # USD PPP (World Bank 2023)
DISC_RATE   = 0.03
N_BOOT      = 1000         # bootstrap simulations for ETS PI
TRAIN_END   = 2019
EVAL_YEARS  = [2020, 2021, 2022, 2023]
H           = 4            # forecast horizon

DISEASE_NAMES  = ['Depressive disorders', 'Anxiety disorders']
DISEASE_COLORS = {'Depressive disorders': '#1B4F72', 'Anxiety disorders': '#C0392B'}
CF_COLOR       = '#5D6D7E'   # counterfactual line
EXCESS_ALPHA   = 0.30
PI_ALPHA       = 0.18
UI_ALPHA       = 0.18

INCOME_LEVELS = ['Low income', 'Lower middle income', 'Upper middle income']
INCOME_COLORS = {
    'Low income':          '#C0392B',
    'Lower middle income': '#E67E22',
    'Upper middle income': '#117A65'
}

GBD_TO_WB = {
    'Bolivia (Plurinational State of)':      'Bolivia',
    'Congo':                                 'Congo, Rep.',
    "Côte d'Ivoire":                         "Cote d'Ivoire",
    "Democratic People's Republic of Korea": "Korea, Dem. People's Rep.",
    'Democratic Republic of the Congo':      'Congo, Dem. Rep.',
    'Egypt':                                 'Egypt, Arab Rep.',
    'Gambia':                                'Gambia, The',
    'Iran (Islamic Republic of)':            'Iran, Islamic Rep.',
    'Kyrgyzstan':                            'Kyrgyz Republic',
    "Lao People's Democratic Republic":      'Lao PDR',
    'Micronesia (Federated States of)':      'Micronesia, Fed. Sts.',
    'Palestine':                             'West Bank and Gaza',
    'Republic of Moldova':                   'Moldova',
    'Saint Kitts and Nevis':                 'St. Kitts and Nevis',
    'Saint Lucia':                           'St. Lucia',
    'Saint Vincent and the Grenadines':      'St. Vincent and the Grenadines',
    'Somalia':                               'Somalia, Fed. Rep.',
    'Türkiye':                               'Turkiye',
    'United Republic of Tanzania':           'Tanzania',
    'Venezuela (Bolivarian Republic of)':    'Venezuela, RB',
    'Viet Nam':                              'Viet Nam',
    'Yemen':                                 'Yemen, Rep.'
}

out_dir = os.path.join('results_VLW_MDD', f'covid19_counterfactual_IE{IE}')
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================

def norm_name(s):
    """Normalise Unicode country name to plain ASCII for matching."""
    if not isinstance(s, str):
        return ''
    s = s.replace('\u2019', "'").replace('\u2018', "'").replace('\u00a0', ' ')
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'\s+', ' ', s).strip()

def calc_discount(remaining_yrs, r=0.03):
    if remaining_yrs <= 0:
        return 0.0
    return (1 - (1 + r) ** (-remaining_yrs)) / (r * remaining_yrs)

def calc_vlw(daly, gdp_pc, hale, ie=IE, vsl_us=VSL_US, gdp_pc_us=GDP_PC_USA,
             disc_rate=DISC_RATE, gdp_total=None):
    """Vectorised VLW calculation (pandas Series or scalars)."""
    vsl_i   = vsl_us * (gdp_pc / gdp_pc_us) ** ie
    vsly    = vsl_i / (hale / 2)
    rem_yrs = hale * 0.4
    disc    = rem_yrs.apply(calc_discount) if hasattr(rem_yrs, 'apply') else calc_discount(rem_yrs)
    vlw     = vsly * daly / 1e9
    vlw_disc = vlw * disc
    result  = {'VLW': vlw, 'VLW_disc': vlw_disc}
    if gdp_total is not None:
        result['VLW_GDP_pct'] = vlw * 1e9 / gdp_total * 100
    return result

def fit_best_ets(series):
    """
    Try ETS(A,N,N), ETS(A,A,N), ETS(A,Ad,N) and return best by AIC.
    Returns (fitted_model, model_label, aic).
    """
    configs = [
        (None,  False, 'ETS(A,N,N)'),
        ('add', False, 'ETS(A,A,N)'),
        ('add', True,  'ETS(A,Ad,N)'),
    ]
    best_aic   = np.inf
    best_fit   = None
    best_label = ''
    for trend, damped, label in configs:
        try:
            kw = dict(trend=trend, seasonal=None)
            if trend is not None:
                kw['damped_trend'] = damped
            m   = ExponentialSmoothing(series, **kw)
            fit = m.fit(optimized=True, remove_bias=False)
            if fit.aic < best_aic:
                best_aic   = fit.aic
                best_fit   = fit
                best_label = label
        except Exception:
            pass
    return best_fit, best_label, best_aic

def ets_forecast_with_pi(series, h=H, n_boot=N_BOOT):
    """
    Fit best ETS on series, forecast h steps.
    Returns: forecast (array h), pi_lo (array h), pi_hi (array h),
             label (str), aic (float)
    """
    fit, label, aic = fit_best_ets(series)
    if fit is None:
        nans = np.full(h, np.nan)
        return nans, nans, nans, 'FAILED', np.nan

    fc = np.asarray(fit.forecast(h))

    # Bootstrap PI using ETS state simulation (correct error propagation)
    # statsmodels simulate() re-runs the state equations with bootstrapped innovations,
    # avoiding the cumsum random-walk inflation of naive residual accumulation.
    try:
        sims_raw = fit.simulate(nsimulations=h, anchor='end',
                                repetitions=n_boot, random_errors='bootstrap')
        sims = np.asarray(sims_raw)   # shape (h, n_boot)
    except Exception:
        # Analytical fallback: σ × sqrt(1 + (h-1)·α²)
        sigma = np.std(np.asarray(fit.resid), ddof=1)
        alpha = float(getattr(fit.params, 'smoothing_level',
                              getattr(fit, 'params', {}).get('smoothing_level', 0.2)))
        sims = np.zeros((h, n_boot))
        for j in range(h):
            scale = sigma * np.sqrt(1.0 + j * alpha ** 2)
            sims[j, :] = fc[j] + np.random.normal(0.0, scale, n_boot)

    pi_lo = np.percentile(sims, 2.5,  axis=1)
    pi_hi = np.percentile(sims, 97.5, axis=1)
    return fc, pi_lo, pi_hi, label, aic

# =============================================================================
# 2. LOAD DATA
# =============================================================================
print('══════════════════════════════════════════════════════════════════')
print(f'  COVID-19 Counterfactual VLW  |  IE = {IE}  |  LMIC  |  0-19 yrs')
print('══════════════════════════════════════════════════════════════════\n')
print('[1/6] Loading data...')

# LMIC classification
df_lmic_raw = pd.read_csv('204_with_LMIC.csv')
df_lmic = df_lmic_raw[df_lmic_raw['LMIC'] == 1][
    ['location_id', 'location_name', 'LMIC_group']
].copy()
df_lmic['wb_country'] = df_lmic['location_name'].map(
    lambda x: GBD_TO_WB.get(x, x))
df_lmic['wb_norm'] = df_lmic['wb_country'].apply(norm_name)

# GDP (2023)
df_gdp_raw = pd.read_csv('gdp.csv')
df_gdp_raw = df_gdp_raw[
    (df_gdp_raw['year'] == 2023) &
    df_gdp_raw['NY.GDP.PCAP.PP.CD'].notna() &
    df_gdp_raw['NY.GDP.MKTP.PP.CD'].notna()
][['country', 'NY.GDP.PCAP.PP.CD', 'NY.GDP.MKTP.PP.CD']].copy()
df_gdp_raw.columns = ['country', 'GDP_pc_PPP', 'GDP_PPP_total']
df_gdp_raw['country_norm'] = df_gdp_raw['country'].apply(norm_name)

df_gdp = df_lmic.merge(df_gdp_raw, left_on='wb_norm', right_on='country_norm',
                        how='left').dropna(subset=['GDP_pc_PPP'])
df_gdp = df_gdp[['location_id', 'location_name', 'LMIC_group',
                  'GDP_pc_PPP', 'GDP_PPP_total']].copy()
lmic_ids = set(df_gdp['location_id'])
print(f'   LMIC countries with GDP data: {len(lmic_ids)}')

# HALE (2023, All ages, Both sexes = sex_id 3)
df_hale_raw = pd.read_csv('HALE.csv')
df_hale = df_hale_raw[
    (df_hale_raw['metric_name'] == 'Years') &
    (df_hale_raw['year'] == 2023) &
    (df_hale_raw['age_name'] == 'All ages')
][['location_id', 'sex_id', 'val']].copy()
df_hale.columns = ['location_id', 'sex_id', 'HALE']

# DALYs — <20 years, Both sexes, 1990-2023
print('   Loading DALYs (<20 years, Both sexes)...')
df_daly_raw = pd.read_csv('merged_gbd_data.csv')
df_daly = df_daly_raw[
    (df_daly_raw['cause_name'].isin(DISEASE_NAMES)) &
    (df_daly_raw['measure_name'] == 'DALYs (Disability-Adjusted Life Years)') &
    (df_daly_raw['metric_name']  == 'Number') &
    (df_daly_raw['age_name']     == '<20 years') &
    (df_daly_raw['sex_name']     == 'Both') &
    (df_daly_raw['location_id'].isin(lmic_ids))
][['location_id', 'location_name', 'cause_name', 'year',
   'val', 'lower', 'upper']].copy()
df_daly.rename(columns={'cause_name': 'disease', 'val': 'DALY'}, inplace=True)
print(f'   DALYs rows: {len(df_daly):,}')

# =============================================================================
# 3. COMPUTE VLW  (country × disease × year)
# =============================================================================
print('\n[2/6] Computing VLW (IE = 1.0)...')

# Merge GDP and HALE (sex_id=3 for Both)
hale_both = df_hale[df_hale['sex_id'] == 3][['location_id', 'HALE']].copy()
df_vlw = (df_daly
          .merge(df_gdp[['location_id', 'LMIC_group', 'GDP_pc_PPP', 'GDP_PPP_total']],
                 on='location_id', how='inner')
          .merge(hale_both, on='location_id', how='inner'))

df_vlw['VSL_i']         = VSL_US * (df_vlw['GDP_pc_PPP'] / GDP_PC_USA) ** IE
df_vlw['VSLY']          = df_vlw['VSL_i'] / (df_vlw['HALE'] / 2)
df_vlw['remaining_yrs'] = df_vlw['HALE'] * 0.4
df_vlw['disc_factor']   = df_vlw['remaining_yrs'].apply(calc_discount)

df_vlw['VLW']       = df_vlw['VSLY'] * df_vlw['DALY']  / 1e9
df_vlw['VLW_lower'] = df_vlw['VSLY'] * df_vlw['lower'] / 1e9
df_vlw['VLW_upper'] = df_vlw['VSLY'] * df_vlw['upper'] / 1e9

df_vlw['VLW_GDP_pct']       = df_vlw['VLW']       * 1e9 / df_vlw['GDP_PPP_total'] * 100
df_vlw['VLW_GDP_pct_lower'] = df_vlw['VLW_lower'] * 1e9 / df_vlw['GDP_PPP_total'] * 100
df_vlw['VLW_GDP_pct_upper'] = df_vlw['VLW_upper'] * 1e9 / df_vlw['GDP_PPP_total'] * 100

n_ctry = df_vlw['location_id'].nunique()
n_yr   = df_vlw['year'].nunique()
print(f'   VLW computed: {n_ctry} countries × {n_yr} years × 2 diseases')

# =============================================================================
# 4. ETS COUNTERFACTUAL  (LMIC aggregate + country-level)
# =============================================================================
print('\n[3/6] Fitting ETS models and computing counterfactuals...')

# ── 4a. LMIC aggregate series (sum across countries) ─────────────────────────
agg = (df_vlw.groupby(['year', 'disease'])
             .agg(VLW_total=('VLW', 'sum'),
                  VLW_lo   =('VLW_lower', 'sum'),
                  VLW_hi   =('VLW_upper', 'sum'))
             .reset_index())

cf_records = []   # store counterfactual results
ets_info   = []   # ETS model info

for disease in DISEASE_NAMES:
    sub = agg[agg['disease'] == disease].sort_values('year')
    train = sub[sub['year'] <= TRAIN_END]['VLW_total'].values

    fc, pi_lo, pi_hi, label, aic = ets_forecast_with_pi(train, h=H)
    print(f'   {disease}: {label}  (AIC = {aic:.1f})')

    ets_info.append({'Disease': disease, 'ETS_model': label, 'AIC': round(aic, 1),
                     'Train_period': '1990–2019', 'Forecast_horizon': H})

    for i, yr in enumerate(EVAL_YEARS):
        row_actual = sub[sub['year'] == yr]
        if len(row_actual) == 0:
            continue
        actual     = row_actual['VLW_total'].values[0]
        actual_lo  = row_actual['VLW_lo'].values[0]
        actual_hi  = row_actual['VLW_hi'].values[0]
        cf_val     = fc[i]
        cf_lo      = pi_lo[i]
        cf_hi      = pi_hi[i]
        excess     = actual - cf_val
        excess_lo  = actual_lo - cf_hi   # conservative lower
        excess_hi  = actual_hi - cf_lo   # conservative upper

        cf_records.append({
            'disease':     disease,
            'year':        yr,
            'actual':      actual,
            'actual_lo':   actual_lo,
            'actual_hi':   actual_hi,
            'cf_point':    cf_val,
            'cf_pi_lo':    cf_lo,
            'cf_pi_hi':    cf_hi,
            'excess':      excess,
            'excess_lo':   excess_lo,
            'excess_hi':   excess_hi
        })

df_cf = pd.DataFrame(cf_records)

# ── 4b. Country-level ETS ─────────────────────────────────────────────────────
print('   Fitting country-level ETS...')
country_cf_records = []

for disease in DISEASE_NAMES:
    for loc_id, grp in df_vlw[df_vlw['disease'] == disease].groupby('location_id'):
        grp = grp.sort_values('year')
        train_grp = grp[grp['year'] <= TRAIN_END]['VLW'].values

        if len(train_grp) < 10 or np.all(train_grp == 0):
            continue

        fc_c, pi_lo_c, pi_hi_c, _, _ = ets_forecast_with_pi(
            train_grp, h=H, n_boot=200)  # lighter bootstrap for speed

        for i, yr in enumerate(EVAL_YEARS):
            row_a = grp[grp['year'] == yr]
            if len(row_a) == 0 or np.isnan(fc_c[i]):
                continue
            actual_c    = row_a['VLW'].values[0]
            actual_lo_c = row_a['VLW_lower'].values[0]
            actual_hi_c = row_a['VLW_upper'].values[0]
            excess_c    = actual_c - fc_c[i]

            country_cf_records.append({
                'location_id':   loc_id,
                'location_name': grp['location_name'].iloc[0],
                'LMIC_group':    grp['LMIC_group'].iloc[0],
                'disease':       disease,
                'year':          yr,
                'actual':        actual_c,
                'cf_point':      fc_c[i],
                'excess':        excess_c,
                'excess_lo':     actual_lo_c - pi_hi_c[i],
                'excess_hi':     actual_hi_c - pi_lo_c[i]
            })

df_country_cf = pd.DataFrame(country_cf_records)

# Cumulative country excess 2020-2023
df_country_cum = (df_country_cf
                  .groupby(['location_id', 'location_name', 'LMIC_group', 'disease'])
                  .agg(excess_cum   =('excess',    'sum'),
                       excess_lo_cum=('excess_lo', 'sum'),
                       excess_hi_cum=('excess_hi', 'sum'))
                  .reset_index())

# Combined (2 diseases) country cumulative
df_country_comb = (df_country_cf
                   .groupby(['location_id', 'location_name', 'LMIC_group', 'year'])
                   .agg(excess=('excess', 'sum'),
                        excess_lo=('excess_lo', 'sum'),
                        excess_hi=('excess_hi', 'sum'))
                   .reset_index())
df_country_comb_cum = (df_country_comb
                       .groupby(['location_id', 'location_name', 'LMIC_group'])
                       .agg(excess_cum   =('excess', 'sum'),
                            excess_lo_cum=('excess_lo', 'sum'),
                            excess_hi_cum=('excess_hi', 'sum'))
                       .reset_index())

# Income-group aggregation
df_income_cf = (df_country_cf
                .groupby(['LMIC_group', 'disease', 'year'])
                .agg(excess=('excess', 'sum'),
                     excess_lo=('excess_lo', 'sum'),
                     excess_hi=('excess_hi', 'sum'))
                .reset_index())

print(f'   Country-level ETS done: {df_country_cf["location_name"].nunique()} countries fitted')

# =============================================================================
# 5. SAVE CSV
# =============================================================================
print('\n[4/6] Saving CSV outputs...')

# Full LMIC aggregate series (historical + CF for eval years)
rows_hist = []
for disease in DISEASE_NAMES:
    sub = agg[agg['disease'] == disease].sort_values('year')
    for _, r in sub[sub['year'] <= TRAIN_END].iterrows():
        rows_hist.append({'disease': disease, 'year': int(r['year']),
                          'period': 'Historical',
                          'actual': r['VLW_total'],
                          'actual_lo': r['VLW_lo'], 'actual_hi': r['VLW_hi'],
                          'cf_point': np.nan, 'cf_pi_lo': np.nan, 'cf_pi_hi': np.nan,
                          'excess': np.nan, 'excess_lo': np.nan, 'excess_hi': np.nan})

rows_eval = []
for _, r in df_cf.iterrows():
    rows_eval.append({'disease': r['disease'], 'year': int(r['year']),
                      'period': 'Evaluation (COVID-19)',
                      'actual': r['actual'],
                      'actual_lo': r['actual_lo'], 'actual_hi': r['actual_hi'],
                      'cf_point': r['cf_point'],
                      'cf_pi_lo': r['cf_pi_lo'], 'cf_pi_hi': r['cf_pi_hi'],
                      'excess': r['excess'],
                      'excess_lo': r['excess_lo'], 'excess_hi': r['excess_hi']})

df_full_series = pd.DataFrame(rows_hist + rows_eval)
df_full_series.to_csv(os.path.join(out_dir, 'CF_lmic_aggregate_series.csv'), index=False)
print('   CF_lmic_aggregate_series.csv saved.')

# Summary: cumulative excess per disease
def fmt_ui_val(v, lo, hi, dec=3):
    return f'{v:.{dec}f} ({lo:.{dec}f}–{hi:.{dec}f})'

total_gdp = df_gdp['GDP_PPP_total'].sum()
summary_rows = []
for disease in DISEASE_NAMES:
    sub = df_cf[df_cf['disease'] == disease]
    tot   = sub['excess'].sum()
    tot_lo = sub['excess_lo'].sum()
    tot_hi = sub['excess_hi'].sum()
    summary_rows.append({
        'Disease': disease,
        'Cumulative_excess_VLW_bn': round(tot, 4),
        'Cumulative_excess_lo_bn':  round(tot_lo, 4),
        'Cumulative_excess_hi_bn':  round(tot_hi, 4),
        'Excess_VLW_formatted':     fmt_ui_val(tot, tot_lo, tot_hi),
        'Excess_as_pct_GDP':        round(tot * 1e9 / total_gdp * 100, 4),
        **{f'Excess_{yr}_bn': round(sub[sub['year']==yr]['excess'].values[0], 4)
           for yr in EVAL_YEARS}
    })
comb_excess = df_cf.groupby('year')['excess'].sum()
summary_rows.append({
    'Disease': 'Combined (2 diseases)',
    'Cumulative_excess_VLW_bn': round(comb_excess.sum(), 4),
    'Cumulative_excess_lo_bn':  round(df_cf['excess_lo'].sum(), 4),
    'Cumulative_excess_hi_bn':  round(df_cf['excess_hi'].sum(), 4),
    'Excess_VLW_formatted':     fmt_ui_val(comb_excess.sum(),
                                            df_cf['excess_lo'].sum(),
                                            df_cf['excess_hi'].sum()),
    'Excess_as_pct_GDP':        round(comb_excess.sum() * 1e9 / total_gdp * 100, 4),
    **{f'Excess_{yr}_bn': round(comb_excess.get(yr, 0), 4) for yr in EVAL_YEARS}
})
pd.DataFrame(summary_rows).to_csv(
    os.path.join(out_dir, 'CF_summary_lmic.csv'), index=False)
print('   CF_summary_lmic.csv saved.')

# Country-level cumulative
df_country_cum.round(6).sort_values(['disease', 'excess_cum'], ascending=[True, False])\
    .to_csv(os.path.join(out_dir, 'CF_country_cumulative_2020_2023.csv'), index=False)
df_country_comb_cum.round(6).sort_values('excess_cum', ascending=False)\
    .to_csv(os.path.join(out_dir, 'CF_country_combined_cumulative.csv'), index=False)
print('   CF_country_cumulative_2020_2023.csv saved.')
print('   CF_country_combined_cumulative.csv saved.')

# ETS model info
pd.DataFrame(ets_info).to_csv(
    os.path.join(out_dir, 'CF_ets_model_info.csv'), index=False)
print('   CF_ets_model_info.csv saved.')

# Annual excess detail
df_cf.round(6).to_csv(os.path.join(out_dir, 'CF_annual_excess_lmic.csv'), index=False)
print('   CF_annual_excess_lmic.csv saved.')

# =============================================================================
# 6. FIGURES
# =============================================================================
print('\n[5/6] Generating figures...')

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         8,
    'axes.titlesize':    8.5,
    'axes.titleweight':  'bold',
    'axes.labelsize':    8,
    'axes.labelweight':  'bold',
    'axes.linewidth':    0.7,
    'axes.edgecolor':    '#444444',
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'xtick.major.size':  3.5,
    'ytick.major.size':  3.5,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'legend.fontsize':   7,
    'legend.frameon':    False,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'pdf.fonttype':      42,
})

def bn_fmt(x, pos=None):
    if x >= 1000: return f'{x/1000:.1f}K'
    if x >= 1:    return f'{x:.2f}'
    return f'{x:.3f}'

def pct_fmt(x, pos=None):
    if abs(x) >= 0.1:  return f'{x:.3f}%'
    return f'{x:.4f}%'

# ── Figure 1: LMIC aggregate time series + CF + shading ──────────────────────
fig1, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

for ax, disease in zip(axes, DISEASE_NAMES):
    col = DISEASE_COLORS[disease]
    sub = agg[agg['disease'] == disease].sort_values('year')

    # Historical actual (1990-2023 full line for context)
    yrs_all = sub['year'].values
    vlw_all = sub['VLW_total'].values
    vlo_all = sub['VLW_lo'].values
    vhi_all = sub['VLW_hi'].values

    # 1990-2019 fitted segment
    mask_train = yrs_all <= TRAIN_END
    # 2020-2023 actual segment
    mask_eval  = yrs_all >= TRAIN_END

    # GBD 95% UI ribbon (full history)
    ax.fill_between(yrs_all, vlo_all, vhi_all,
                    color=col, alpha=UI_ALPHA, linewidth=0, zorder=1)
    # Actual line
    ax.plot(yrs_all, vlw_all, color=col, linewidth=1.6,
            label='Actual VLW (GBD 2023)', zorder=4)

    # CF forecast line (2019-2023, connect from 2019 actual)
    cf_sub   = df_cf[df_cf['disease'] == disease].sort_values('year')
    cf_years = np.array([TRAIN_END] + list(cf_sub['year']))
    cf_vals  = np.array([sub[sub['year'] == TRAIN_END]['VLW_total'].values[0]]
                        + list(cf_sub['cf_point']))
    pi_lo_line = np.array([sub[sub['year'] == TRAIN_END]['VLW_lo'].values[0]]
                           + list(cf_sub['cf_pi_lo']))
    pi_hi_line = np.array([sub[sub['year'] == TRAIN_END]['VLW_hi'].values[0]]
                           + list(cf_sub['cf_pi_hi']))

    # ETS 95% PI band
    ax.fill_between(cf_years, pi_lo_line, pi_hi_line,
                    color=CF_COLOR, alpha=PI_ALPHA, linewidth=0, zorder=2)
    # CF dashed line
    ax.plot(cf_years, cf_vals, color=CF_COLOR, linewidth=1.4,
            linestyle='--', dashes=(5, 3), label='Counterfactual (ETS, no COVID-19)', zorder=5)

    # Excess shading between actual and CF (2020-2023)
    actual_eval = np.array([sub[sub['year'] == TRAIN_END]['VLW_total'].values[0]]
                            + [sub[sub['year'] == yr]['VLW_total'].values[0]
                               for yr in EVAL_YEARS])
    # COVID excess fill
    y_up = np.maximum(actual_eval, cf_vals)
    y_dn = np.minimum(actual_eval, cf_vals)
    excess_is_pos = actual_eval >= cf_vals
    # Positive excess (actual > CF) = red; negative = blue
    ax.fill_between(cf_years, cf_vals, actual_eval,
                    where=excess_is_pos,
                    color='#E74C3C', alpha=EXCESS_ALPHA, linewidth=0,
                    label='COVID-19 excess VLW (actual > CF)', zorder=3)
    ax.fill_between(cf_years, cf_vals, actual_eval,
                    where=~excess_is_pos,
                    color='#2980B9', alpha=EXCESS_ALPHA, linewidth=0,
                    label='Actual below CF', zorder=3)

    # COVID-19 onset line
    ax.axvline(x=2019.5, color='#7F8C8D', linewidth=0.9,
               linestyle=':', zorder=6)
    ax.text(2019.6, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else vlw_all.max() * 0.95,
            'COVID-19\nonset', fontsize=6.5, color='#7F8C8D',
            va='top', ha='left')

    # Annotate total excess
    d_cf = df_cf[df_cf['disease'] == disease]
    tot_ex  = d_cf['excess'].sum()
    tot_lo  = d_cf['excess_lo'].sum()
    tot_hi  = d_cf['excess_hi'].sum()
    sign    = '+' if tot_ex >= 0 else ''
    ax.text(0.99, 0.06,
            f'Cumulative excess 2020–2023:\n{sign}{tot_ex:.3f} bn USD\n'
            f'({tot_lo:.3f}–{tot_hi:.3f})',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color='#E74C3C' if tot_ex >= 0 else '#2980B9',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#CCCCCC',
                      alpha=0.85, lw=0.5))

    ax.set_xlim(1989, 2024)
    ax.set_xticks([1990, 1995, 2000, 2005, 2010, 2015, 2019, 2023])
    ax.set_xticklabels(['1990', '1995', '2000', '2005',
                         '2010', '2015', '2019', '2023'])
    ax.yaxis.set_major_formatter(FuncFormatter(bn_fmt))
    ax.set_xlabel('Year')
    ax.set_ylabel('VLW (billion USD)')
    ax.legend(loc='upper left', fontsize=6.5, ncol=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].text(-0.07, 1.02, 'a', transform=axes[0].transAxes,
             fontsize=12, fontweight='bold', va='bottom')
axes[1].text(-0.07, 1.02, 'b', transform=axes[1].transAxes,
             fontsize=12, fontweight='bold', va='bottom')

out_f1 = os.path.join(out_dir, 'CF_F1_lmic_aggregate_timeseries.pdf')
fig1.savefig(out_f1, bbox_inches='tight')
plt.close(fig1)
print(f'   CF_F1 saved.')

# ── Figure 2: Annual excess bar chart  (2 diseases × 4 years) ────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)

for ax, disease in zip(axes2, DISEASE_NAMES):
    col = DISEASE_COLORS[disease]
    sub = df_cf[df_cf['disease'] == disease].sort_values('year')
    yrs = [str(y) for y in sub['year']]
    ex  = sub['excess'].values
    lo  = ex - sub['excess_lo'].values
    hi  = sub['excess_hi'].values - ex
    err = np.array([np.abs(lo), hi])

    bar_colors = ['#E74C3C' if v >= 0 else '#2980B9' for v in ex]
    bars = ax.bar(yrs, ex, color=bar_colors, edgecolor='grey',
                  linewidth=0.4, zorder=3)
    ax.errorbar(yrs, ex, yerr=err, fmt='none',
                ecolor='#333333', elinewidth=0.8, capsize=3.5, capthick=0.8, zorder=4)

    ax.axhline(0, color='#555555', linewidth=0.7, zorder=2)
    ax.yaxis.set_major_formatter(FuncFormatter(bn_fmt))
    ax.set_xlabel('Year')
    ax.set_ylabel('Excess VLW (billion USD)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Value labels
    for bar, v in zip(bars, ex):
        va = 'bottom' if v >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, v + (0.002 if v >= 0 else -0.002),
                f'{v:+.3f}', ha='center', va=va, fontsize=7, fontweight='bold',
                color='#C0392B' if v >= 0 else '#2980B9')

axes2[0].text(-0.12, 1.04, 'a', transform=axes2[0].transAxes,
              fontsize=12, fontweight='bold')
axes2[1].text(-0.12, 1.04, 'b', transform=axes2[1].transAxes,
              fontsize=12, fontweight='bold')
out_f2 = os.path.join(out_dir, 'CF_F2_annual_excess_bar.pdf')
fig2.savefig(out_f2, bbox_inches='tight')
plt.close(fig2)
print('   CF_F2 saved.')

# ── Figure 3: Top 20 countries — cumulative excess (combined 2 diseases) ─────
top20 = (df_country_comb_cum
         .nlargest(20, 'excess_cum')
         .sort_values('excess_cum'))

fig3, axes3 = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)

# Panel a: combined excess
ax3a = axes3[0]
bar_cols_t20 = [INCOME_COLORS.get(g, '#888888') for g in top20['LMIC_group']]
ax3a.barh(top20['location_name'], top20['excess_cum'],
          color=bar_cols_t20, edgecolor='grey', linewidth=0.3, zorder=3)
# Error bars
xerr_lo = top20['excess_cum'] - top20['excess_lo_cum']
xerr_hi = top20['excess_hi_cum'] - top20['excess_cum']
ax3a.errorbar(top20['excess_cum'], top20['location_name'],
              xerr=[xerr_lo.clip(0), xerr_hi.clip(0)],
              fmt='none', ecolor='#333333', elinewidth=0.7, capsize=2.5)
ax3a.axvline(0, color='#555555', linewidth=0.6)
ax3a.xaxis.set_major_formatter(FuncFormatter(bn_fmt))
ax3a.set_xlabel('Cumulative excess VLW 2020–2023\n(billion USD)')
ax3a.set_title('')
ax3a.tick_params(axis='y', labelsize=6.5)
ax3a.spines['top'].set_visible(False)
ax3a.spines['right'].set_visible(False)
ax3a.text(-0.12, 1.03, 'a', transform=ax3a.transAxes,
          fontsize=12, fontweight='bold')

# Panel b: by disease side-by-side for the same top 20 countries
ax3b = axes3[1]
country_order = top20['location_name'].values
y_pos = np.arange(len(country_order))
width = 0.38

for i, disease in enumerate(DISEASE_NAMES):
    sub_d = df_country_cum[df_country_cum['disease'] == disease].copy()
    sub_d = sub_d.set_index('location_name')
    vals  = [sub_d.loc[c, 'excess_cum'] if c in sub_d.index else 0
             for c in country_order]
    offset = (i - 0.5) * width
    ax3b.barh(y_pos + offset, vals,
              height=width, color=DISEASE_COLORS[disease],
              alpha=0.85, edgecolor='grey', linewidth=0.2,
              label=disease, zorder=3)

ax3b.set_yticks(y_pos)
ax3b.set_yticklabels(country_order, fontsize=6.5)
ax3b.axvline(0, color='#555555', linewidth=0.6)
ax3b.xaxis.set_major_formatter(FuncFormatter(bn_fmt))
ax3b.set_xlabel('Cumulative excess VLW 2020–2023\n(billion USD)')
ax3b.legend(fontsize=7, loc='lower right')
ax3b.spines['top'].set_visible(False)
ax3b.spines['right'].set_visible(False)
ax3b.text(-0.1, 1.03, 'b', transform=ax3b.transAxes,
          fontsize=12, fontweight='bold')

# Income group legend for panel a
patches = [mpatches.Patch(color=INCOME_COLORS[g], label=g) for g in INCOME_LEVELS]
fig3.legend(handles=patches, loc='lower center', ncol=3, fontsize=7,
            title='Income Group', title_fontsize=7.5,
            frameon=False, bbox_to_anchor=(0.5, -0.04))
out_f3 = os.path.join(out_dir, 'CF_F3_top20_country_excess.pdf')
fig3.savefig(out_f3, bbox_inches='tight')
plt.close(fig3)
print('   CF_F3 saved.')

# ── Figure 4: Annual excess by income group ───────────────────────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)

for ax, disease in zip(axes4, DISEASE_NAMES):
    col = DISEASE_COLORS[disease]
    sub_inc = df_income_cf[df_income_cf['disease'] == disease].sort_values('year')
    x = np.arange(len(EVAL_YEARS))
    width_i = 0.26

    for j, inc_grp in enumerate(INCOME_LEVELS):
        sub_g = sub_inc[sub_inc['LMIC_group'] == inc_grp].sort_values('year')
        if len(sub_g) == 0:
            continue
        vals  = sub_g['excess'].values
        lo_e  = np.abs(vals - sub_g['excess_lo'].values)
        hi_e  = np.abs(sub_g['excess_hi'].values - vals)
        offset = (j - 1) * width_i
        ax.bar(x + offset, vals, width=width_i,
               color=INCOME_COLORS[inc_grp], alpha=0.85,
               edgecolor='grey', linewidth=0.3,
               label=inc_grp, zorder=3)
        ax.errorbar(x + offset, vals, yerr=[lo_e, hi_e],
                    fmt='none', ecolor='#333333',
                    elinewidth=0.7, capsize=2, capthick=0.7, zorder=4)

    ax.axhline(0, color='#555555', linewidth=0.6, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in EVAL_YEARS])
    ax.set_xlabel('Year')
    ax.set_ylabel('Excess VLW (billion USD)')
    ax.yaxis.set_major_formatter(FuncFormatter(bn_fmt))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes4[0].legend(fontsize=6.5, loc='upper left', title='Income Group',
                title_fontsize=7)
axes4[0].text(-0.12, 1.04, 'a', transform=axes4[0].transAxes,
              fontsize=12, fontweight='bold')
axes4[1].text(-0.12, 1.04, 'b', transform=axes4[1].transAxes,
              fontsize=12, fontweight='bold')
out_f4 = os.path.join(out_dir, 'CF_F4_excess_by_income_group.pdf')
fig4.savefig(out_f4, bbox_inches='tight')
plt.close(fig4)
print('   CF_F4 saved.')

# ── Figure 5: Combined 2-disease aggregate (overlay + shading) ───────────────
fig5, ax5 = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

# Sum across diseases
agg_comb = (agg.groupby('year')
               .agg(VLW_total=('VLW_total', 'sum'),
                    VLW_lo   =('VLW_lo',    'sum'),
                    VLW_hi   =('VLW_hi',    'sum'))
               .reset_index().sort_values('year'))

cf_comb_yr   = np.array(EVAL_YEARS)
cf_comb_pt   = df_cf.groupby('year')['cf_point'].sum().reindex(EVAL_YEARS).values
cf_comb_lo   = df_cf.groupby('year')['cf_pi_lo'].sum().reindex(EVAL_YEARS).values
cf_comb_hi   = df_cf.groupby('year')['cf_pi_hi'].sum().reindex(EVAL_YEARS).values

yrs_all = agg_comb['year'].values
vlw_all = agg_comb['VLW_total'].values
vlo_all = agg_comb['VLW_lo'].values
vhi_all = agg_comb['VLW_hi'].values

# UI ribbon
ax5.fill_between(yrs_all, vlo_all, vhi_all,
                 color='#2C3E50', alpha=0.13, linewidth=0, zorder=1)
# Actual line
ax5.plot(yrs_all, vlw_all, color='#2C3E50', linewidth=1.8,
         label='Actual combined VLW', zorder=4)

# CF line + PI
anchor_val = agg_comb[agg_comb['year'] == TRAIN_END]['VLW_total'].values[0]
anchor_lo  = agg_comb[agg_comb['year'] == TRAIN_END]['VLW_lo'].values[0]
anchor_hi  = agg_comb[agg_comb['year'] == TRAIN_END]['VLW_hi'].values[0]

cf_x = np.array([TRAIN_END] + list(cf_comb_yr))
cf_y = np.array([anchor_val] + list(cf_comb_pt))
pi_lo_c = np.array([anchor_lo] + list(cf_comb_lo))
pi_hi_c = np.array([anchor_hi] + list(cf_comb_hi))

ax5.fill_between(cf_x, pi_lo_c, pi_hi_c,
                 color=CF_COLOR, alpha=PI_ALPHA, linewidth=0, zorder=2,
                 label='CF 95% PI')
ax5.plot(cf_x, cf_y, color=CF_COLOR, linewidth=1.5,
         linestyle='--', dashes=(5, 3), label='Counterfactual (no COVID-19)', zorder=5)

# Actual at eval years for shading
actual_x = np.array([TRAIN_END] + [
    agg_comb[agg_comb['year'] == yr]['VLW_total'].values[0]
    for yr in EVAL_YEARS])
ax5.fill_between(cf_x, cf_y, actual_x,
                 where=(actual_x >= cf_y),
                 color='#E74C3C', alpha=EXCESS_ALPHA, linewidth=0,
                 label='COVID-19 excess (actual > CF)', zorder=3)
ax5.fill_between(cf_x, cf_y, actual_x,
                 where=(actual_x < cf_y),
                 color='#2980B9', alpha=EXCESS_ALPHA, linewidth=0,
                 label='Actual below CF', zorder=3)

ax5.axvline(x=2019.5, color='#7F8C8D', linewidth=0.9, linestyle=':', zorder=6)
ax5.text(2019.65, vlw_all.max() * 0.97, 'COVID-19\nonset',
         fontsize=6.5, color='#7F8C8D', va='top')

tot_comb   = df_cf['excess'].sum()
tot_comb_lo = df_cf['excess_lo'].sum()
tot_comb_hi = df_cf['excess_hi'].sum()
ax5.text(0.02, 0.97,
         f'Combined excess (Dep + Anx) 2020–2023:\n'
         f'{tot_comb:+.3f} bn USD  ({tot_comb_lo:.3f}–{tot_comb_hi:.3f})',
         transform=ax5.transAxes, fontsize=7.5, va='top', ha='left',
         color='#C0392B',
         bbox=dict(boxstyle='round,pad=0.35', fc='white',
                   ec='#CCCCCC', alpha=0.9, lw=0.5))

ax5.set_xlim(1989, 2024)
ax5.set_xticks([1990, 1995, 2000, 2005, 2010, 2015, 2019, 2023])
ax5.xaxis.set_tick_params(labelsize=7)
ax5.yaxis.set_major_formatter(FuncFormatter(bn_fmt))
ax5.set_xlabel('Year')
ax5.set_ylabel('Total VLW (billion USD)')
ax5.legend(loc='upper left', fontsize=6.5)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

out_f5 = os.path.join(out_dir, 'CF_F5_combined_timeseries.pdf')
fig5.savefig(out_f5, bbox_inches='tight')
plt.close(fig5)
print('   CF_F5 saved.')

# =============================================================================
# 7. PRINT SUMMARY
# =============================================================================
print('\n[6/6] Summary:')
print(f"{'Disease':<30}  {'Excess VLW 2020-2023 (bn USD)':<35}  {'% LMIC GDP'}")
print('-' * 80)
for _, row in pd.DataFrame(summary_rows).iterrows():
    print(f"  {row['Disease']:<28}  {row['Excess_VLW_formatted']:<35}  "
          f"{row['Excess_as_pct_GDP']:.4f}%")

print('\n══════════════════════════════════════════════════════════════════')
print(f'  All outputs saved to: {out_dir}')
print('══════════════════════════════════════════════════════════════════')
