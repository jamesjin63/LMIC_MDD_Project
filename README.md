# LMIC MDD Project: analysis code

This repository contains the analysis scripts associated with the manuscript **“Rising disease burden and economic welfare losses attributable to depressive and anxiety disorders among children and adolescents in low- and middle-income countries, 1990–2023.”**

## Scope

The `analysis_code/` directory contains 9 Python scripts and 1 R script for the descriptive epidemiology, welfare valuation, counterfactual analysis, manuscript figures and tables, and supplementary analyses. **No study data, source-data tables, figures, or manuscript files are included in this repository.** The scripts require input data to reproduce the outputs.

| Script | Main role |
| --- | --- |
| `run_descriptive_epi_MDD.py` | Descriptive burden, trend, age and sex analyses |
| `build_incidence_0_19.py` | Child and adolescent incidence aggregation |
| `run_2disease_VLW_MDD.R` | Welfare-loss valuation and sensitivity analyses |
| `run_covid19_counterfactual.py` | Historical-trend counterfactual analysis |
| `build_tables_T1_T2.py` | Main summary tables |
| `run_fig4_VLW_maps.py` | Welfare-loss maps |
| `run_fig5_VLW_scatter.py` | GDP and welfare-loss scatterplot |
| `run_fig6_COVID_timeseries.py` | Counterfactual time-series figure |
| `run_suppl_MS.py` | Supplementary figures and tables |
| `rebuild_suppl_121.py` | Supplementary rebuild for the 121-country analysis set |

## Data access

The analysis inputs and processed source data are not published in this repository. Publicly available source estimates can be obtained from the Institute for Health Metrics and Evaluation's Global Burden of Disease 2023 resources and the World Bank World Development Indicators, subject to their respective terms. Editors and reviewers may request the processed study data needed to evaluate the manuscript by emailing the corresponding author, **Jinxin Zheng** (jamesjin63@163.com).

## Running the scripts

The Python scripts use relative paths based on their location; the R script expects its input CSVs in the working directory. They are preserved in the layout used for the manuscript analysis. After obtaining the required input data, place them in the paths named near the top of each script before running it. The main Python dependencies include pandas, NumPy, Matplotlib, GeoPandas, SciPy and statsmodels. The R analysis uses tidyverse, sf, patchwork and scales.

No data are bundled with the code, and the scripts have not been packaged as a one-command pipeline. Please contact the corresponding author for the matching input layout and access to the processed data.
