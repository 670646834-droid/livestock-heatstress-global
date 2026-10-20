# Global livestock heat-stress losses with endogenized adaptation feasibility (2000-2024)

Code and data repository accompanying the manuscript:

> Gao, B., Xiao, M., Sun, M., Xing, J., Wang, Y., Zhou, L., Kang, D., Hu, J., Mahfuz, S., Zhang, Q. & Zhang, J.
> *Cost-effective adaptation can avert less than one-tenth of global livestock heat-stress losses.*
> Manuscript prepared for **Nature Food** (2026).

## Overview

This repository contains the analysis code and derived data for a global assessment of livestock
heat-stress losses and the share that cost-effective adaptation can avoid. The model combines:

1. a threshold-linear dose-response for six livestock species (cattle, buffalo, sheep, goats, pigs, chickens),
2. an adaptation layer with six cooling measures, humidity-dependent efficiency, capital constraints and payback criteria,
3. an environmental intensity account linking heat-induced production losses to enteric methane emissions,
4. validation against independent farm, regional and national datasets (Cyprus dairy records, Bangladeshi
   physiological measurements, Turkish fertility records and a US NASS state-month dairy panel).

## Headline numbers (2024, THI >= 72, FAOSTAT-scaled, 5-year payback)

| Quantity | Value |
|---|---|
| Gross heat-stress losses | US$289.57 billion (2005 constant prices) |
| Avoidable through cost-effective adaptation | US$27.32 billion (9.4%) |
| Residual losses | US$262.25 billion (90.6%) |
| Avoidable share across 3/5/10-year payback | 8.0% / 9.4% / 13.4% (US$23.22 / 27.32 / 38.93 billion) |
| Increase in enteric methane intensity | 0.8-3.3% across species; +0.09 Gt CO2eq per year (2.6% of livestock methane) |
| Growth in gross losses, 2000-2024 | +47.6% |

The US NASS state-month panel (n = 3,144, 24 states, 2013-2025) supports a threshold response under a
threshold-linear model with state, month and year fixed effects: breakpoint THI 66.2, post-threshold slope
-14.8 lbs per cow per month per THI unit, delta-AIC = 49.4. An unconstrained two-segment fit gives THI 63.1
and delta-AIC = 54.8 (sensitivity). The lower breakpoint than in the daily-maximum datasets is expected
because monthly mean THI is used. See `DATA_DICTIONARY.md` and the manuscript's Supplementary
Methods S1.9 for details.

## Repository contents

```
data/                 derived data files (Supplementary Data 1-5 and core model outputs)
scripts/              core model and analysis scripts
figure_scripts/       scripts that generate the main and supplementary figures
DATA_DICTIONARY.md    variables, units, sources and reproduction notes
README.md             this file
LICENSE               MIT licence for code
DATA_LICENSE.md       CC BY 4.0 licence for data
CITATION.cff          citation metadata
.zenodo.json          Zenodo metadata
```

Key files:

- `data/loss_estimation_v4_1_elevcorr_2000-2024.csv` (also distributed as `Supplementary_Data_2_loss_estimates_2000-2024.csv`):
  the adopted FAOSTAT-scaled loss series by species, threshold and year.
- `data/Supplementary_Data_4_nass_state_month_panel.csv`: merged US NASS state-month panel (3,144 observations).
- `data/Supplementary_Data_5_nass_model_comparison.csv`: linear vs threshold-linear vs unconstrained
  two-segment models across fixed-effects specifications.
- `scripts/loss_estimation_v4_1_scaled.py`: main loss model that produces the adopted series.
- `scripts/loss_estimation_v4_endogenized.py`: adaptation/payback model.
- `scripts/us_nass_reanalysis.py`: reproduces the NASS panel and model comparison (Data 4-5, SI S1.9).

## Reproducing the analysis

The scripts were run with Python 3.10 (NumPy, pandas, xarray, Rasterio). Input datasets are public:

- Gridded Livestock of the World v4 (FAO Geospatial Data Catalog)
- NASA POWER meteorology (and ERA5 via the Open-Meteo archive for the sampling sensitivity)
- FAOSTAT prices and herd statistics
- ETOPO1 topography and GADM administrative boundaries
- USDA NASS Quick Stats (milk production) and state-monthly mean THI for the NASS reanalysis

Raw gridded inputs are large and are not redistributed here; the derived data needed to reproduce the
manuscript's figures and tables are included. The NASS reanalysis can be reproduced from
`data/Supplementary_Data_4_nass_state_month_panel.csv` with `scripts/us_nass_reanalysis.py` once the
two public input tables are placed at the paths configured at the top of the script.

## Data and code availability

- Archived data and code (latest version v5.0, 2026-10-20): **Zenodo**, https://doi.org/10.5281/zenodo.23489005
- Concept DOI (always resolves to the latest version): https://doi.org/10.5281/zenodo.23476982
- Licence: CC BY 4.0
- Code repository: this GitHub repository
- Internal review history and the revised manuscript package are available from the corresponding authors.

## Licence

- Code: MIT (see `LICENSE`)
- Data: Creative Commons Attribution 4.0 International (CC BY 4.0; see `DATA_LICENSE.md`)

## Contact

Corresponding authors: Shad Mahfuz (Shadmahfuz@yahoo.com), Qianqian Zhang (zhangqianqian@lyu.edu.cn),
Jianying Zhang (zhangjianying@lyu.edu.cn).
