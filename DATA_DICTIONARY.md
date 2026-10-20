# Data and code dictionary — Nature Food submission package v5 (2026-10-20)

All monetary values are in 2005 constant US dollars (USD). "B USD" = billion USD.
The manuscript's adopted internally consistent basis is the FAOSTAT-scaled 2000-2024
series at THI >= 72 (2024): gross 289.57 B USD, avoidable 27.32 B (9.4%),
residual 262.25 B (90.6%).

## 05_data

| File | Rows | Description |
|---|---|---|
| `a1_global_bias_summary_2024.csv` | 5 | Headcount-weighted dairy-cattle production-loss comparison between the threshold and linear dose-response forms, by climate zone (tropical, subtropical, temperate, boreal, global). Columns: `climate_zone`, `cattle_head`, `pw_loss_pct_weighted`, `lin_loss_pct_weighted`, `abs_bias_pp`, `rel_bias_pct`. Note: `rel_bias_pct` is a headcount-weighted mean of per-cell ratios and is dominated by cells with small denominators; the manuscript reports the aggregate relative difference computed from `pw_loss_pct_weighted` and `lin_loss_pct_weighted` (50.6% globally). |
| `b_global_pia_summary_2024.csv` | 3 | Physical incompatibility area (cattle), 2024: `pia_tier` (>=25%, >=50%, >=75% of days), `cattle_in_pia`, `pct_global_cattle`, `land_area_deg2`. |
| `china_provincial_loss_v4_elevcorr_2024.csv` | 32 | Provincial losses for China, 2024 (elevation-corrected, GLW4-2015 cattle layer): exposure days, species headcounts, species losses, total gross/avoidable/residual loss and avoidable fraction. |
| `environmental_intensity_ch4.csv` | 150 | Heat-stress-induced increase in enteric methane emission intensity, by species/year/threshold: `hw_ch4_increase_pct`, `additional_ch4_GtCO2eq` (computed against an implied total-livestock baseline of 7.19 Gt CO2eq), `mean_exposure_days`. The manuscript reports methane-specific values obtained by rescaling with 3.5/7.19 (total 0.09 Gt CO2eq/yr; 2.6% of the 3.5 Gt methane budget). The SI payback table uses the FAOSTAT-scaled basis (gross 289.57 B USD; 3/5/10-year avoidable 23.22/27.32/38.93 B USD), not the unscaled values in the payback CSVs. |
| `era5_global_sampling_sensitivity.csv` | 450 | ERA5 versus NASA POWER THI comparison at 450 stratified cells (`nasa_thi_2024`, `era5_thi_2024`, `delta_thi`, `n_days`); 100 cells have valid NASA POWER values. |
| `exposure_results_elevcorr_ge30.csv` | 600 | Daily THI exposure by year, threshold and species (headcount, exposed headcount, exposure percentage, mean days above threshold). |
| `loss_estimation_v4_endogenized_2024_china2015.csv` | 24 | Endogenized 2024 losses by species and threshold with the China-2015 cattle layer (the run that produced the originally published Table 1 values, 279.15 B USD; superseded in the revision by the FAOSTAT-scaled series below). |
| `loss_estimation_v4_endogenized_2024_payback{3,5,10}yr.csv` | 24 each | Payback-horizon sensitivity runs (unscaled GLW4 base) used only for the per-species avoidable fractions; absolute values are rescaled to the FAOSTAT-scaled 2024 gross in the manuscript. |
| `Supplementary_Data_1_dose_response_parameter_annotations.csv` | 23 | Compiled dose-response parameter entries with species, response variable, threshold, coefficient, evidence type, credibility score and literature source (column headers in Chinese; the species-level values implemented in the global model are given in Supplementary Table 4). |
| `Supplementary_Data_2_loss_estimates_2000-2024.csv` | 600 | FAOSTAT-scaled loss estimates for 2000-2024, four thresholds and six species: `herd_scaling_factor`, `total_head_scaled`, `gross_loss_usd`, `avoidable_loss_usd`, `residual_loss_usd`, `avoidable_fraction`, `mean_effectiveness`, `yield_loss_fraction`. This is the primary adopted dataset. |
| `Supplementary_Data_3_value_per_head_2005USD.csv` | 204 | FAOSTAT-derived value per head, six species, 1991-2024, 2005 constant USD. |
| `Supplementary_Data_4_nass_state_month_panel.csv` | 3,144 | Merged US NASS state-month panel used in the NASS reanalysis (source: USDA NASS Quick Stats, https://quickstats.nass.usda.gov; state-monthly mean THI from NASA POWER): `state_abbr`, `year`, `month`, `milk_production_mlbs`, `milk_cows_khead`, `milk_per_cow_lbs`, `thi_mean`. The merge of the public NASS milk-production table (3,696 rows, 2013-2026, 24 states) with the state-monthly mean THI table (2013-2025) yields 3,144 valid observations after removing missing milk-per-cow values. |
| `Supplementary_Data_5_nass_model_comparison.csv` | 12 | NASS model comparison: linear versus two-segment models under three fixed-effects specifications (`state+month+year`, `state+time`, `state+year`), with breakpoint, RSS, AIC, `delta_AIC` (standard; positive favours two-segment), `delta_AIC_text` (4/8 parameter convention used in the manuscript), pre/post slopes and P value for the slope change. Key result: the threshold-linear model (zero pre-breakpoint slope) with state+month+year fixed effects selects a breakpoint at THI 66.2 (post-threshold slope -14.82 lbs per cow per month per THI unit; delta_AIC = 49.4 with the breakpoint counted as an estimated parameter); an unconstrained two-segment fit gives 63.1 (delta_AIC = 54.8; pre-breakpoint slope +2.20, P = 0.005). A state+year variant without month fixed effects gives delta_AIC = 34.7 at THI 63.2, close to the value quoted in the v4 text, but the manuscript uses the state+month+year specification because it controls for seasonal variation.

## 06_scripts

| File | Purpose |
|---|---|
| `a1_global_bias.py` | Threshold-versus-linear comparison for dairy cattle (writes `a1_global_bias_2024.nc` and summary CSV). |
| `b_global_pia.py` | Physical incompatibility area computation (writes `b_global_pia_2024.nc` and summary CSV). |
| `loss_estimation_v4_1_scaled.py` | Main FAOSTAT-scaled loss model for 2000-2024 (writes `loss_estimation_v4_1_scaled_2000-2024.csv`). |
| `loss_estimation_v4_endogenized.py` | Endogenized adaptation model for 2024 (payback-horizon runs). |
| `us_nass_reanalysis.py` | Independent US NASS reanalysis; writes Supplementary Data 4 and 5 and reproduces the numbers in Supplementary Methods S1.9. Raw NASS and THI inputs are the public tables described in the SI (paths are set at the top of the script). |

## Notes

- The NASS raw input tables are not redistributed in this package (public sources: USDA NASS Quick Stats; NASA POWER-derived state-monthly THI). The merged panel needed to reproduce the SI analysis is provided as Supplementary Data 4.
- Scripts were run with Python 3.10 (NumPy, pandas, xarray, Rasterio) on the study's compute environment; figure scripts are available in the code repository.
