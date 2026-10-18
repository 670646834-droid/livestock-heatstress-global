#!/usr/bin/env python3
"""
Brazil JDD dataset analysis: municipal annual milk production vs THI.
Data: 17,322 rows, MinasGerais state, 2004-2023.
Match with NASA POWER THI (0.5deg) and test piecewise dose-response.
"""
import pandas as pd
import numpy as np
import xarray as xr
from scipy import stats
import os

DATA_DIR = '/mnt/disk/GBC/livestock_heatstress'
JDD_FILE = '/mnt/disk/GBC/livestock_heatstress/data/BR_MinasGerais_JDD1_dataset.csv'
THI_FILE = os.path.join(DATA_DIR, 'data', 'era5_nasapower', 'thi_daily_2000-2024_0.5deg.nc')
OUT_DIR = os.path.join(DATA_DIR, 'data_prep')

print('=== Step 1: Load and explore JDD data ===')
df = pd.read_csv(JDD_FILE)
print(f'  Shape: {df.shape}')
print(f'  Columns: {list(df.columns)}')
print(f'  Years: {df["Year"].min()} - {df["Year"].max()}')
print(f'  Unique municipalities: {df["Municipality"].nunique()}')
print(f'  Unique regions: {df["Region"].nunique()}')
print(f'  Productivity range: {df["Productivity"].min():.0f} - {df["Productivity"].max():.0f} L/cow/year')
print(f'  Productivity mean: {df["Productivity"].mean():.0f} L/cow/year')

# MinasGerais bounding box: -22.9 to -14.2 lat, -51.0 to -39.9 lon
print(f'\n  MinasGerais is in tropical/subtropical zone (lat -14 to -23)')

print('\n=== Step 2: Load NASA POWER THI for MinasGerais ===')
ds = xr.open_dataset(THI_FILE)
# MinasGerais bounding box
lat_min, lat_max = -23.0, -14.0
lon_min, lon_max = -51.5, -39.5
thi_mg = ds['thi'].sel(
    lat=slice(lat_max, lat_min),  # lat is descending
    lon=slice(lon_min, lon_max)
)
print(f'  THI MinasGerais shape: {thi_mg.shape}')
print(f'  THI range: {thi_mg.min().values:.1f} - {thi_mg.max().values:.1f}')

# Compute annual mean THI and heat stress days for each grid cell
years = np.arange(2004, 2024)
thi_annual = []
for year in years:
    year_data = thi_mg.sel(time=str(year))
    annual_mean = year_data.mean(dim='time').values
    days_ge68 = (year_data >= 68).sum(dim='time').values
    days_ge72 = (year_data >= 72).sum(dim='time').values
    thi_annual.append({
        'year': year,
        'annual_mean_thi': annual_mean,
        'days_ge68': days_ge68,
        'days_ge72': days_ge72,
        'lats': thi_mg.lat.values,
        'lons': thi_mg.lon.values
    })

print(f'  Computed annual THI for {len(years)} years')

# Get grid cell coordinates
lats = thi_mg.lat.values
lons = thi_mg.lon.values
print(f'  Grid: {len(lats)} lat x {len(lons)} lon = {len(lats)*len(lons)} cells')

# For each municipality, find nearest grid cell and extract THI
# Need municipality coordinates - use approximate from IBGE or simple lookup
# For now, use region-level aggregation
print('\n=== Step 3: Region-level analysis ===')
# Aggregate JDD by region and year
df_region = df.groupby(['Region', 'Year']).agg({
    'Milk_Production': 'sum',
    'Milked_Cows': 'sum',
    'Productivity': 'mean'
}).reset_index()
df_region['productivity_calc'] = df_region['Milk_Production'] / df_region['Milked_Cows']
print(f'  Region-year observations: {len(df_region)}')
print(f'  Regions: {df_region["Region"].unique()}')

# For each region, compute average THI from grid cells in that region
# Approximate region coordinates (need IBGE data, use rough estimates)
region_coords = {
    'UBERLANDIA': (-18.9, -48.3),
    'UBERABA': (-19.7, -47.9),
    'PATOS DE MINAS': (-18.6, -46.5),
    'BELO HORIZONTE': (-19.9, -43.9),
    'JUIZ DE FORA': (-21.8, -43.3),
    'POCOS DE CALDAS': (-21.8, -46.6),
    'Varginha': (-21.6, -45.4),
    'PASSOS': (-20.7, -46.6),
}

# Match each region-year with THI
matched_data = []
for _, row in df_region.iterrows():
    region = row['Region']
    year = row['Year']
    if region in region_coords:
        rlat, rlon = region_coords[region]
        # Find nearest grid cell
        lat_idx = np.argmin(np.abs(lats - rlat))
        lon_idx = np.argmin(np.abs(lons - rlon))
        year_idx = year - 2004
        if 0 <= year_idx < len(thi_annual):
            annual_mean = thi_annual[year_idx]['annual_mean_thi'][lat_idx, lon_idx]
            days_ge68 = thi_annual[year_idx]['days_ge68'][lat_idx, lon_idx]
            days_ge72 = thi_annual[year_idx]['days_ge72'][lat_idx, lon_idx]
            matched_data.append({
                'region': region,
                'year': year,
                'productivity': row['productivity_calc'],
                'annual_mean_thi': annual_mean,
                'days_ge68': days_ge68,
                'days_ge72': days_ge72,
                'milked_cows': row['Milked_Cows']
            })

df_matched = pd.DataFrame(matched_data)
print(f'  Matched observations: {len(df_matched)}')
if len(df_matched) > 0:
    print(f'  THI range: {df_matched["annual_mean_thi"].min():.1f} - {df_matched["annual_mean_thi"].max():.1f}')
    print(f'  Days>=68 range: {df_matched["days_ge68"].min():.0f} - {df_matched["days_ge68"].max():.0f}')

print('\n=== Step 4: Piecewise dose-response test ===')
if len(df_matched) > 10:
    # Test linear vs piecewise for productivity vs days_ge68
    x = df_matched['days_ge68'].values
    y = df_matched['productivity'].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    if len(x) > 10:
        # Linear model
        slope_lin, intercept_lin, r_lin, p_lin, _ = stats.linregress(x, y)
        ss_res_lin = np.sum((y - (intercept_lin + slope_lin * x))**2)

        # Piecewise model (breakpoint at median)
        bp = np.median(x)
        x1 = np.where(x < bp, x, bp)
        x2 = np.where(x >= bp, x - bp, 0)
        X = np.column_stack([np.ones_like(x), x1, x2])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ beta
        ss_res_pw = np.sum((y - y_pred)**2)

        # AIC comparison
        n = len(x)
        aic_lin = n * np.log(ss_res_lin / n) + 2 * 2
        aic_pw = n * np.log(ss_res_pw / n) + 2 * 3
        delta_aic = aic_lin - aic_pw

        r2_pw = 1 - ss_res_pw / np.sum((y - np.mean(y))**2)
        slope1 = beta[1]
        slope2 = beta[1] + beta[2]
        slope_ratio = abs(slope2 / slope1) if slope1 != 0 else np.inf

        print(f'  Linear: slope={slope_lin:.2f}, R2={r_lin**2:.3f}, p={p_lin:.4f}')
        print(f'  Piecewise: breakpoint={bp:.0f} days, slope1={slope1:.2f}, slope2={slope2:.2f}')
        print(f'  Slope ratio (high/low): {slope_ratio:.2f}')
        print(f'  R2 piecewise: {r2_pw:.3f}')
        print(f'  Delta AIC (lin - pw): {delta_aic:.2f} (positive = piecewise better)')
        print(f'  Piecewise better: {delta_aic > 2}')

        # Save results
        results = pd.DataFrame([{
            'dataset': 'Brazil_JDD_MinasGerais',
            'n_obs': n,
            'response_var': 'milk_productivity_L_cow_year',
            'exposure_var': 'days_THI>=68',
            'linear_slope': slope_lin,
            'linear_r2': r_lin**2,
            'linear_p': p_lin,
            'piecewise_breakpoint': bp,
            'piecewise_slope1': slope1,
            'piecewise_slope2': slope2,
            'piecewise_slope_ratio': slope_ratio,
            'piecewise_r2': r2_pw,
            'delta_aic': delta_aic,
            'piecewise_better': delta_aic > 2
        }])
        out_file = os.path.join(OUT_DIR, 'brazil_jdd_piecewise_results.csv')
        results.to_csv(out_file, index=False)
        print(f'\n  Results saved: {out_file}')

        # Save matched data
        df_matched.to_csv(os.path.join(OUT_DIR, 'brazil_jdd_thi_matched.csv'), index=False)
        print(f'  Matched data saved: {os.path.join(OUT_DIR, "brazil_jdd_thi_matched.csv")}')
else:
    print('  Insufficient matched data for analysis')

ds.close()
print('\nBrazil JDD analysis complete.')
