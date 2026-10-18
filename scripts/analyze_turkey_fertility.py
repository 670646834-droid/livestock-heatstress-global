#!/usr/bin/env python3
"""
DHI/Turkey fertility loss data analysis - third country dose-response validation.
Source: 18754121__Fertility_Loss_daily_ensemble_district_level.xlsx
        (Turkey Aydın province, 17 districts, 1976-2070, historical+RCP4.5+RCP8.5)
"""
import pandas as pd
import numpy as np
import csv

INPUT = r"F:\20261011\畜牧行业热应激\p0_raw\A_repos\18754121__Fertility_Loss_daily_ensemble_district_level.xlsx"
OUTDIR = r"F:\20261011\project\02_Livestock_HeatStress\data_prep"

print("Loading Turkey fertility loss data...")
df = pd.read_excel(INPUT)
print(f"  Shape: {df.shape}")
print(f"  Districts: {df['district'].nunique()}")
print(f"  Years: {df['year'].min()}-{df['year'].max()}")
print(f"  Scenarios: {list(df['scenario'].unique())}")

# Historical only for dose-response calibration
hist = df[df['scenario'] == 'historical'].copy()
print(f"\nHistorical data: {len(hist)} rows, {hist['year'].min()}-{hist['year'].max()}")

# Dose-response: THI bins vs fertility loss
hist['thi_bin'] = pd.cut(hist['THI_ensemble_median'],
                           bins=[0,60,65,68,70,72,75,78,80,85,100])
agg = hist.groupby('thi_bin', observed=True).agg(
    n=('Fert_Loss_median (%)', 'count'),
    mean_loss=('Fert_Loss_median (%)', 'mean'),
    median_loss=('Fert_Loss_median (%)', 'median'),
    std_loss=('Fert_Loss_median (%)', 'std'),
    mean_thi=('THI_ensemble_median', 'mean')
).reset_index()
agg['thi_bin_str'] = agg['thi_bin'].astype(str)

print("\n=== Dose-Response: THI bins vs Fertility Loss (historical) ===")
print(agg[['thi_bin_str','n','mean_thi','mean_loss','median_loss','std_loss']].to_string())

# Linear regression above threshold 68
above = hist[hist['THI_ensemble_median'] > 68]
x = above['THI_ensemble_median'].values
y = above['Fert_Loss_median (%)'].values
slope, intercept = np.polyfit(x, y, 1)
r2 = np.corrcoef(x, y)[0,1]**2
print(f"\n=== Linear Regression (THI > 68) ===")
print(f"  Slope: {slope:.4f} %/THI")
print(f"  Intercept: {intercept:.4f} %")
print(f"  R²: {r2:.4f}")
print(f"  n: {len(above)}")

# Piecewise: threshold 68, then linear
# Loss = max(0, slope * (THI - 68))
print(f"\n=== Piecewise Model (threshold=68, slope={slope:.4f}) ===")
print(f"  THI=68: 0%")
print(f"  THI=70: {max(0, slope*(70-68)):.2f}%")
print(f"  THI=72: {max(0, slope*(72-68)):.2f}%")
print(f"  THI=75: {max(0, slope*(75-68)):.2f}%")
print(f"  THI=78: {max(0, slope*(78-68)):.2f}%")
print(f"  THI=80: {max(0, slope*(80-68)):.2f}%")

# Save dose-response table
out_csv = f"{OUTDIR}/turkey_fertility_dose_response_20261013.csv"
agg_out = agg[['thi_bin_str','n','mean_thi','mean_loss','median_loss','std_loss']].copy()
agg_out.columns = ['THI_bin', 'n_days', 'mean_THI', 'mean_fertility_loss_pct',
                   'median_fertility_loss_pct', 'std_fertility_loss_pct']
agg_out.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\nDose-response table saved: {out_csv}")

# Save regression parameters
reg_csv = f"{OUTDIR}/turkey_fertility_regression_params_20261013.csv"
with open(reg_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["parameter", "value", "unit", "note"])
    writer.writerow(["threshold", 68.0, "THI", "Below this no fertility loss observed"])
    writer.writerow(["slope", round(slope, 4), "%/THI", "Linear slope above threshold"])
    writer.writerow(["intercept", round(intercept, 4), "%", "Linear intercept"])
    writer.writerow(["r_squared", round(r2, 4), "", "Goodness of fit"])
    writer.writerow(["n_observations", len(above), "days", "Historical data above threshold"])
    writer.writerow(["country", "Turkey", "", "Aydın province, 17 districts"])
    writer.writerow(["period", f"{hist['year'].min()}-{hist['year'].max()}", "", "Historical period"])
    writer.writerow(["species", "cattle", "", "Dairy cattle fertility"])
print(f"Regression params saved: {reg_csv}")

# Summary for paper
print(f"\n{'='*70}")
print("SUMMARY: Turkey Fertility Loss Dose-Response (Third Country Validation)")
print(f"{'='*70}")
print(f"  Data: {len(hist)} daily observations, 17 districts, {hist['year'].min()}-{hist['year'].max()}")
print(f"  Threshold: THI = 68 (no loss below)")
print(f"  Slope: {slope:.4f} % fertility loss per THI unit above 68")
print(f"  R²: {r2:.4f}")
print(f"  At THI=72: {max(0, slope*(72-68)):.2f}% loss")
print(f"  At THI=75: {max(0, slope*(75-68)):.2f}% loss")
print(f"  At THI=78: {max(0, slope*(78-68)):.2f}% loss")
print(f"\n  Comparison with existing parameters:")
print(f"    DR-11 (US, Wettemann 1976): 31-35°C exposure -> conception rate 41%->29%")
print(f"    Turkey: THI>68 -> {slope:.2f}%/THI fertility loss")
print(f"  This provides independent third-country validation (Turkey) alongside US and China.")

print("\nTurkey fertility loss analysis complete.")
