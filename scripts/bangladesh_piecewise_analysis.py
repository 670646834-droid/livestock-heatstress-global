#!/usr/bin/env python3
"""
A-1 tropical validation: Bangladesh dose-response piecewise analysis.
Pure numpy/pandas (no scipy). Tests slope change between THI ranges.
"""
import pandas as pd
import numpy as np
import os

BASE = r'F:\20261011\畜牧行业热应激\tropical_raw'
OUT = r'F:\20261011\project\02_Livestock_HeatStress\data_prep'
os.makedirs(OUT, exist_ok=True)

# Load data
df_phys = pd.read_excel(os.path.join(BASE, 'BD_2d6r7j7rgf_physiological_responses.xlsx'))
df_dmi = pd.read_excel(os.path.join(BASE, 'BD_954f6g36sb_DMI,_milk_yield_and_composition.xlsx'))

# Standardize labels
thi_map = {'T0 (≤75)': 'T0', 'T1 (75-80)': 'T1', 'T2 (≥80)': 'T2',
           'T0': 'T0', 'T1': 'T1', 'T2': 'T2'}
df_phys['THI'] = df_phys['THI Range'].map(thi_map)
df_dmi['THI'] = df_dmi['THI Range'].map(thi_map)

breed_map_phys = {'Local': 'Local', 'HF_50': 'HF50', 'HF_62.5': 'HF62.5', 'HF_75': 'HF75', 'HF_87.5': 'HF87.5'}
breed_map_dmi = {'Local': 'Local', 'HF50': 'HF50', 'HF62.5': 'HF62.5', 'HF75': 'HF75', 'HF87.5': 'HF87.5'}
df_phys['Breed'] = df_phys['Genetic group'].map(breed_map_phys)
df_dmi['Breed'] = df_dmi['Genetic Group'].map(breed_map_dmi)

# THI midpoints
thi_mid = {'T0': 72.0, 'T1': 77.5, 'T2': 82.5}
df_phys['THI_mid'] = df_phys['THI'].map(thi_mid)
df_dmi['THI_mid'] = df_dmi['THI'].map(thi_mid)

variables = [
    ('Rectal Temp (F)', df_phys, 'rectal_temp'),
    ('Respiration Rate (bpm)', df_phys, 'resp_rate'),
    ('DMI (kg)', df_dmi, 'dmi'),
    ('Milk Yield (L/day/cow)', df_dmi, 'milk_yield'),
]

breeds = ['Local', 'HF50', 'HF62.5', 'HF75', 'HF87.5']

def slope_test(df, var_name, breed_label='All'):
    """Test if slope changes between T0->T1 and T1->T2."""
    means = df.groupby('THI')[var_name].agg(['mean', 'std', 'count'])
    means = means.reindex(['T0', 'T1', 'T2'])
    
    # Slopes per THI unit
    d_thi_1 = thi_mid['T1'] - thi_mid['T0']  # 5.5
    d_thi_2 = thi_mid['T2'] - thi_mid['T1']  # 5.0
    slope1 = (means.loc['T1', 'mean'] - means.loc['T0', 'mean']) / d_thi_1
    slope2 = (means.loc['T2', 'mean'] - means.loc['T1', 'mean']) / d_thi_2
    
    # Linear slope (overall)
    x = np.array([thi_mid[t] for t in ['T0', 'T1', 'T2']])
    y = means['mean'].values
    linear_slope = np.polyfit(x, y, 1)[0]
    
    # R2 for linear
    y_pred = np.polyval(np.polyfit(x, y, 1), x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    linear_r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    
    # Slope ratio (piecewise indicator): |slope2/slope1|
    if abs(slope1) > 1e-10:
        slope_ratio = slope2 / slope1
    else:
        slope_ratio = np.nan
    
    # Saturation indicator: |slope2| < |slope1| (approaching plateau)
    saturation = abs(slope2) < abs(slope1)
    
    # t-test for slope difference (using standard errors)
    se1 = np.sqrt(means.loc['T0', 'std']**2/means.loc['T0', 'count'] + 
                   means.loc['T1', 'std']**2/means.loc['T1', 'count']) / d_thi_1
    se2 = np.sqrt(means.loc['T1', 'std']**2/means.loc['T1', 'count'] + 
                   means.loc['T2', 'std']**2/means.loc['T2', 'count']) / d_thi_2
    se_diff = np.sqrt(se1**2 + se2**2)
    t_stat = (slope2 - slope1) / se_diff if se_diff > 0 else 0
    # Approximate p-value (two-tailed, using normal approx)
    p_value = 2 * (1 - 0.5 * (1 + np.sign(t_stat) * (1 - np.exp(-t_stat**2/2))))
    
    return {
        'breed': breed_label,
        'T0_mean': means.loc['T0', 'mean'],
        'T1_mean': means.loc['T1', 'mean'],
        'T2_mean': means.loc['T2', 'mean'],
        'slope_T0_T1': slope1,
        'slope_T1_T2': slope2,
        'linear_slope': linear_slope,
        'linear_r2': linear_r2,
        'slope_ratio': slope_ratio,
        'saturation': saturation,
        't_stat': t_stat,
        'p_approx': p_value,
        'piecewise_evidence': abs(t_stat) > 1.96,  # p<0.05
    }

all_results = []

for var_name, df, var_short in variables:
    print(f'\n{"="*70}')
    print(f'Variable: {var_name}')
    print(f'{"="*70}')
    
    # Overall
    res = slope_test(df, var_name, 'All')
    res['variable'] = var_short
    all_results.append(res)
    print(f'  All:    s1={res["slope_T0_T1"]:+.4f} s2={res["slope_T1_T2"]:+.4f} '
          f'ratio={res["slope_ratio"]:+.2f} sat={res["saturation"]} t={res["t_stat"]:+.2f} pw={res["piecewise_evidence"]}')
    
    # By breed
    for breed in breeds:
        df_b = df[df['Breed'] == breed]
        if len(df_b) < 10:
            continue
        res = slope_test(df_b, var_name, breed)
        res['variable'] = var_short
        all_results.append(res)
        print(f'  {breed:8s}: s1={res["slope_T0_T1"]:+.4f} s2={res["slope_T1_T2"]:+.4f} '
              f'ratio={res["slope_ratio"]:+.2f} sat={res["saturation"]} t={res["t_stat"]:+.2f} pw={res["piecewise_evidence"]}')

# Save
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUT, 'bangladesh_piecewise_slope_test.csv'), index=False)
print(f'\nResults saved: {os.path.join(OUT, "bangladesh_piecewise_slope_test.csv")}')

# Summary
print(f'\n{"="*70}')
print('SUMMARY: Piecewise evidence by variable')
print(f'{"="*70}')
summary = results_df.groupby('variable').agg(
    n_tests=('breed', 'count'),
    pw_evidence_count=('piecewise_evidence', 'sum'),
    saturation_count=('saturation', 'sum'),
    mean_abs_t=('t_stat', lambda x: np.mean(np.abs(x))),
).round(3)
print(summary.to_string())

# Breed comparison: Local vs HF87.5
print(f'\n{"="*70}')
print('BREED COMPARISON: Local (tropical zebu) vs HF87.5 (temperate)')
print(f'{"="*70}')
for var_short in ['rectal_temp', 'resp_rate', 'dmi', 'milk_yield']:
    local = results_df[(results_df['variable']==var_short) & (results_df['breed']=='Local')]
    hf875 = results_df[(results_df['variable']==var_short) & (results_df['breed']=='HF87.5')]
    if len(local) > 0 and len(hf875) > 0:
        l_s1 = local['slope_T0_T1'].values[0]
        l_s2 = local['slope_T1_T2'].values[0]
        h_s1 = hf875['slope_T0_T1'].values[0]
        h_s2 = hf875['slope_T1_T2'].values[0]
        print(f'  {var_short:15s}: Local s1={l_s1:+.4f} s2={l_s2:+.4f} | HF87.5 s1={h_s1:+.4f} s2={h_s2:+.4f}')

print('\nBangladesh piecewise analysis complete.')
