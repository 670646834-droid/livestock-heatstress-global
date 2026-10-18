import pandas as pd
import os

base = r'F:\20261011\畜牧行业热应激\tropical_raw'

# Physiological responses
df1 = pd.read_excel(os.path.join(base, 'BD_2d6r7j7rgf_physiological_responses.xlsx'))
print('=== Physiological responses ===')
print(f'THI Ranges: {sorted(df1["THI Range"].unique())}')
print(f'Genetic groups: {df1["Genetic group"].unique()}')
print(f'Animals: {df1["Animal ID"].nunique()}')
print()
print('Mean by THI range:')
print(df1.groupby('THI Range')[['Rectal Temp (F)','Pulse Rate (bpm)','Respiration Rate (bpm)']].mean().round(2).to_string())

print()
print('=== DMI and Milk Yield ===')
df2 = pd.read_excel(os.path.join(base, 'BD_954f6g36sb_DMI,_milk_yield_and_composition.xlsx'))
print(f'THI Ranges: {sorted(df2["THI Range"].unique())}')
print(f'Genetic groups: {df2["Genetic Group"].unique()}')
print()
print('Mean by THI range:')
print(df2.groupby('THI Range')[['DMI (kg)','Milk Yield (L/day/cow)']].mean().round(3).to_string())
