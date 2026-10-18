#!/usr/bin/env python3
"""Check buffalo price data in FAOSTAT for 2024 anomaly."""
import pandas as pd

prices = pd.read_csv(
    r'F:\20261011\project\02_Livestock_HeatStress\data\faostat\PP_Prices\Prices_E_All_Data_(Normalized).csv',
    encoding='utf-8', low_memory=False)
prices_usd = prices[prices['Element'] == 'Producer Price (USD/tonne)']

# Buffalo meat
bf_meat = prices_usd[prices_usd['Item'] == 'Meat of buffalo, fresh or chilled']
print('=== Buffalo Meat Prices (USD/tonne) ===')
print('  Countries with data:', bf_meat['Area'].nunique())
print('  Years:', bf_meat['Year'].min(), '-', bf_meat['Year'].max())
for yr in [2019, 2020, 2021, 2022, 2023, 2024]:
    yr_data = bf_meat[bf_meat['Year'] == yr]
    if len(yr_data) > 0:
        print(f'  {yr}: n={len(yr_data)}, mean={yr_data["Value"].mean():.0f}, '
              f'median={yr_data["Value"].median():.0f}, min={yr_data["Value"].min():.0f}, '
              f'max={yr_data["Value"].max():.0f}')
    else:
        print(f'  {yr}: NO DATA')

print()
# Buffalo milk
bf_milk = prices_usd[prices_usd['Item'] == 'Raw milk of buffalo']
print('=== Buffalo Milk Prices (USD/tonne) ===')
print('  Countries with data:', bf_milk['Area'].nunique())
for yr in [2019, 2020, 2021, 2022, 2023, 2024]:
    yr_data = bf_milk[bf_milk['Year'] == yr]
    if len(yr_data) > 0:
        print(f'  {yr}: n={len(yr_data)}, mean={yr_data["Value"].mean():.0f}, '
              f'median={yr_data["Value"].median():.0f}, min={yr_data["Value"].min():.0f}, '
              f'max={yr_data["Value"].max():.0f}')
    else:
        print(f'  {yr}: NO DATA')

# Check which countries have 2024 data
print()
print('=== Countries with buffalo meat price in 2024 ===')
bf_meat_2024 = bf_meat[bf_meat['Year'] == 2024]
if len(bf_meat_2024) > 0:
    print(bf_meat_2024[['Area', 'Value']].sort_values('Value').to_string())
else:
    print('  NO COUNTRIES')

print()
print('=== Countries with buffalo milk price in 2024 ===')
bf_milk_2024 = bf_milk[bf_milk['Year'] == 2024]
if len(bf_milk_2024) > 0:
    print(bf_milk_2024[['Area', 'Value']].sort_values('Value').to_string())
else:
    print('  NO COUNTRIES')
