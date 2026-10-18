import pandas as pd

df = pd.read_csv(r'F:\20261011\畜牧行业热应激\tropical_raw\CY_kouloumbris_farm_data.csv')
print(f'Total rows: {len(df)}')
print(f'Animals: {df["Animal Number"].nunique()}')
print(f'Date range: {df["Date"].min()} to {df["Date"].max()}')
print(f'THI_max range: {df["THI_max"].min():.1f} to {df["THI_max"].max():.1f}')
print(f'THI_max quantiles: {df["THI_max"].quantile([0.1,0.25,0.5,0.75,0.9]).round(1).to_dict()}')
print(f'Day Production range: {df["Day Production"].min():.2f} to {df["Day Production"].max():.2f}')
valid = df[df["THI_max"].notna() & df["Day Production"].notna()]
print(f'Valid records (both THI and production): {len(valid)}')
print()
df['THI_bin'] = pd.cut(df['THI_max'], bins=[0,68,72,75,78,82,90,100])
print('Records by THI bin:')
print(df.groupby('THI_bin')['Day Production'].agg(['count','mean']).round(2).to_string())
