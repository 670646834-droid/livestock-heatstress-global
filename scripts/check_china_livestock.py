import pandas as pd
df = pd.read_csv(r'F:\20261011\project\02_Livestock_HeatStress\data_prep\china_analysis\china_provincial_livestock_2024.csv')
print('Columns:', df.columns.tolist())
print('Shape:', df.shape)
print()
print(df.head(10).to_string())
print()
print('=== Totals ===')
for col in df.columns:
    if col not in ['province', 'Province', '省']:
        try:
            print(f'{col}: {df[col].sum():,.0f}')
        except:
            pass
