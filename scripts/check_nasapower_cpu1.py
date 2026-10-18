#!/usr/bin/env python3
"""Check NASA POWER data structure on CPU1."""
import xarray as xr
import os

data_dir = '/mnt/disk/GBC/livestock_heatstress/data/era5_nasapower'
print('Files:', os.listdir(data_dir))

thi_file = os.path.join(data_dir, 'thi_daily_2000-2024_0.5deg.nc')
if os.path.exists(thi_file):
    ds = xr.open_dataset(thi_file)
    print('\nTHI file:')
    print(f'  Variables: {list(ds.data_vars)}')
    print(f'  Coords: {list(ds.coords)}')
    print(f'  Dims: {dict(ds.dims)}')
    for v in ds.data_vars:
        print(f'  {v}: shape={ds[v].shape}, dims={ds[v].dims}')
    ds.close()

# Check for raw T2M/RH2M files
for f in os.listdir(data_dir):
    if f.endswith('.nc') and f != 'thi_daily_2000-2024_0.5deg.nc':
        print(f'\nOther NC: {f}')
        try:
            ds = xr.open_dataset(os.path.join(data_dir, f))
            print(f'  Variables: {list(ds.data_vars)}')
            print(f'  Dims: {dict(ds.dims)}')
            ds.close()
        except Exception as e:
            print(f'  Error: {e}')
