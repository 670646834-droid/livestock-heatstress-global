#!/usr/bin/env python3
import sys
print(f'Python: {sys.version}')
for mod in ['numpy', 'pandas', 'xarray', 'netCDF4', 'scipy', 'rasterio', 'requests']:
    try:
        m = __import__(mod)
        print(f'{mod}: {getattr(m, "__version__", "ok")}')
    except ImportError:
        print(f'{mod}: NOT INSTALLED')
