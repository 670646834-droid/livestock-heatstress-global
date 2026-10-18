#!/usr/bin/env python3
"""Quick benchmark: cattle THI>=68.8, >=30 days/year, head-weighted."""
import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling

print("Loading THI...")
ds = xr.open_dataset("data/era5_nasapower/thi_daily_2000-2024_0.5deg.nc")
thi = ds["thi"]

print("Loading GLW4 cattle...")
with rasterio.open("data/glw4/cattle_2020_10km.tif") as src:
    glw = src.read(1).astype(np.float64)
    glw[glw < 0] = 0
    glw_transform = src.transform
    glw_crs = src.crs

# Reproject GLW to THI grid (average density)
dst_shape = (len(thi.lat), len(thi.lon))
lats = thi.lat.values
lons = thi.lon.values
lat_res = abs(float(lats[1]) - float(lats[0]))
lon_res = abs(float(lons[1]) - float(lons[0]))
dst_transform = rasterio.transform.from_bounds(
    float(lons.min()) - lon_res/2, float(lats.min()) - lat_res/2,
    float(lons.max()) + lon_res/2, float(lats.max()) + lat_res/2,
    dst_shape[1], dst_shape[0]
)
glw_reproj = np.zeros(dst_shape, dtype=np.float64)
reproject(source=glw, destination=glw_reproj,
          src_transform=glw_transform, src_crs=glw_crs,
          dst_transform=dst_transform, dst_crs="EPSG:4326",
          resampling=Resampling.average)
lats_2d = np.broadcast_to(lats[:, None], dst_shape)
pixel_area = (111.32 * 0.5) * (111.32 * 0.5 * np.cos(np.radians(lats_2d)))
headcount = glw_reproj * pixel_area
total_head = np.nansum(headcount[headcount > 0])
print(f"Total cattle: {total_head/1e9:.2f}B")

# Compute exposure for each year
threshold = 68.8
min_days = 30

print(f"\n=== Cattle THI>={threshold}, >={min_days} days/year ===")
print(f"North 2023 target: ~80%, acceptance 75%-85%")
print("-" * 50)

for year in range(2000, 2025):
    thi_year = thi.sel(time=str(year))
    # exposure_days: (lat, lon)
    exp_days = (thi_year > threshold).sum(dim="time").values
    # exposed: >=30 days
    exposed_mask = (exp_days >= min_days) & (headcount > 0)
    exposed_head = np.nansum(headcount[exposed_mask])
    frac = exposed_head / total_head * 100
    mean_days = np.nansum(exp_days[headcount > 0] * headcount[headcount > 0]) / total_head
    status = "PASS" if 75 <= frac <= 85 else "FAIL"
    print(f"  {year}: {frac:.1f}% exposed, mean {mean_days:.1f} days [{status}]")

# Also test >=1 day for comparison
print(f"\n=== Cattle THI>={threshold}, >=1 day/year (for reference) ===")
for year in [2023, 2024]:
    thi_year = thi.sel(time=str(year))
    exp_days = (thi_year > threshold).sum(dim="time").values
    exposed_mask = (exp_days >= 1) & (headcount > 0)
    exposed_head = np.nansum(headcount[exposed_mask])
    frac = exposed_head / total_head * 100
    print(f"  {year}: {frac:.1f}% exposed (>=1 day)")

ds.close()
print("\nDone.")
