#!/usr/bin/env python3
"""
B方案: Global Physical Inadaptability Area (PIA) calculation.

PIA = areas where even with optimal cooling equipment, THI_in >= 72
for more than X% of days in a year.

Cooling effectiveness constrained by RH:
- Low RH (<45%): full efficiency (ΔT_max = 10°C)
- Medium RH (45-55%): marginalized (ΔT = 6°C)
- High RH (>55%): <5°C cooling (ΔT = 3.5°C)

Using latitude as RH proxy (tropical=high, subtropical=medium, temperate=low).

Usage: python b_global_pia.py --year 2024
"""
import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
import os
import csv
import argparse

THI_FILE = "data/era5_nasapower/thi_daily_2000-2024_0.5deg.nc"
GLW_DIR = "data/glw4"
OUTDIR = "data_prep"

SPECIES = ["cattle", "pigs", "chickens", "sheep", "goats", "buffalo"]

# Cooling capacity by RH zone (°C reduction, optimal equipment = tunnel ventilation)
COOLING_CAPACITY = {
    "low": 10.0,     # RH < 45%: full evaporative efficiency
    "medium": 6.0,   # RH 45-55%: marginalized
    "high": 3.5,     # RH > 55%: <5°C cooling
}

# PIA threshold: % of days with THI_in >= 72 even after optimal cooling
PIA_THRESHOLDS = [25, 50, 75]  # three tiers

# THI threshold for heat stress (dairy cattle)
THI_THRESHOLD = 72.0


def classify_rh_zone(lats):
    """Classify RH zone using latitude as proxy."""
    abs_lat = np.abs(lats)
    rh_zone = np.empty_like(abs_lat, dtype='<U10')
    rh_zone[abs_lat < 23.5] = 'high'
    rh_zone[(abs_lat >= 23.5) & (abs_lat < 35)] = 'medium'
    rh_zone[abs_lat >= 35] = 'low'
    return rh_zone


def load_glw_headcount(glw_dir, species, dst_lats, dst_lons):
    """Load GLW4 headcount, reprojected to THI grid."""
    fpath = os.path.join(glw_dir, f"{species}_2020_10km.tif")
    with rasterio.open(fpath) as src:
        data = src.read(1).astype(np.float64)
        data[data < 0] = 0
        dst_shape = (len(dst_lats), len(dst_lons))
        lat_res = abs(float(dst_lats[1]) - float(dst_lats[0]))
        lon_res = abs(float(dst_lons[1]) - float(dst_lons[0]))
        dst_transform = rasterio.transform.from_bounds(
            float(dst_lons.min()) - lon_res/2, float(dst_lats.min()) - lat_res/2,
            float(dst_lons.max()) + lon_res/2, float(dst_lats.max()) + lat_res/2,
            dst_shape[1], dst_shape[0]
        )
        reprojected = np.zeros(dst_shape, dtype=np.float64)
        reproject(source=data, destination=reprojected,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=dst_transform, dst_crs="EPSG:4326",
                  resampling=Resampling.average)
        lats_2d = np.broadcast_to(dst_lats[:, None], dst_shape)
        pixel_area = (111.32 * 0.5) * (111.32 * 0.5 * np.cos(np.radians(lats_2d)))
        return reprojected * pixel_area


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()
    year = args.year

    print("=" * 70)
    print(f"B方案: Global Physical Inadaptability Area (PIA) - {year}")
    print("=" * 70)

    # Load THI
    print(f"\nLoading THI: {THI_FILE}")
    ds = xr.open_dataset(THI_FILE)
    thi = ds["thi"]
    lats = thi.lat.values
    lons = thi.lon.values
    thi_year = thi.sel(time=str(year)).values  # (days, lat, lon)
    n_days = thi_year.shape[0]
    print(f"  THI shape: {thi_year.shape}, days={n_days}")

    # RH zone classification
    rh_zone_1d = classify_rh_zone(lats)
    rh_zone_2d = np.broadcast_to(rh_zone_1d[:, None], (len(lats), len(lons)))

    # Cooling capacity map
    cooling_capacity = np.zeros((len(lats), len(lons)), dtype=np.float64)
    for zone, cap in COOLING_CAPACITY.items():
        cooling_capacity[rh_zone_2d == zone] = cap

    # Calculate THI after optimal cooling
    # THI_in = THI_out - cooling_capacity (broadcast over days)
    thi_cooled = thi_year - cooling_capacity[np.newaxis, :, :]

    # Days with THI_cooled >= threshold (after optimal cooling)
    days_above = np.sum(thi_cooled >= THI_THRESHOLD, axis=0)  # (lat, lon)
    pct_days_above = days_above / n_days * 100  # %

    # PIA classification
    pia_25 = (pct_days_above >= 25).astype(np.int8)
    pia_50 = (pct_days_above >= 50).astype(np.int8)
    pia_75 = (pct_days_above >= 75).astype(np.int8)

    # Load GLW4 for headcount-weighted statistics
    print("\nLoading GLW4 headcount...")
    glw_data = {}
    for species in SPECIES:
        glw_data[species] = load_glw_headcount(GLW_DIR, species, lats, lons)

    # Global cattle total
    cattle_total = np.nansum(glw_data['cattle'])
    print(f"  Total cattle: {cattle_total/1e9:.2f}B head")

    # PIA statistics (cattle headcount-weighted)
    print(f"\nPIA Statistics (cattle headcount-weighted, THI threshold={THI_THRESHOLD}):")
    print(f"  {'PIA Tier':<15s} {'Cattle in PIA':>15s} {'% of global':>12s} {'Land area':>12s}")
    results = []
    for tier, pia_mask in [(">=25% days", pia_25), (">=50% days", pia_50), (">=75% days", pia_75)]:
        cattle_in_pia = np.nansum(glw_data['cattle'][pia_mask > 0])
        pct_cattle = cattle_in_pia / cattle_total * 100
        land_area = np.sum(pia_mask > 0) * 0.25  # 0.5° x 0.5° = 0.25 deg²
        print(f"  {tier:<15s} {cattle_in_pia/1e9:>14.2f}B {pct_cattle:>11.1f}% {land_area:>11.0f} deg²")
        results.append({
            'pia_tier': tier,
            'cattle_in_pia': cattle_in_pia,
            'pct_global_cattle': pct_cattle,
            'land_area_deg2': land_area,
        })

    # By climate zone
    print(f"\nPIA by climate zone (cattle headcount-weighted, >=50% days tier):")
    zones = ['tropical', 'subtropical', 'temperate', 'boreal']
    for z in zones:
        if z == 'tropical':
            mask_lat = np.abs(lats) < 23.5
        elif z == 'subtropical':
            mask_lat = (np.abs(lats) >= 23.5) & (np.abs(lats) < 35)
        elif z == 'temperate':
            mask_lat = (np.abs(lats) >= 35) & (np.abs(lats) < 50)
        else:
            mask_lat = np.abs(lats) >= 50
        mask_2d = np.broadcast_to(mask_lat[:, None], glw_data['cattle'].shape)
        zone_cattle = np.nansum(glw_data['cattle'][mask_2d])
        zone_pia_cattle = np.nansum(glw_data['cattle'][mask_2d & (pia_50 > 0)])
        pct = zone_pia_cattle / zone_cattle * 100 if zone_cattle > 0 else 0
        print(f"  {z:<12s}: cattle={zone_cattle/1e9:.2f}B, in PIA={zone_pia_cattle/1e9:.2f}B ({pct:.1f}%)")

    # All species summary
    print(f"\nPIA by species (>=50% days tier):")
    for species in SPECIES:
        sp_total = np.nansum(glw_data[species])
        sp_pia = np.nansum(glw_data[species][pia_50 > 0])
        pct = sp_pia / sp_total * 100 if sp_total > 0 else 0
        print(f"  {species:<12s}: total={sp_total/1e9:.2f}B, in PIA={sp_pia/1e9:.2f}B ({pct:.1f}%)")

    # Save NetCDF
    out_nc = os.path.join(OUTDIR, f"b_global_pia_{year}.nc")
    ds_out = xr.Dataset(
        {
            'cooling_capacity': (['lat', 'lon'], cooling_capacity.astype(np.float32)),
            'pct_days_above_after_cooling': (['lat', 'lon'], pct_days_above.astype(np.float32)),
            'pia_25pct': (['lat', 'lon'], pia_25.astype(np.int8)),
            'pia_50pct': (['lat', 'lon'], pia_50.astype(np.int8)),
            'pia_75pct': (['lat', 'lon'], pia_75.astype(np.int8)),
            'cattle_head': (['lat', 'lon'], glw_data['cattle'].astype(np.float32)),
        },
        coords={'lat': lats, 'lon': lons},
        attrs={
            'description': 'B方案: Global Physical Inadaptability Area (PIA)',
            'year': year,
            'thi_threshold': THI_THRESHOLD,
            'cooling_assumption': 'Optimal equipment = tunnel ventilation, capacity constrained by RH zone',
            'cooling_capacity_low_rh': f'{COOLING_CAPACITY["low"]}°C',
            'cooling_capacity_medium_rh': f'{COOLING_CAPACITY["medium"]}°C',
            'cooling_capacity_high_rh': f'{COOLING_CAPACITY["high"]}°C',
            'rh_proxy': 'latitude (tropical=high RH, subtropical=medium, temperate=low)',
            'thi_source': 'NASA POWER daily, 0.5deg',
            'glw_source': 'GLW4 2020, 10km',
        }
    )
    ds_out.to_netcdf(out_nc)
    print(f"\nNetCDF saved: {out_nc}")

    # Save CSV
    out_csv = os.path.join(OUTDIR, f"b_global_pia_summary_{year}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV saved: {out_csv}")

    ds.close()
    print("\nB方案 global PIA calculation complete.")


if __name__ == "__main__":
    main()
