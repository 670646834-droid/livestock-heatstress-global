#!/usr/bin/env python3
"""
A-1: Global bias calculation - linear vs piecewise dose-response.

Core finding from design doc: linear dose-response systematically overstates
losses in temperate regions (+31.7%) vs tropical (+3.1%).

This script extends the 4-climate-zone calculation to global grid cells,
using GLW4 cattle headcount weighting and NASA POWER daily THI.

Piecewise function (Zhengkun, dairy cattle):
  THI < 71:  slope = -0.004 kg/THI  (shallow zone, near-zero loss)
  71 <= THI < 80: slope = -0.393 kg/THI  (moderate-steep)
  THI >= 80: slope = -0.933 kg/THI  (steep zone)

Linear function:
  slope = -0.423 kg/THI (full-range OLS)

Baseline yield: 32 kg/head/day (for % conversion)

Usage: python a1_global_bias.py --year 2024
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

# Piecewise parameters (Zhengkun 58-point curve, dairy cattle)
BREAKPOINT_1 = 71.0
BREAKPOINT_2 = 80.0
SLOPE_SHALLOW = 0.004   # kg/THI (near-zero, actually +0.021 in 68-71 but use 0.004 for 71+)
SLOPE_MODERATE = 0.393  # kg/THI
SLOPE_STEEP = 0.933     # kg/THI
LINEAR_SLOPE = 0.423    # kg/THI (full-range OLS)
BASELINE_YIELD = 32.0   # kg/head/day
THRESHOLD_LINEAR = 68.0 # linear model starts counting from THI 68
THRESHOLD_PIECEWISE = 71.0  # piecewise starts from breakpoint 1

# Climate zone classification by latitude
def climate_zone(lat):
    abs_lat = abs(lat)
    if abs_lat < 23.5:
        return 'tropical'
    elif abs_lat < 35:
        return 'subtropical'
    elif abs_lat < 50:
        return 'temperate'
    else:
        return 'boreal'


def load_glw_cattle(glw_dir, dst_lats, dst_lons):
    """Load GLW4 cattle headcount, reprojected to THI grid."""
    fpath = os.path.join(glw_dir, "cattle_2020_10km.tif")
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
        headcount = reprojected * pixel_area
        return headcount


def piecewise_loss(thi_daily):
    """
    Calculate cumulative yield loss using piecewise function.
    Returns loss in kg/head/year (sum over all days).
    """
    loss = np.zeros_like(thi_daily, dtype=np.float64)
    # Shallow zone: THI >= 71 but < 80
    mask1 = (thi_daily >= THRESHOLD_PIECEWISE) & (thi_daily < BREAKPOINT_2)
    loss[mask1] += (thi_daily[mask1] - THRESHOLD_PIECEWISE) * SLOPE_SHALLOW
    # Moderate zone: THI >= 71 but < 80, use moderate slope for the part above 71
    # Actually piecewise: from 71 to 80, slope = 0.393; above 80, slope = 0.933
    # Recalculate properly:
    loss = np.zeros_like(thi_daily, dtype=np.float64)
    # Zone 1: 71 <= THI < 80, loss = (THI - 71) * 0.393
    mask1 = (thi_daily >= BREAKPOINT_1) & (thi_daily < BREAKPOINT_2)
    loss[mask1] = (thi_daily[mask1] - BREAKPOINT_1) * SLOPE_MODERATE
    # Zone 2: THI >= 80, loss = (80-71)*0.393 + (THI-80)*0.933
    mask2 = thi_daily >= BREAKPOINT_2
    loss[mask2] = (BREAKPOINT_2 - BREAKPOINT_1) * SLOPE_MODERATE + (thi_daily[mask2] - BREAKPOINT_2) * SLOPE_STEEP
    return loss


def linear_loss(thi_daily):
    """
    Calculate cumulative yield loss using linear function.
    Returns loss in kg/head/year.
    """
    loss = np.maximum(0, thi_daily - THRESHOLD_LINEAR) * LINEAR_SLOPE
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()
    year = args.year

    print("=" * 70)
    print(f"A-1 Global Bias: Linear vs Piecewise Dose-Response ({year})")
    print("=" * 70)

    # Load THI
    print(f"\nLoading THI: {THI_FILE}")
    ds = xr.open_dataset(THI_FILE)
    thi = ds["thi"]
    lats = thi.lat.values
    lons = thi.lon.values
    thi_year = thi.sel(time=str(year)).values  # (days, lat, lon)
    print(f"  THI shape: {thi_year.shape}")

    # Load GLW4 cattle
    print("\nLoading GLW4 cattle headcount...")
    headcount = load_glw_cattle(GLW_DIR, lats, lons)
    total_head = np.nansum(headcount)
    print(f"  Total cattle: {total_head/1e9:.2f}B head")

    # Calculate losses for each grid cell (vectorized over days)
    print("\nCalculating piecewise and linear losses...")
    # Sum over days
    pw_loss_daily = piecewise_loss(thi_year)  # (days, lat, lon)
    lin_loss_daily = linear_loss(thi_year)     # (days, lat, lon)

    pw_loss_annual = np.nansum(pw_loss_daily, axis=0)  # (lat, lon), kg/head/year
    lin_loss_annual = np.nansum(lin_loss_daily, axis=0)  # (lat, lon)

    # Convert to % of baseline annual yield
    annual_yield = BASELINE_YIELD * 365  # kg/head/year
    pw_loss_pct = pw_loss_annual / annual_yield * 100  # %
    lin_loss_pct = lin_loss_annual / annual_yield * 100  # %

    # Bias
    abs_bias = lin_loss_pct - pw_loss_pct  # percentage points
    rel_bias = np.where(pw_loss_pct > 0.01, abs_bias / pw_loss_pct * 100, 0)  # %

    # Climate zone summary (headcount-weighted)
    print("\nClimate zone summary (headcount-weighted):")
    zones = ['tropical', 'subtropical', 'temperate', 'boreal']
    zone_results = []

    for z in zones:
        if z == 'tropical':
            mask_lat = np.abs(lats) < 23.5
        elif z == 'subtropical':
            mask_lat = (np.abs(lats) >= 23.5) & (np.abs(lats) < 35)
        elif z == 'temperate':
            mask_lat = (np.abs(lats) >= 35) & (np.abs(lats) < 50)
        else:
            mask_lat = np.abs(lats) >= 50

        # Broadcast to 2D
        mask_2d = np.broadcast_to(mask_lat[:, None], headcount.shape)
        valid = mask_2d & (headcount > 0)

        if np.any(valid):
            hc = headcount[valid]
            pw = pw_loss_pct[valid]
            lin = lin_loss_pct[valid]
            rb = rel_bias[valid]

            hc_weighted_pw = np.sum(pw * hc) / np.sum(hc)
            hc_weighted_lin = np.sum(lin * hc) / np.sum(hc)
            hc_weighted_rb = np.sum(rb * hc) / np.sum(hc)
            zone_head = np.sum(hc)

            print(f"  {z:<12s}: head={zone_head/1e9:.2f}B, "
                  f"pw={hc_weighted_pw:.2f}%, lin={hc_weighted_lin:.2f}%, "
                  f"rel_bias={hc_weighted_rb:.1f}%")

            zone_results.append({
                'climate_zone': z,
                'cattle_head': zone_head,
                'pw_loss_pct_weighted': hc_weighted_pw,
                'lin_loss_pct_weighted': hc_weighted_lin,
                'abs_bias_pp': hc_weighted_lin - hc_weighted_pw,
                'rel_bias_pct': hc_weighted_rb,
            })
        else:
            print(f"  {z:<12s}: no cattle")
            zone_results.append({
                'climate_zone': z,
                'cattle_head': 0,
                'pw_loss_pct_weighted': 0,
                'lin_loss_pct_weighted': 0,
                'abs_bias_pp': 0,
                'rel_bias_pct': 0,
            })

    # Global summary
    valid_global = headcount > 0
    hc_global = headcount[valid_global]
    global_pw = np.sum(pw_loss_pct[valid_global] * hc_global) / np.sum(hc_global)
    global_lin = np.sum(lin_loss_pct[valid_global] * hc_global) / np.sum(hc_global)
    global_rb = np.sum(rel_bias[valid_global] * hc_global) / np.sum(hc_global)
    print(f"\n  {'GLOBAL':<12s}: head={total_head/1e9:.2f}B, "
          f"pw={global_pw:.2f}%, lin={global_lin:.2f}%, rel_bias={global_rb:.1f}%")

    # Save NetCDF
    out_nc = os.path.join(OUTDIR, f"a1_global_bias_{year}.nc")
    ds_out = xr.Dataset(
        {
            'pw_loss_pct': (['lat', 'lon'], pw_loss_pct.astype(np.float32)),
            'lin_loss_pct': (['lat', 'lon'], lin_loss_pct.astype(np.float32)),
            'abs_bias_pp': (['lat', 'lon'], abs_bias.astype(np.float32)),
            'rel_bias_pct': (['lat', 'lon'], rel_bias.astype(np.float32)),
            'cattle_head': (['lat', 'lon'], headcount.astype(np.float32)),
        },
        coords={'lat': lats, 'lon': lons},
        attrs={
            'description': 'A-1 Global bias: linear vs piecewise dose-response (dairy cattle)',
            'year': year,
            'piecewise_breakpoints': f'{BREAKPOINT_1}, {BREAKPOINT_2}',
            'piecewise_slopes': f'{SLOPE_SHALLOW}, {SLOPE_MODERATE}, {SLOPE_STEEP} kg/THI',
            'linear_slope': f'{LINEAR_SLOPE} kg/THI',
            'baseline_yield': f'{BASELINE_YIELD} kg/head/day',
            'thi_source': 'NASA POWER daily, 0.5deg',
            'glw_source': 'GLW4 cattle 2020, 10km',
        }
    )
    ds_out.to_netcdf(out_nc)
    print(f"\nNetCDF saved: {out_nc}")

    # Save CSV summary
    out_csv = os.path.join(OUTDIR, f"a1_global_bias_summary_{year}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=zone_results[0].keys())
        writer.writeheader()
        writer.writerows(zone_results)
        # Add global row
        writer.writerow({
            'climate_zone': 'GLOBAL',
            'cattle_head': total_head,
            'pw_loss_pct_weighted': global_pw,
            'lin_loss_pct_weighted': global_lin,
            'abs_bias_pp': global_lin - global_pw,
            'rel_bias_pct': global_rb,
        })
    print(f"CSV saved: {out_csv}")

    # Comparison with design doc's 4-zone result
    print(f"\n{'='*70}")
    print("Comparison with design doc (4-climate-zone,省会单点):")
    print(f"{'='*70}")
    print(f"  {'Zone':<12s} {'Design doc':>12s} {'Our global':>12s} {'Diff':>8s}")
    design_doc = {'tropical': 3.1, 'subtropical': 13.9, 'temperate': 31.7, 'boreal': 0.5}
    for zr in zone_results:
        z = zr['climate_zone']
        dd = design_doc.get(z, 0)
        og = zr['rel_bias_pct']
        print(f"  {z:<12s} {dd:>11.1f}% {og:>11.1f}% {og-dd:>+7.1f}%")

    ds.close()
    print("\nA-1 global bias calculation complete.")


if __name__ == "__main__":
    main()
