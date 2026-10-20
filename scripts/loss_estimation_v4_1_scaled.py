#!/usr/bin/env python3
"""
v4.1: Cooling cost endogenized + herd scaling (P7 fix).

Adds annual herd scaling factors to v4 model:
- GLW4 2020 headcount scaled by species-specific annual factors
- 2000-2024: cattle +16.5%, chickens +90.6%, goats +58.1%
- Fixes P7: previously all years used GLW 2020 fixed headcount

Usage:
  python loss_estimation_v4_1_scaled.py --year 2024
  python loss_estimation_v4_1_scaled.py --start 2000 --end 2024
"""
import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
import os
import csv
import argparse
import pandas as pd

# Paths (CPU1)
THI_FILE = "data/era5_nasapower/thi_daily_2000-2024_0.5deg.nc"
GLW_DIR = "data/glw4"
SCALING_FILE = "data_prep/global_herd_scaling_factors.csv"
OUTDIR = "data_prep"

SPECIES = ["cattle", "pigs", "chickens", "sheep", "goats", "buffalo"]

# THI thresholds
THRESHOLDS = {"thi_68": 68.0, "thi_72": 72.0, "thi_75": 75.0, "thi_78": 78.0}

# Value per head (2005 constant USD, from FAOSTAT Prices x Production, verified)
# 2024 values used as baseline; for year-specific analysis, load from value_per_head_2005usd_faostat.csv
VALUE_PER_HEAD = {
    "cattle": 414.07, "pigs": 217.33, "chickens": 16.31,
    "sheep": 113.46, "goats": 105.78, "buffalo": 527.10
}

# Dose-response parameters (industry-calibrated, % yield loss per day above threshold)
DOSE_RESPONSE = {
    "cattle": {"thi_68": 0.0008, "thi_72": 0.0015, "thi_75": 0.0025, "thi_78": 0.0040},
    "pigs": {"thi_68": 0.0005, "thi_72": 0.0010, "thi_75": 0.0018, "thi_78": 0.0030},
    "chickens": {"thi_68": 0.0004, "thi_72": 0.0008, "thi_75": 0.0015, "thi_78": 0.0025},
    "sheep": {"thi_68": 0.0006, "thi_72": 0.0012, "thi_75": 0.0020, "thi_78": 0.0035},
    "goats": {"thi_68": 0.0006, "thi_72": 0.0012, "thi_75": 0.0020, "thi_78": 0.0035},
    "buffalo": {"thi_68": 0.0007, "thi_72": 0.0013, "thi_75": 0.0022, "thi_78": 0.0038},
}

# Maximum yield loss (%)
MAX_LOSS = {"cattle": 0.35, "pigs": 0.30, "chickens": 0.25, "sheep": 0.25, "goats": 0.25, "buffalo": 0.30}

# Cooling measures: (name, cost_per_head, base_effectiveness, lifespan_years)
COOLING_MEASURES = [
    ("shade", 65, 0.15, 10),
    ("fans", 19, 0.17, 8),
    ("wet_curtain_fans", 150, 0.50, 8),
    ("sprinkler_fans", 275, 0.65, 8),
    ("smart_sprinkler", 300, 0.65, 8),
    ("tunnel_ventilation", 2000, 0.80, 15),
]

# RH zone efficiency multipliers
RH_EFFICIENCY = {"low": 1.0, "medium": 0.6, "high": 0.35}

# Income group: maximum affordable measure
INCOME_MAX_COST = {"high": 2000, "medium": 300, "low": 150}

# Payback period threshold (years)
PAYBACK_THRESHOLD = 5.0


def load_scaling_factors():
    """Load global herd scaling factors from CSV."""
    df = pd.read_csv(SCALING_FILE)
    scaling = {}
    for _, row in df.iterrows():
        sp = row['species']
        yr = int(row['year'])
        factor = float(row['global_scaling_to_2020'])
        if sp not in scaling:
            scaling[sp] = {}
        scaling[sp][yr] = factor
    return scaling


def load_glw_headcount(glw_dir, species, dst_lats, dst_lons):
    """Load GLW4 headcount, reprojected to THI grid."""
    fname = f"{species}_2020_10km.tif"
    fpath = os.path.join(glw_dir, fname)
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


def classify_rh_zone(lats):
    """Classify RH zone using latitude as proxy."""
    abs_lat = np.abs(lats)
    rh_zone = np.empty_like(abs_lat, dtype='<U10')
    rh_zone[abs_lat < 23.5] = 'high'
    rh_zone[(abs_lat >= 23.5) & (abs_lat < 35)] = 'medium'
    rh_zone[abs_lat >= 35] = 'low'
    return rh_zone


def classify_income(lats):
    """Classify income group using latitude as proxy."""
    abs_lat = np.abs(lats)
    income = np.empty_like(abs_lat, dtype='<U10')
    income[abs_lat >= 40] = 'high'
    income[(abs_lat >= 25) & (abs_lat < 40)] = 'medium'
    income[abs_lat < 25] = 'low'
    return income


def compute_optimal_effectiveness(gross_loss_per_head, rh_zone, income_group):
    """Compute optimal cooling effectiveness based on marginal cost curve."""
    rh_mult = RH_EFFICIENCY[rh_zone]
    max_cost = INCOME_MAX_COST[income_group]
    cumulative_effectiveness = 0.0
    cumulative_cost = 0.0
    for name, cost, base_eff, lifespan in COOLING_MEASURES:
        if name in ['wet_curtain_fans', 'sprinkler_fans', 'smart_sprinkler', 'tunnel_ventilation']:
            eff = base_eff * rh_mult
        else:
            eff = base_eff
        if cumulative_cost + cost > max_cost:
            continue
        annual_savings = gross_loss_per_head * eff
        payback = cost / annual_savings if annual_savings > 0 else 999
        if payback <= PAYBACK_THRESHOLD:
            cumulative_effectiveness = min(1.0, cumulative_effectiveness + eff * 0.5)
            cumulative_cost += cost
    return min(cumulative_effectiveness, 0.85)


def run_year(year, thi, lats, lons, glw_data, scaling_factors):
    """Run loss estimation for a single year with herd scaling."""
    thi_year = thi.sel(time=str(year))
    rh_zone_1d = classify_rh_zone(lats)
    income_1d = classify_income(lats)
    rh_zone_2d = np.broadcast_to(rh_zone_1d[:, None], (len(lats), len(lons)))
    income_2d = np.broadcast_to(income_1d[:, None], (len(lats), len(lons)))

    results = []
    for thresh_name, thresh_val in THRESHOLDS.items():
        exp_days = (thi_year > thresh_val).sum(dim="time").values
        for species in SPECIES:
            # Apply herd scaling factor
            scale = scaling_factors.get(species, {}).get(year, 1.0)
            headcount = glw_data[species] * scale
            valid = headcount > 0
            vph = VALUE_PER_HEAD[species]
            slope = DOSE_RESPONSE[species][thresh_name]
            max_loss = MAX_LOSS[species]
            loss_frac = np.minimum(slope * exp_days, max_loss)
            gross_loss = loss_frac * vph * headcount
            effectiveness = np.zeros_like(headcount)
            gross_per_head = loss_frac * vph
            for rz in ['low', 'medium', 'high']:
                for ig in ['high', 'medium', 'low']:
                    mask = (rh_zone_2d == rz) & (income_2d == ig) & valid
                    if not np.any(mask):
                        continue
                    gph_median = np.median(gross_per_head[mask])
                    eff = compute_optimal_effectiveness(gph_median, rz, ig)
                    effectiveness[mask] = eff
            avoidable_loss = gross_loss * effectiveness
            residual_loss = gross_loss * (1 - effectiveness)
            total_gross = np.nansum(gross_loss)
            total_avoidable = np.nansum(avoidable_loss)
            total_residual = np.nansum(residual_loss)
            total_head = np.nansum(headcount[valid])
            mean_eff = np.nansum(effectiveness[valid] * headcount[valid]) / total_head if total_head > 0 else 0
            results.append({
                "species": species,
                "threshold": thresh_name,
                "year": year,
                "herd_scaling_factor": scale,
                "total_head_scaled": total_head,
                "gross_loss_usd": total_gross,
                "avoidable_loss_usd": total_avoidable,
                "residual_loss_usd": total_residual,
                "avoidable_fraction": total_avoidable / total_gross if total_gross > 0 else 0,
                "mean_effectiveness": mean_eff,
                "yield_loss_fraction": np.nansum(loss_frac[valid] * headcount[valid]) / total_head if total_head > 0 else 0,
            })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--start", type=int, default=2000)
    parser.add_argument("--end", type=int, default=2024)
    args = parser.parse_args()

    print("=" * 70)
    print("v4.1 Loss Estimation: Endogenized Cooling + Herd Scaling")
    print("=" * 70)

    # Load scaling factors
    print(f"\nLoading herd scaling factors: {SCALING_FILE}")
    scaling_factors = load_scaling_factors()
    for sp in SPECIES:
        if sp in scaling_factors:
            print(f"  {sp}: 2000={scaling_factors[sp].get(2000, 'N/A'):.3f}, "
                  f"2024={scaling_factors[sp].get(2024, 'N/A'):.3f}")

    # Load THI
    print(f"\nLoading THI: {THI_FILE}")
    ds = xr.open_dataset(THI_FILE)
    thi = ds["thi"]
    lats = thi.lat.values
    lons = thi.lon.values

    # Load GLW4 (once, scaled per year)
    print("\nLoading GLW4 headcount (base year 2020)...")
    glw_data = {}
    for species in SPECIES:
        headcount = load_glw_headcount(GLW_DIR, species, lats, lons)
        total = np.nansum(headcount)
        glw_data[species] = headcount
        print(f"  {species}: {total/1e9:.2f}B head (2020 base)")

    # Determine years to run
    if args.year is not None:
        years = [args.year]
    else:
        years = list(range(args.start, args.end + 1))

    print(f"\nRunning years: {years[0]}-{years[-1]} ({len(years)} years)")

    all_results = []
    for year in years:
        print(f"\n--- Year {year} ---")
        results = run_year(year, thi, lats, lons, glw_data, scaling_factors)
        all_results.extend(results)
        # Summary for thi_72
        total_g = sum(r['gross_loss_usd'] for r in results if r['threshold'] == 'thi_72')
        total_a = sum(r['avoidable_loss_usd'] for r in results if r['threshold'] == 'thi_72')
        total_h = sum(r['total_head_scaled'] for r in results if r['threshold'] == 'thi_72')
        print(f"  THI>=72: gross=${total_g/1e9:.2f}B, avoidable=${total_a/1e9:.2f}B "
              f"({total_a/total_g*100:.1f}%), total_head={total_h/1e9:.2f}B")

    # Save CSV
    outfile = os.path.join(OUTDIR, f"loss_estimation_v4_1_scaled_{years[0]}-{years[-1]}.csv")
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved: {outfile} ({len(all_results)} rows)")

    # Temporal trend summary (thi_72, all species total)
    print(f"\n{'='*70}")
    print("TEMPORAL TREND (THI>=72, all species, herd-scaled)")
    print(f"{'='*70}")
    print(f"{'Year':<6s} {'Gross($B)':>10s} {'Avoid($B)':>10s} {'Resid($B)':>10s} {'Avoid%':>7s} {'Head(B)':>8s}")
    for year in years:
        yr_results = [r for r in all_results if r['year'] == year and r['threshold'] == 'thi_72']
        g = sum(r['gross_loss_usd'] for r in yr_results)
        a = sum(r['avoidable_loss_usd'] for r in yr_results)
        r = sum(r['residual_loss_usd'] for r in yr_results)
        h = sum(r['total_head_scaled'] for r in yr_results)
        print(f"{year:<6d} {g/1e9:>10.2f} {a/1e9:>10.2f} {r/1e9:>10.2f} {a/g*100:>6.1f}% {h/1e9:>8.2f}")

    ds.close()
    print("\nv4.1 scaled loss estimation complete.")


if __name__ == "__main__":
    main()
