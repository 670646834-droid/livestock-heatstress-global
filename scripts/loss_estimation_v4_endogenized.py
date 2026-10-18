#!/usr/bin/env python3
"""
P1-2: Cooling cost curve fully endogenized into spatial loss model.

Key improvements over v3 (fixed effectiveness 0.2-0.5):
1. RH-zone stratification: evaporative cooling efficiency constrained by humidity
   - Low RH (<45%): full efficiency
   - Medium RH (45-55%): marginalized
   - High RH (>55%): <5C cooling, efficiency <40%
2. Income-group stratification: capital availability determines feasible measures
   - High income: can afford tunnel ventilation ($2000/head)
   - Medium income: can afford sprinkler+fans ($275/head)
   - Low income: only shade ($65/head) or fans ($19/head)
3. Marginal cost curve: 6 measures sorted by cost, compute optimal adoption
   based on payback period threshold (3-5 years typical farm investment horizon)

Usage: python loss_estimation_v4_endogenized.py --year 2024
"""
import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
import os
import csv
import argparse

# Paths (CPU1)
THI_FILE = "data/era5_nasapower/thi_daily_2000-2024_0.5deg.nc"
GLW_DIR = "data/glw4"
OUTDIR = "data_prep"

SPECIES = ["cattle", "pigs", "chickens", "sheep", "goats", "buffalo"]

# THI thresholds
THRESHOLDS = {"thi_68": 68.0, "thi_72": 72.0, "thi_75": 75.0, "thi_78": 78.0}

# Value per head (2005 constant USD, from FAOSTAT Prices x Production, deflated by US GDP deflator)
# Source: data_prep/value_per_head_2005usd_faostat.csv (verified, not hardcoded approximation)
# 2024 values: cattle $414.07, pigs $217.33, chickens $16.31, sheep $113.46, goats $105.78, buffalo $527.10
# Note: Previous version used hardcoded $650 for cattle, which overestimated cattle losses by 57% ($650/$414 = 1.57)
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
# Effectiveness is % of heat stress loss that can be avoided under optimal conditions
COOLING_MEASURES = [
    ("shade", 65, 0.15, 10),
    ("fans", 19, 0.17, 8),
    ("wet_curtain_fans", 150, 0.50, 8),
    ("sprinkler_fans", 275, 0.65, 8),
    ("smart_sprinkler", 300, 0.65, 8),  # same effectiveness, lower opex
    ("tunnel_ventilation", 2000, 0.80, 15),
]

# RH zone efficiency multipliers (for evaporative cooling measures)
# shade and fans are not evaporative, so RH multiplier = 1.0
RH_EFFICIENCY = {
    "low": 1.0,      # RH < 45%: full evaporative efficiency
    "medium": 0.6,   # RH 45-55%: marginalized
    "high": 0.35,    # RH > 55%: <5C cooling, <40% efficiency
}

# Income group: determines maximum affordable measure (by cost_per_head)
# Using latitude as proxy: high latitude = high income (temperate developed),
# low latitude = low income (tropical developing)
# More precise would use World Bank income classification, but latitude is reasonable proxy
INCOME_MAX_COST = {
    "high": 2000,    # can afford tunnel ventilation
    "medium": 300,   # can afford smart sprinkler
    "low": 150,      # can only afford wet curtain or below
}

# Payback period threshold (years) - typical farm investment horizon
# Can be overridden via --payback command line argument
PAYBACK_THRESHOLD = 5.0


def load_glw_headcount(glw_dir, species, dst_lats, dst_lons):
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


def classify_rh_zone(lats, lons):
    """Classify RH zone using latitude as proxy.
    Tropical (|lat|<23.5): high RH
    Subtropical (23.5<=|lat|<35): medium RH
    Temperate (|lat|>=35): low RH
    Note: arid regions (Sahara, Central Asia) would be low RH even at low latitudes,
    but latitude proxy is a reasonable first approximation.
    """
    abs_lat = np.abs(lats)
    rh_zone = np.empty_like(abs_lat, dtype='<U10')
    rh_zone[abs_lat < 23.5] = 'high'
    rh_zone[(abs_lat >= 23.5) & (abs_lat < 35)] = 'medium'
    rh_zone[abs_lat >= 35] = 'low'
    return rh_zone


def classify_income(lats):
    """Classify income group using latitude as proxy.
    High latitude (|lat|>=40): high income (North America, Europe, Japan)
    Medium latitude (25<=|lat|<40): medium income (China, Argentina, Mediterranean)
    Low latitude (|lat|<25): low income (South Asia, sub-Saharan Africa, Southeast Asia)
    """
    abs_lat = np.abs(lats)
    income = np.empty_like(abs_lat, dtype='<U10')
    income[abs_lat >= 40] = 'high'
    income[(abs_lat >= 25) & (abs_lat < 40)] = 'medium'
    income[abs_lat < 25] = 'low'
    return income


def compute_optimal_effectiveness(gross_loss_per_head, rh_zone, income_group):
    """Compute optimal cooling effectiveness based on marginal cost curve.

    For each measure (sorted by cost), check if:
    1. cost <= income_max_cost[income_group]
    2. payback_period = cost / (gross_loss * effectiveness) <= PAYBACK_THRESHOLD

    Return the maximum achievable effectiveness from the optimal measure combination.
    Measures are cumulative (shade + fans + sprinkler = combined effectiveness).
    """
    # RH efficiency multiplier for evaporative measures
    rh_mult = RH_EFFICIENCY[rh_zone]
    max_cost = INCOME_MAX_COST[income_group]

    cumulative_effectiveness = 0.0
    cumulative_cost = 0.0

    for name, cost, base_eff, lifespan in COOLING_MEASURES:
        # Apply RH efficiency for evaporative measures (wet_curtain, sprinkler, tunnel)
        if name in ['wet_curtain_fans', 'sprinkler_fans', 'smart_sprinkler', 'tunnel_ventilation']:
            eff = base_eff * rh_mult
        else:
            eff = base_eff

        # Check affordability
        if cumulative_cost + cost > max_cost:
            continue

        # Check payback period
        annual_savings = gross_loss_per_head * eff
        if annual_savings > 0:
            payback = cost / annual_savings
        else:
            payback = 999

        if payback <= PAYBACK_THRESHOLD:
            cumulative_effectiveness = min(1.0, cumulative_effectiveness + eff * 0.5)  # diminishing returns
            cumulative_cost += cost

    return min(cumulative_effectiveness, 0.85)  # cap at 85% (even best tech can't eliminate all)


def main():
    global PAYBACK_THRESHOLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--payback", type=float, default=5.0, help="Payback period threshold in years (default: 5.0)")
    args = parser.parse_args()
    year = args.year
    PAYBACK_THRESHOLD = args.payback

    print("=" * 70)
    print(f"Loss Estimation v4: Cooling Cost Endogenized (Year {year}, Payback {PAYBACK_THRESHOLD}yr)")
    print("=" * 70)

    # Load THI
    print(f"\nLoading THI: {THI_FILE}")
    ds = xr.open_dataset(THI_FILE)
    thi = ds["thi"]
    lats = thi.lat.values
    lons = thi.lon.values

    # Classify RH zone and income group for each latitude
    rh_zone_1d = classify_rh_zone(lats, lons)
    income_1d = classify_income(lats)
    # Broadcast to 2D
    rh_zone_2d = np.broadcast_to(rh_zone_1d[:, None], (len(lats), len(lons)))
    income_2d = np.broadcast_to(income_1d[:, None], (len(lats), len(lons)))

    # Load GLW4
    print("\nLoading GLW4 headcount...")
    glw_data = {}
    for species in SPECIES:
        headcount = load_glw_headcount(GLW_DIR, species, lats, lons)
        total = np.nansum(headcount)
        glw_data[species] = headcount
        print(f"  {species}: {total/1e9:.2f}B head")

    # Compute exposure for the year
    print(f"\nComputing exposure for {year}...")
    thi_year = thi.sel(time=str(year))

    results = []

    for thresh_name, thresh_val in THRESHOLDS.items():
        exp_days = (thi_year > thresh_val).sum(dim="time").values

        for species in SPECIES:
            headcount = glw_data[species]
            valid = headcount > 0
            vph = VALUE_PER_HEAD[species]
            slope = DOSE_RESPONSE[species][thresh_name]
            max_loss = MAX_LOSS[species]

            # Gross loss fraction per grid cell
            loss_frac = np.minimum(slope * exp_days, max_loss)
            gross_loss = loss_frac * vph * headcount  # USD

            # Compute effectiveness for each grid cell (vectorized)
            # For simplicity, compute unique (rh_zone, income_group) combinations
            effectiveness = np.zeros_like(headcount)
            gross_per_head = loss_frac * vph

            for rz in ['low', 'medium', 'high']:
                for ig in ['high', 'medium', 'low']:
                    mask = (rh_zone_2d == rz) & (income_2d == ig) & valid
                    if not np.any(mask):
                        continue
                    # Use median gross_per_head in this zone to compute effectiveness
                    gph_median = np.median(gross_per_head[mask])
                    eff = compute_optimal_effectiveness(gph_median, rz, ig)
                    effectiveness[mask] = eff

            # Avoidable and residual loss
            avoidable_loss = gross_loss * effectiveness
            residual_loss = gross_loss * (1 - effectiveness)

            total_gross = np.nansum(gross_loss)
            total_avoidable = np.nansum(avoidable_loss)
            total_residual = np.nansum(residual_loss)
            total_head = np.nansum(headcount[valid])
            mean_eff = np.nansum(effectiveness[valid] * headcount[valid]) / total_head

            results.append({
                "species": species,
                "threshold": thresh_name,
                "year": year,
                "total_head": total_head,
                "gross_loss_usd": total_gross,
                "avoidable_loss_usd": total_avoidable,
                "residual_loss_usd": total_residual,
                "avoidable_fraction": total_avoidable / total_gross if total_gross > 0 else 0,
                "mean_effectiveness": mean_eff,
                "yield_loss_fraction": np.nansum(loss_frac[valid] * headcount[valid]) / total_head,
            })

            print(f"  {species} {thresh_name}: gross=${total_gross/1e9:.2f}B, "
                  f"avoidable=${total_avoidable/1e9:.2f}B ({total_avoidable/total_gross*100:.1f}%), "
                  f"residual=${total_residual/1e9:.2f}B, mean_eff={mean_eff:.3f}")

    # Save CSV
    payback_tag = f"payback{int(PAYBACK_THRESHOLD)}yr"
    outfile = os.path.join(OUTDIR, f"loss_estimation_v4_endogenized_{year}_{payback_tag}.csv")
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved: {outfile}")

    # Summary for thi_72
    print(f"\n{'='*70}")
    print(f"SUMMARY: {year} THI>=72 (Endogenized Cooling)")
    print(f"{'='*70}")
    total_g = total_a = total_r = 0
    for r in results:
        if r['threshold'] == 'thi_72':
            total_g += r['gross_loss_usd']
            total_a += r['avoidable_loss_usd']
            total_r += r['residual_loss_usd']
            print(f"  {r['species']:<12s}: ${r['gross_loss_usd']/1e9:>6.2f}B gross, "
                  f"${r['avoidable_loss_usd']/1e9:>6.2f}B avoidable ({r['avoidable_fraction']*100:>4.1f}%), "
                  f"eff={r['mean_effectiveness']:.3f}")
    print(f"  {'TOTAL':<12s}: ${total_g/1e9:>6.2f}B gross, "
          f"${total_a/1e9:>6.2f}B avoidable ({total_a/total_g*100:.1f}%), "
          f"${total_r/1e9:>6.2f}B residual")

    ds.close()
    print("\nv4 endogenized loss estimation complete.")


if __name__ == "__main__":
    main()
