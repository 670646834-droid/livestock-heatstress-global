#!/usr/bin/env python3
"""
P1-1: Environmental Intensity Account (CH4 emission intensity)
Compute spatial distribution of CH4 intensity increase due to heat stress.

Mechanism: heat stress -> reduced feed intake + direct metabolic effects
-> reduced feed efficiency -> increased CH4 per unit product.

Literature anchors:
- Korea dairy study: +3.1% CH4 intensity (heat stress vs thermoneutral)
  0.400 vs 0.388 kg CO2eq/kg FPCM (PMC9353352, citing Vitali)
- Aarhus University modeling: +0.8% to +6.6% across 9 scenarios
- Rhoads et al. 2009 J Dairy Sci: reduced feed intake explains only 35%
  of yield decline; 65% is direct metabolic effect

Model: CH4 intensity increase = baseline_slope * exposure_days * species_factor
where baseline_slope is calibrated from literature (0.015% per day above threshold
for cattle, giving ~3% increase at 200 exposure days, matching Korea study).
"""
import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
import os
import csv

# Paths (CPU1)
THI_FILE = "data/era5_nasapower/thi_daily_2000-2024_0.5deg.nc"
GLW_DIR = "data/glw4"
OUTDIR = "data_prep"

SPECIES = ["cattle", "pigs", "chickens", "sheep", "goats", "buffalo"]

# CH4 intensity increase per exposure day above threshold (percent per day)
# Calibrated: cattle 0.015%/day -> 200 days = 3.0% (matches Korea study)
# Other species scaled by relative metabolic sensitivity
CH4_SLOPE = {
    "cattle": 0.015,    # % per day above THI threshold
    "buffalo": 0.012,
    "sheep": 0.010,
    "goats": 0.010,
    "pigs": 0.008,
    "chickens": 0.006,
}

# THI threshold for each species (from literature)
THI_THRESHOLD = {
    "cattle": 68.0,
    "buffalo": 70.0,
    "sheep": 65.0,
    "goats": 65.0,
    "pigs": 72.0,
    "chickens": 75.0,
}

# Maximum CH4 intensity increase (cap)
MAX_INCREASE = {
    "cattle": 8.0,     # Aarhus max 6.6%, add margin
    "buffalo": 6.0,
    "sheep": 5.0,
    "goats": 5.0,
    "pigs": 4.0,
    "chickens": 3.0,
}

# Baseline CH4 intensity (kg CO2eq per kg product) - from GLEAM/FAO
BASELINE_CH4_INTENSITY = {
    "cattle_milk": 1.25,    # kg CO2eq/kg FPCM (global avg)
    "cattle_meat": 25.0,    # kg CO2eq/kg beef
    "buffalo_milk": 1.10,
    "buffalo_meat": 18.0,
    "sheep_milk": 1.50,
    "sheep_meat": 20.0,
    "goats_milk": 1.30,
    "goats_meat": 15.0,
    "pigs": 3.5,            # kg CO2eq/kg pork
    "chickens_meat": 2.5,
    "chickens_eggs": 2.8,
}


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


def main():
    print("=" * 70)
    print("Environmental Intensity Account: CH4 intensity due to heat stress")
    print("=" * 70)

    # Load THI
    print(f"\nLoading THI: {THI_FILE}")
    ds = xr.open_dataset(THI_FILE)
    thi = ds["thi"]
    lats = thi.lat.values
    lons = thi.lon.values
    print(f"  Shape: {thi.shape}")

    # Load GLW4
    print("\nLoading GLW4 headcount...")
    glw_data = {}
    for species in SPECIES:
        headcount = load_glw_headcount(GLW_DIR, species, lats, lons)
        total = np.nansum(headcount)
        glw_data[species] = headcount
        print(f"  {species}: {total/1e9:.2f}B head")

    # Compute CH4 intensity increase for each year and species
    results = []
    ch4_maps = {}  # store 2024 maps for spatial output

    for year in range(2000, 2025):
        thi_year = thi.sel(time=str(year))
        print(f"\nYear {year}:")

        for species in SPECIES:
            threshold = THI_THRESHOLD[species]
            slope = CH4_SLOPE[species]
            max_inc = MAX_INCREASE[species]
            headcount = glw_data[species]

            # Exposure days above threshold
            exp_days = (thi_year > threshold).sum(dim="time").values

            # CH4 intensity increase (%) = slope * exp_days, capped
            ch4_increase = np.minimum(slope * exp_days, max_inc)

            # Head-count weighted global average
            valid = headcount > 0
            hw_increase = np.nansum(ch4_increase[valid] * headcount[valid]) / np.nansum(headcount[valid])

            # Total additional CH4 emissions (approximate)
            # Using baseline intensity * production * increase%
            # For simplicity, use headcount * baseline intensity per head per year
            baseline_per_head = {
                "cattle": 2.8,      # tonnes CO2eq/head/year (milk+meat weighted)
                "buffalo": 2.2,
                "sheep": 0.5,
                "goats": 0.4,
                "pigs": 0.35,
                "chickens": 0.04,
            }
            total_head = np.nansum(headcount[valid])
            additional_ch4 = total_head * baseline_per_head[species] * (hw_increase / 100) / 1e9  # Gt CO2eq

            results.append({
                "species": species,
                "year": year,
                "threshold": threshold,
                "hw_ch4_increase_pct": round(hw_increase, 3),
                "additional_ch4_GtCO2eq": round(additional_ch4, 4),
                "mean_exposure_days": round(np.nansum(exp_days[valid] * headcount[valid]) / total_head, 1),
            })

            if year == 2024:
                ch4_maps[species] = ch4_increase
                print(f"  {species}: +{hw_increase:.2f}% CH4 intensity, {additional_ch4:.4f} Gt CO2eq additional")

    # Save CSV
    outfile = os.path.join(OUTDIR, "environmental_intensity_ch4.csv")
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["species", "year", "threshold", "hw_ch4_increase_pct",
                                                   "additional_ch4_GtCO2eq", "mean_exposure_days"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved: {outfile}")

    # Save 2024 CH4 intensity maps as NetCDF
    ds_out = xr.Dataset()
    for species in SPECIES:
        ds_out[f"ch4_increase_{species}"] = xr.DataArray(
            ch4_maps[species], dims=["lat", "lon"],
            coords={"lat": lats, "lon": lons},
            attrs={"units": "%", "long_name": f"CH4 intensity increase due to heat stress ({species})"}
        )
    nc_outfile = os.path.join(OUTDIR, "ch4_intensity_increase_2024.nc")
    ds_out.to_netcdf(nc_outfile)
    print(f"Spatial maps saved: {nc_outfile}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: 2024 CH4 Intensity Account")
    print("=" * 70)
    total_additional = 0
    for species in SPECIES:
        r = [x for x in results if x["species"] == species and x["year"] == 2024][0]
        total_additional += r["additional_ch4_GtCO2eq"]
        print(f"  {species:<12s}: +{r['hw_ch4_increase_pct']:>5.2f}%  +{r['additional_ch4_GtCO2eq']:>7.4f} Gt CO2eq")
    print(f"  {'TOTAL':<12s}:            +{total_additional:>7.4f} Gt CO2eq")
    print(f"\n  Global livestock CH4 emissions ~7.1 Gt CO2eq/year (FAO 2020)")
    print(f"  Heat stress additional: {total_additional/7.1*100:.1f}% of baseline")

    ds.close()
    print("\nEnvironmental intensity account complete.")


if __name__ == "__main__":
    main()
