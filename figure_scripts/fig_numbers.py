"""
fig_numbers.py -- single source of truth for every number plotted in the
Nature Food revision figures (09_Figures/main, 09_Figures/si).

All headline totals are recomputed from 05_Data/loss_estimation_v4_1_
elevcorr_2000-2024.csv (year 2024, threshold thi_72) and checked against the
adopted internally consistent basis:
    gross 289.57 B USD (2005), avoidable 27.32 B (9.4%), residual 262.25 B
    cattle 142.45, buffalo 28.85, goats 23.84, pigs 22.38, sheep 16.66,
    chickens 55.39 (B USD)
    payback 3 yr 8.1% (23.4 B), 5 yr 9.4% (27.3 B), 10 yr 13.5% (39.0 B)
    2000 gross 196.22 B, headcount 18.25 -> 32.12 B (+76%),
    loss per animal 10.75 -> 9.02 USD (2005 USD)

Nothing in this module is hard-coded except values that are only documented
in the Supplementary Information (SI S2.2 cooling-measure cost ranges,
SI S2.5 PIA shares by species, SI S2.3 validation statistics); every such
value is tagged as "documented" in the returned dict and in
09_Figures/figure_notes.md.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "05_Data")
SRC = os.path.join(ROOT, "11_Source", "data_prep")
FIG = os.path.join(ROOT, "09_Figures")

YEAR = 2024
THRESHOLD = "thi_72"

SPECIES = ["cattle", "buffalo", "goats", "pigs", "sheep", "chickens"]
SPECIES_ORDER = ["cattle", "chickens", "buffalo", "goats", "pigs", "sheep"]
SPECIES_LABELS = {
    "cattle": "Cattle", "buffalo": "Buffalo", "goats": "Goats",
    "pigs": "Pigs", "sheep": "Sheep", "chickens": "Chickens",
}

# adopted internally consistent basis (B USD, 2005 constant)
ADOPTED_TOTAL_GROSS_B = 289.57
ADOPTED_SPECIES_GROSS_B = {
    "cattle": 142.45, "buffalo": 28.85, "goats": 23.84,
    "pigs": 22.38, "sheep": 16.66, "chickens": 55.39,
}
ADOPTED_AVOIDABLE_B = 27.32
ADOPTED_RESIDUAL_B = 262.25


def _read(name):
    return pd.read_csv(os.path.join(DATA, name), encoding="utf-8-sig")


def _read_src(name):
    return pd.read_csv(os.path.join(SRC, name), encoding="utf-8-sig")


def _nc(name):
    """Read one of the three netCDF4/HDF5 files with h5py."""
    import h5py
    path = os.path.join(SRC, name)
    out = {}
    with h5py.File(path, "r") as fh:
        for key in fh.keys():
            if isinstance(fh[key], h5py.Dataset):
                out[key] = np.array(fh[key][:])
        out["attrs"] = {k: (v.decode() if isinstance(v, bytes) else v)
                        for k, v in dict(fh.attrs).items()}
    return out


# --------------------------------------------------------------------------
# 1. loss basis (adopted 2024, thi_72) and 2000-2024 series
# --------------------------------------------------------------------------
def loss_basis():
    df = _read("loss_estimation_v4_1_elevcorr_2000-2024.csv")
    df = df[df["threshold"] == THRESHOLD]
    cur = df[df["year"] == YEAR].set_index("species")
    base = df[df["year"] == 2000].set_index("species")

    species = {}
    for sp in SPECIES:
        r = cur.loc[sp]
        species[sp] = {
            "gross_B": float(r["gross_loss_usd"]) / 1e9,
            "avoidable_B": float(r["avoidable_loss_usd"]) / 1e9,
            "residual_B": float(r["residual_loss_usd"]) / 1e9,
            "avoidable_fraction": float(r["avoidable_fraction"]),
            "yield_loss_fraction": float(r["yield_loss_fraction"]),
            "mean_effectiveness": float(r["mean_effectiveness"]),
            "head_B": float(r["total_head_scaled"]) / 1e9,
            "loss_per_head_usd": (float(r["gross_loss_usd"])
                                  / float(r["total_head_scaled"])),
        }
    gross_B = float(cur["gross_loss_usd"].sum()) / 1e9
    avoid_B = float(cur["avoidable_loss_usd"].sum()) / 1e9
    resid_B = float(cur["residual_loss_usd"].sum()) / 1e9
    head_B = float(cur["total_head_scaled"].sum()) / 1e9
    totals = {
        "gross_B": gross_B,
        "avoidable_B": avoid_B,
        "residual_B": resid_B,
        "avoidable_fraction": avoid_B / gross_B,
        "residual_fraction": resid_B / gross_B,
        "head_B": head_B,
        "loss_per_head_usd": float(cur["gross_loss_usd"].sum())
        / float(cur["total_head_scaled"].sum()),
    }

    # 2000 baseline
    y2000 = {
        "gross_B": float(base["gross_loss_usd"].sum()) / 1e9,
        "avoidable_B": float(base["avoidable_loss_usd"].sum()) / 1e9,
        "residual_B": float(base["residual_loss_usd"].sum()) / 1e9,
        "head_B": float(base["total_head_scaled"].sum()) / 1e9,
        "loss_per_head_usd": float(base["gross_loss_usd"].sum())
        / float(base["total_head_scaled"].sum()),
        "avoidable_fraction": float(base["avoidable_loss_usd"].sum())
        / float(base["gross_loss_usd"].sum()),
    }

    # 2000-2024 annual series
    g = df.groupby("year")
    series = {
        "years": sorted(int(y) for y in df["year"].unique()),
        "gross_B": (g["gross_loss_usd"].sum() / 1e9).reindex(
            sorted(df["year"].unique())).values,
        "avoidable_B": (g["avoidable_loss_usd"].sum() / 1e9).reindex(
            sorted(df["year"].unique())).values,
        "residual_B": (g["residual_loss_usd"].sum() / 1e9).reindex(
            sorted(df["year"].unique())).values,
        "head_B": (g["total_head_scaled"].sum() / 1e9).reindex(
            sorted(df["year"].unique())).values,
    }
    series["loss_per_head_usd"] = (series["gross_B"] / series["head_B"])
    series["avoid_fraction_pct"] = (series["avoidable_B"]
                                    / series["gross_B"] * 100.0)
    series["species_head_B"] = {
        sp: (df[df["species"] == sp].groupby("year")["total_head_scaled"]
             .sum().reindex(sorted(df["year"].unique())).values / 1e9)
        for sp in SPECIES
    }
    series["species_gross_B"] = {
        sp: (df[df["species"] == sp].groupby("year")["gross_loss_usd"]
             .sum().reindex(sorted(df["year"].unique())).values / 1e9)
        for sp in SPECIES
    }

    growth = {
        "gross_growth_pct": (totals["gross_B"] / y2000["gross_B"] - 1) * 100,
        "head_growth_pct": (totals["head_B"] / y2000["head_B"] - 1) * 100,
        "year_start": 2000,
        "year_end": YEAR,
    }

    # sensitivity of the avoidable fraction to the payback horizon.
    # computed species by species: each horizon's per-species avoidable
    # fraction (from the endogenized payback-horizon files) is multiplied by
    # that species' FAOSTAT-scaled 2024 gross from the adopted v4.1 basis.
    payback = {}
    for tag, frac_col in (("3yr", "payback3yr"),
                          ("5yr", None),
                          ("10yr", "payback10yr")):
        if frac_col is None:
            avoid_sp = {sp: species[sp]["avoidable_B"] for sp in SPECIES}
            avoid_h = sum(avoid_sp.values())
            source = "loss_estimation_v4_1 (5 yr baseline, species values)"
        else:
            d = _read(f"loss_estimation_v4_endogenized_2024_{frac_col}.csv")
            d = d[(d["year"] == YEAR) & (d["threshold"] == THRESHOLD)]
            d = d.set_index("species")
            avoid_sp = {sp: float(d.loc[sp, "avoidable_fraction"])
                        * species[sp]["gross_B"] for sp in SPECIES}
            avoid_h = sum(avoid_sp.values())
            source = (f"loss_estimation_v4_endogenized_2024_{frac_col} "
                      "(per-species avoidable fraction x adopted gross)")
        frac = avoid_h / gross_B
        payback[tag] = {
            "avoidable_fraction": frac,
            "avoidable_B": avoid_h,
            "residual_B": gross_B - avoid_h,
            "residual_fraction": 1.0 - frac,
            "species_avoidable_B": avoid_sp,
            "source": source,
        }

    # documented Table 1 species split (adopted basis, 5 yr)
    table1 = [{
        "species": sp,
        "gross_B": species[sp]["gross_B"],
        "avoidable_B": species[sp]["avoidable_B"],
        "residual_B": species[sp]["residual_B"],
        "avoidable_fraction_pct": species[sp]["avoidable_fraction"] * 100.0,
    } for sp in SPECIES_ORDER]
    table1.sort(key=lambda r: r["gross_B"], reverse=True)

    out = {
        "year": YEAR,
        "threshold": "THI >= 72",
        "species": species,
        "totals": totals,
        "table1": table1,
        "y2000": y2000,
        "series": series,
        "growth": growth,
        "payback": payback,
        "source": "05_Data/loss_estimation_v4_1_elevcorr_2000-2024.csv",
    }
    return out


# --------------------------------------------------------------------------
# 2. climate-zone shares (Fig 1b)
# --------------------------------------------------------------------------
def climate_zones():
    df = _read("a1_global_bias_summary_2024.csv")
    zones = []
    for _, r in df.iterrows():
        z = str(r["climate_zone"]).strip().lower()
        if z == "global":
            continue
        weight = float(r["cattle_head"]) * float(r["pw_loss_pct_weighted"])
        zones.append({
            "zone": z,
            "cattle_head": float(r["cattle_head"]),
            "pw_loss_pct_weighted": float(r["pw_loss_pct_weighted"]),
            "lin_loss_pct_weighted": float(r["lin_loss_pct_weighted"]),
            "loss_weight": weight,
        })
    head_tot = sum(z["cattle_head"] for z in zones)
    loss_tot = sum(z["loss_weight"] for z in zones)
    for z in zones:
        z["head_share_pct"] = z["cattle_head"] / head_tot * 100.0
        z["loss_share_pct"] = z["loss_weight"] / loss_tot * 100.0
    order = ["tropical", "subtropical", "temperate", "boreal"]
    zones.sort(key=lambda z: order.index(z["zone"]))
    glob = df[df["climate_zone"].astype(str).str.lower() == "global"].iloc[0]
    return {
        "zones": zones,
        "global_pw_loss_pct": float(glob["pw_loss_pct_weighted"]),
        "global_lin_loss_pct": float(glob["lin_loss_pct_weighted"]),
        "global_rel_bias_pct": float(glob["rel_bias_pct"]),
        "source": "05_Data/a1_global_bias_summary_2024.csv",
    }


# --------------------------------------------------------------------------
# 3. global cattle-loss map (Fig 1a) and PIA map (SI Fig 3)
# --------------------------------------------------------------------------
def _mollweide_prep(loss, head, lat, lon, factor=2, sigma=1.2):
    """Block-sum to `factor` degrees, smooth for cartography, renormalise."""
    from scipy.ndimage import gaussian_filter, zoom
    z = loss.reshape(loss.shape[0] // factor, factor,
                     loss.shape[1] // factor, factor).sum(axis=(1, 3))
    h = head.reshape(head.shape[0] // factor, factor,
                     head.shape[1] // factor, factor).sum(axis=(1, 3))
    z = gaussian_filter(z, sigma=sigma, mode="nearest")
    z *= loss.sum() / z.sum()
    lat_c = lat.reshape(-1, factor).mean(axis=1)
    lon_c = lon.reshape(-1, factor).mean(axis=1)
    zz = zoom(z, factor, order=1)
    hh = zoom(h, factor, order=1)
    la = zoom(lat_c, factor, order=1)
    lo = zoom(lon_c, factor, order=1)
    return zz, hh, la, lo


def map_cattle_loss():
    nc = _nc("a1_global_bias_2024.nc")
    head = nc["cattle_head"].astype(float)
    pw = nc["pw_loss_pct"].astype(float)
    lat = nc["lat"].astype(float)
    lon = nc["lon"].astype(float)
    raw = head * pw / 100.0
    scale = 142.449e9 / raw.sum()          # scale to the adopted cattle total
    loss = raw * scale
    # light Gaussian smoothing for cartographic display only (sigma = 2 cells
    # = 1 degree); the global total is restored afterwards
    from scipy.ndimage import gaussian_filter
    zz = gaussian_filter(loss, sigma=2.0, mode="nearest")
    zz *= loss.sum() / zz.sum()
    # logarithmic class breaks in USD per 0.5 degree cell (10 classes)
    breaks_usd = np.array([1e3, 10 ** 3.5, 1e4, 10 ** 4.5, 1e5, 10 ** 5.5,
                           1e6, 10 ** 6.5, 1e7, 10 ** 7.5, 1e8])
    values = np.where(head > 0, np.log10(np.maximum(zz, 1.0)), np.nan)
    return {
        "lat": lat,
        "lon": lon,
        "head": head,
        "loss_usd": zz,
        "log10_loss_usd": values,
        "levels_log10": np.log10(breaks_usd),
        "breaks_usd": breaks_usd,
        "n_cattle_B": float(head.sum()) / 1e9,
        "map_total_B": float(zz.sum()) / 1e9,
        "adopted_cattle_B": 142.449,
        "scale_usd_per_head_pct": float(scale),
        "smoothing_sigma_deg": 1.0,
        "aggregation_deg": 0.5,
        "max_cell_usd": float(np.nanmax(np.where(head > 0, zz, np.nan))),
        "source": "11_Source/data_prep/a1_global_bias_2024.nc",
        "formula": ("weight = cattle_head * pw_loss_pct / 100; scaled so the "
                    "global sum equals 142.449e9 USD (adopted cattle total); "
                    "Gaussian sigma = 1 degree for display, total restored"),
    }


def pia_data():
    nc = _nc("b_global_pia_2024.nc")
    summ = _read("b_global_pia_summary_2024.csv")
    ch = nc["cattle_head"].astype(float)
    lat = nc["lat"].astype(float)
    lon = nc["lon"].astype(float)
    # recompute the cattle-weighted PIA shares from the gridded field
    tiers = []
    var = {"25": "pia_25pct", "50": "pia_50pct", "75": "pia_75pct"}
    for _, r in summ.iterrows():
        tier = str(r["pia_tier"])
        num = "".join(ch for ch in tier if ch.isdigit())
        field = nc[var[num]].astype(float)
        recomputed = float((field * ch).sum() / ch.sum() * 100.0)
        tiers.append({
            "tier": tier,
            "days_pct": int(num),
            "cattle_in_pia_B": float(r["cattle_in_pia"]) / 1e9,
            "pct_global_cattle": float(r["pct_global_cattle"]),
            "pct_recomputed": recomputed,
        })
    # species PIA shares: documented in SI S2.5
    species_share = {
        "buffalo": 62.3, "cattle": 30.9, "goats": 28.7,
        "chickens": 15.6, "sheep": 12.1, "pigs": 8.3,
    }
    return {
        "tiers": tiers,
        "species_share_pct": species_share,
        "species_share_source": "Supplementary Information S2.5 (documented)",
        "map": {
            "pia_25pct": nc["pia_25pct"].astype(float),
            "pia_50pct": nc["pia_50pct"].astype(float),
            "pia_75pct": nc["pia_75pct"].astype(float),
            "cattle_head": ch,
            "lat": lat,
            "lon": lon,
        },
        "source": "11_Source/data_prep/b_global_pia_2024.nc, "
                  "05_Data/b_global_pia_summary_2024.csv",
    }


# --------------------------------------------------------------------------
# 4. methane intensity account (Fig 5)
# --------------------------------------------------------------------------
# The authors' CSV expresses additional emissions against an implied total
# livestock GHG baseline of 7.19 Gt CO2eq; the manuscript uses the methane
# budget (enteric + manure) of 3.5 Gt CO2eq.  All absolute additional-methane
# values are therefore rescaled by 3.5 / 7.19 (Lead correction, 2026-10-19).
METHANE_IMPLIED_BASELINE_Gt = 7.19
METHANE_BUDGET_Gt = 3.5
# 3.5 / 7.19 = 0.4868; the authors' correction specifies 0.487 (rounded)
METHANE_SCALE = 0.487


def methane():
    df = _read("environmental_intensity_ch4.csv")
    cur = df[df["year"] == YEAR].set_index("species")
    by_species = {}
    for sp in SPECIES:
        r = cur.loc[sp]
        by_species[sp] = {
            "increase_pct": float(r["hw_ch4_increase_pct"]),
            "additional_GtCO2eq": float(r["additional_ch4_GtCO2eq"])
            * METHANE_SCALE,
            "mean_exposure_days": float(r["mean_exposure_days"]),
            "threshold": float(r["threshold"]),
        }
    total = float(cur["additional_ch4_GtCO2eq"].sum()) * METHANE_SCALE
    years = sorted(int(y) for y in df["year"].unique())
    ser = df.groupby("year").agg(
        additional=("additional_ch4_GtCO2eq", "sum"),
        exposure_days_sum=("mean_exposure_days", "sum"),
    ).reindex(years)
    # headcount-weighted mean exposure days, using the adopted-basis headcounts
    lb = loss_basis()
    exp_weighted = []
    for y in years:
        d = df[df["year"] == y].set_index("species")
        hh = []
        for sp in SPECIES:
            hh.append(float(d.loc[sp, "mean_exposure_days"])
                      * float(lb["series"]["species_head_B"][sp][
                          years.index(y)]))
        exp_weighted.append(float(np.sum(hh) / np.sum(
            [lb["series"]["species_head_B"][sp][years.index(y)]
             for sp in SPECIES])))
    return {
        "year": YEAR,
        "by_species": by_species,
        "total_GtCO2eq": total,
        "share_of_livestock_ch4_pct": total / METHANE_BUDGET_Gt * 100.0,
        "years": years,
        "series_total_GtCO2eq": ser["additional"].values * METHANE_SCALE,
        "methane_scale": METHANE_SCALE,
        "implied_baseline_Gt": METHANE_IMPLIED_BASELINE_Gt,
        "methane_budget_Gt": METHANE_BUDGET_Gt,
        "exposure_days_headweighted": np.array(exp_weighted),
        "source": "05_Data/environmental_intensity_ch4.csv; absolute values "
                  "rescaled by 3.5/7.19 to the methane-only budget",
        "baseline_livestock_ch4_GtCO2eq": METHANE_BUDGET_Gt,
    }


# --------------------------------------------------------------------------
# 5. China provincial losses (SI Fig 1)
# --------------------------------------------------------------------------
def china():
    df = _read("china_provincial_loss_v4_elevcorr_2024.csv")
    df = df.sort_values("total_gross_usd", ascending=False)
    top = df.head(15)
    rows = [{
        "province": str(r["province"]),
        "gross_B": float(r["total_gross_usd"]) / 1e9,
        "avoidable_B": float(r["total_avoidable_usd"]) / 1e9,
        "residual_B": float(r["total_residual_usd"]) / 1e9,
        "avoidable_fraction": float(r["avoidable_fraction"]),
    } for _, r in top.iterrows()]
    return {
        "rows": rows,
        "total_gross_B": float(df["total_gross_usd"].sum()) / 1e9,
        "total_avoidable_B": float(df["total_avoidable_usd"].sum()) / 1e9,
        "total_avoidable_fraction": float(df["total_avoidable_usd"].sum()
                                          / df["total_gross_usd"].sum()),
        "source": "05_Data/china_provincial_loss_v4_elevcorr_2024.csv",
    }


# --------------------------------------------------------------------------
# 6. cooling measures (Fig 2c, SI Fig 2)
# --------------------------------------------------------------------------
def cooling():
    """Cooling measures as implemented in the revised loss model.

    The revised Methods (loss_estimation_v4_1_scaled.py) reports six measures
    with single capital-cost and effectiveness values:
        shade USD 65/head, 15 %; fans USD 19, 17 %; evaporative pads USD 150,
        50 %; sprinkler-fan systems USD 275, 65 %; smart sprinklers USD 300,
        65 %; tunnel ventilation USD 2,000, 80 %.
    There is no air-conditioning measure in the implemented model.

    The 5-year payback break-even capital cost is
        cost_breakeven = 5 * effectiveness * loss_per_head
    with the 2024 mean cattle gross loss per head (adopted basis, USD/head/yr).
    """
    lb = loss_basis()
    loss_per_head = lb["species"]["cattle"]["loss_per_head_usd"]
    payback = 5
    measures = [
        {"name": "Fans", "cost": 19.0, "eff": 17.0},
        {"name": "Shade", "cost": 65.0, "eff": 15.0},
        {"name": "Evaporative pads", "cost": 150.0, "eff": 50.0},
        {"name": "Sprinkler-fan systems", "cost": 275.0, "eff": 65.0},
        {"name": "Smart sprinklers", "cost": 300.0, "eff": 65.0},
        {"name": "Tunnel ventilation", "cost": 2000.0, "eff": 80.0},
    ]
    for m in measures:
        m["breakeven_cost"] = payback * (m["eff"] / 100.0) * loss_per_head
        m["payback_years_at_point"] = (m["cost"]
                                       / (loss_per_head * m["eff"] / 100.0))
        m["cost_effective"] = bool(m["cost"] <= m["breakeven_cost"])
    return {
        "measures": measures,
        "payback_years": payback,
        "loss_per_head_usd": loss_per_head,
        "source": ("revised Methods, implemented model "
                   "loss_estimation_v4_1_scaled.py (Lead update 2026-10-19)"),
    }


# --------------------------------------------------------------------------
# 7. dose-response validation (Fig 4)
# --------------------------------------------------------------------------
NASS_FIG4_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "nass_fig4_data.json")


def nass_fit():
    """Corrected US NASS continuous-hinge fit for Fig 4 panel (d).

    The derived data are produced by
    `06_Scripts_v2/build_nass_fig4_data.py` from the public NASS milk
    production and monthly state THI files, using models with
    intercept + state + month + year fixed effects and the breakpoint
    profiled by least squares.  Documented result (Lead correction,
    2026-10-20 third revision): n = 3,144, 24 states, 2013-2025.

    Primary model is THRESHOLD-LINEAR (zero slope below the breakpoint,
    matching the manuscript's threshold-linear dose-response form):
    kappa = THI 66.2, post-threshold slope -14.82 lbs/cow/month per THI
    (SE 1.79, P < 0.001), RSS 87,646,055, Gaussian AIC 32,278.62 (kappa
    counted) versus linear 32,327.98, delta-AIC 49.4.

    Sensitivity: unconstrained two-segment hinge, kappa = THI 63.1, slope
    below +2.203 (SE 0.778, P = 0.005), slope above -9.989 (SE 1.889,
    P < 0.001), delta-AIC 54.8.

    Returns a dict with `stats`, `bins` (2-THI-unit adjusted means +/- SE)
    and `lines` (fitted linear, threshold-linear and unconstrained values
    on a THI grid).
    """
    if not os.path.exists(NASS_FIG4_JSON):
        raise FileNotFoundError(
            "missing derived NASS data: run 06_Scripts_v2/"
            "build_nass_fig4_data.py first")
    with open(NASS_FIG4_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    data["bins"] = {k: np.array([r[k] for r in data["bins"]], dtype=float)
                    for k in ("center", "mean", "se", "n", "thi_mean")}
    data["lines"] = {k: np.array(v, dtype=float)
                     for k, v in data["lines"].items()}
    data["source"] = ("public NASS milk production and monthly state THI "
                      "data; corrected two-segment model with intercept + "
                      "state + month + year fixed effects")
    return data


def dose_response():
    cy = _read_src("cyprus_thi_bin_means.csv")
    cf = _read_src("cyprus_piecewise_fit_results.csv").iloc[0]
    bd = _read_src("bangladesh_piecewise_slope_test.csv")
    tk = _read_src("turkey_fertility_dose_response_20261013.csv")
    tp = _read_src("turkey_fertility_regression_params_20261013.csv")

    cyprus = {
        "thi": cy["THI_bin"].astype(float).values,
        "mean": cy["mean"].astype(float).values,
        "std": cy["std"].astype(float).values,
        "count": cy["count"].astype(int).values,
        "breakpoint": float(cf["piecewise_breakpoint"]),
        "slope1": float(cf["piecewise_slope1"]),
        "slope2": float(cf["piecewise_slope2"]),
        "linear_slope": float(cf["linear_slope"]),
        "r2_piecewise": float(cf["piecewise_r2"]),
        "r2_linear": float(cf["linear_r2"]),
        "delta_aic": float(cf["delta_aic"]),
        "slope_ratio": float(cf["slope_ratio"]),
        "n_records": int(cf["n_records"]),
        "n_cows": int(cf["n_cows"]),
        "ci": (67.2, 69.8),  # documented SI S2.3
    }
    # hinge fit of the documented piecewise parameterisation to bin means:
    # y = a + s1*x + (s2-s1)*max(0, x - bp); weights = bin counts
    x = cyprus["thi"]
    y = cyprus["mean"]
    w = cyprus["count"].astype(float)
    bp = cyprus["breakpoint"]
    basis = cyprus["slope1"] * x + (cyprus["slope2"] - cyprus["slope1"]) \
        * np.maximum(0.0, x - bp)
    a_pw = np.average(y - basis, weights=w)
    cyprus["intercept_piecewise"] = float(a_pw)
    lin_basis = cyprus["linear_slope"] * x
    cyprus["intercept_linear"] = float(np.average(y - lin_basis, weights=w))

    # Bangladesh: 5 genetic groups (Local, HF50, HF62.5, HF75, HF87.5)
    groups = ["Local", "HF50", "HF62.5", "HF75", "HF87.5"]
    variables = {"resp_rate": "Respiratory rate",
                 "rectal_temp": "Rectal temperature"}
    bd_out = {}
    for var, vlabel in variables.items():
        sub = bd[bd["variable"] == var].set_index("breed")
        rows = []
        for g in groups:
            r = sub.loc[g]
            p_raw = float(r["p_approx"])
            # the source p column stores a mirrored one-sided tail for some
            # rows (values > 1); recover the two-sided P without changing
            # the significance flag that is documented in the file
            p_corr = p_raw if p_raw <= 1.0 else 2.0 - p_raw
            rows.append({
                "group": g,
                "slope_ratio": float(r["slope_ratio"]),
                "p": p_corr,
                "p_raw": p_raw,
                "significant": bool(r["piecewise_evidence"]),
                "t_stat": float(r["t_stat"]),
            })
        bd_out[var] = {"label": vlabel, "rows": rows}
    pooled = {
        var: {
            "slope_ratio": float(bd[(bd["variable"] == var)
                                    & (bd["breed"] == "All")]["slope_ratio"].iloc[0]),
            "significant": bool(bd[(bd["variable"] == var)
                                   & (bd["breed"] == "All")]["piecewise_evidence"].iloc[0]),
        } for var in variables
    }

    params = dict(zip(tp["parameter"].astype(str), tp["value"]))
    turkey = {
        "thi": tk["mean_THI"].astype(float).values,
        "loss_pct": tk["mean_fertility_loss_pct"].astype(float).values,
        "std": tk["std_fertility_loss_pct"].astype(float).values,
        "n_days": tk["n_days"].astype(int).values,
        "threshold": float(params["threshold"]),
        "slope": float(params["slope"]),
        "intercept": float(params["intercept"]),
        "r2": float(params["r_squared"]),
        "n_obs": int(params["n_observations"]),
        "ci": (66.0, 70.0),  # documented SI S2.3: range across 17 regions
    }
    # zero crossing of the documented fitted line
    turkey["zero_crossing"] = -turkey["intercept"] / turkey["slope"]

    breakpoints = [
        {"label": "Cyprus dairy", "estimate": cyprus["breakpoint"],
         "ci": cyprus["ci"], "kind": "ci"},
        {"label": "Turkey fertility", "estimate": turkey["threshold"],
         "ci": turkey["ci"], "kind": "ci"},
        {"label": "Global (2023)", "estimate": 68.8,
         "ci": None, "kind": "point"},
        {"label": "Literature range", "estimate": None,
         "ci": (68.0, 72.0), "kind": "range"},
    ]
    # Fig 4 panel (d) is the corrected US NASS two-segment fit figure; the
    # forest-plot rows above remain available for provenance only.
    return {
        "cyprus": cyprus,
        "bangladesh": bd_out,
        "bangladesh_pooled": pooled,
        "turkey": turkey,
        "breakpoints": breakpoints,
        "nass": nass_fit(),
        "source": ("11_Source/data_prep/cyprus_thi_bin_means.csv, "
                   "cyprus_piecewise_fit_results.csv, "
                   "bangladesh_piecewise_slope_test.csv, "
                   "turkey_fertility_dose_response_20261013.csv, "
                   "turkey_fertility_regression_params_20261013.csv"),
    }


# --------------------------------------------------------------------------
# 8. cooling capacity by latitude/humidity zone (Fig 6c)
# --------------------------------------------------------------------------
def cooling_capacity_zones():
    nc = _nc("b_global_pia_2024.nc")
    cc = nc["cooling_capacity"].astype(float)
    ch = nc["cattle_head"].astype(float)
    lat = nc["lat"].astype(float)
    lon = nc["lon"].astype(float)
    LAT = np.repeat(lat[:, None], len(lon), axis=1)
    after = nc["pct_days_above_after_cooling"].astype(float)
    bands = [("Tropical", 0, 23.5), ("Subtropical", 23.5, 35.0),
             ("Temperate", 35.0, 55.0), ("Boreal", 55.0, 90.0)]
    rows = []
    for name, lo, hi in bands:
        m = (ch > 0) & (np.abs(LAT) >= lo) & (np.abs(LAT) < hi)
        rows.append({
            "band": name,
            "lat_range": (lo, hi),
            "capacity_degC": float(np.average(cc[m], weights=ch[m])),
            "capacity_min": float(cc[m].min()),
            "capacity_max": float(cc[m].max()),
            "cattle_head_B": float(ch[m].sum()) / 1e9,
            "residual_exposure_pct": float(np.average(after[m],
                                                      weights=ch[m])),
        })
    head_tot = sum(r["cattle_head_B"] for r in rows)
    for r in rows:
        r["head_share_pct"] = r["cattle_head_B"] / head_tot * 100.0
    return {
        "bands": rows,
        "source": "11_Source/data_prep/b_global_pia_2024.nc "
                  "(cooling_capacity, RH proxy = latitude zone)",
    }


# --------------------------------------------------------------------------
# 9. integrity checks against the adopted basis
# --------------------------------------------------------------------------
def checks():
    lb = loss_basis()
    ok = []

    def check(name, value, target, tol):
        good = abs(value - target) <= tol
        ok.append({"check": name, "value": value, "target": target,
                   "pass": bool(good)})

    check("total gross (B USD)", round(lb["totals"]["gross_B"], 2),
          ADOPTED_TOTAL_GROSS_B, 0.01)
    for sp, target in ADOPTED_SPECIES_GROSS_B.items():
        check(f"{sp} gross (B USD)",
              round(lb["species"][sp]["gross_B"], 2), target, 0.01)
    check("avoidable (B USD)", round(lb["totals"]["avoidable_B"], 2),
          ADOPTED_AVOIDABLE_B, 0.01)
    check("residual (B USD)", round(lb["totals"]["residual_B"], 2),
          ADOPTED_RESIDUAL_B, 0.01)
    check("avoidable fraction (%)",
          round(lb["totals"]["avoidable_fraction"] * 100, 2), 9.4, 0.05)
    check("payback 3 yr avoidable (B USD)",
          round(lb["payback"]["3yr"]["avoidable_B"], 2), 23.22, 0.01)
    check("payback 3 yr fraction (%)",
          round(lb["payback"]["3yr"]["avoidable_fraction"] * 100, 2), 8.02,
          0.02)
    check("payback 5 yr avoidable (B USD)",
          round(lb["payback"]["5yr"]["avoidable_B"], 2), 27.3, 0.05)
    check("payback 5 yr fraction (%)",
          round(lb["payback"]["5yr"]["avoidable_fraction"] * 100, 2), 9.4, 0.05)
    check("payback 10 yr avoidable (B USD)",
          round(lb["payback"]["10yr"]["avoidable_B"], 2), 38.93, 0.01)
    check("payback 10 yr fraction (%)",
          round(lb["payback"]["10yr"]["avoidable_fraction"] * 100, 2), 13.44,
          0.02)
    check("2000 gross (B USD)", round(lb["y2000"]["gross_B"], 2), 196.22, 0.01)
    check("2000 headcount (B)", round(lb["y2000"]["head_B"], 2), 18.25, 0.01)
    check("2024 headcount (B)", round(lb["totals"]["head_B"], 2), 32.12, 0.01)
    check("headcount growth (%)", round(lb["growth"]["head_growth_pct"], 1),
          76.0, 0.2)
    check("loss per head 2000 (USD)",
          round(lb["y2000"]["loss_per_head_usd"], 2), 10.75, 0.01)
    check("loss per head 2024 (USD)",
          round(lb["totals"]["loss_per_head_usd"], 2), 9.02, 0.01)
    m = methane()
    check("additional CH4 total (Gt CO2eq)",
          round(m["total_GtCO2eq"], 3), 0.092, 0.0005)
    check("additional CH4 share of methane budget (%)",
          round(m["share_of_livestock_ch4_pct"], 1), 2.6, 0.05)
    for sp, target in (("cattle", 0.0688), ("buffalo", 0.0061),
                       ("goats", 0.0058), ("sheep", 0.0055),
                       ("pigs", 0.0014), ("chickens", 0.0040)):
        check("additional CH4, " + sp + " (Gt CO2eq)",
              round(m["by_species"][sp]["additional_GtCO2eq"], 4),
              target, 0.00005)
    # methane trajectory (methane-only budget, 0.487 scale)
    ch4 = m["series_total_GtCO2eq"]
    check("CH4 2000 (Gt CO2eq)", round(float(ch4[0]), 3), 0.085, 0.001)
    check("CH4 2024 (Gt CO2eq)", round(float(ch4[-1]), 3), 0.092, 0.001)
    check("CH4 growth 2000-2024 (%)",
          round((float(ch4[-1]) / float(ch4[0]) - 1) * 100, 1), 8.1, 0.1)

    # cooling measures of the implemented model (Lead update 2026-10-19)
    cool = cooling()
    check("number of cooling measures", len(cool["measures"]), 6, 0)
    check("air conditioning present",
          int(any("air" in mm["name"].lower() for mm in cool["measures"])),
          0, 0)
    spec = {"Fans": (19.0, 17.0), "Shade": (65.0, 15.0),
            "Evaporative pads": (150.0, 50.0),
            "Sprinkler-fan systems": (275.0, 65.0),
            "Smart sprinklers": (300.0, 65.0),
            "Tunnel ventilation": (2000.0, 80.0)}
    for mm in cool["measures"]:
        c_target, e_target = spec[mm["name"]]
        check("cooling cost, %s (USD/head)" % mm["name"],
              round(mm["cost"], 1), c_target, 0.01)
        check("cooling effectiveness, %s (%%)" % mm["name"],
              round(mm["eff"], 1), e_target, 0.01)
    check("cost-effective measures at 5 yr",
          int(sum(1 for mm in cool["measures"] if mm["cost_effective"])), 4, 0)
    check("loss per head used for break-even (USD)",
          round(cool["loss_per_head_usd"], 2), 90.12, 0.01)

    # US NASS dairy: threshold-linear primary model + unconstrained
    # sensitivity (intercept + state/month/year FE), for Fig 4 panel (d)
    dr = dose_response()
    nf = dr["nass"]["stats"]
    tl = nf["threshold_linear"]
    uh = nf["unconstrained"]
    check("US NASS threshold-linear kappa (THI)",
          round(float(tl["kappa"]), 2), 66.2, 0.01)
    check("US NASS threshold-linear post slope (lbs/THI)",
          round(float(tl["slope_post"]), 3), -14.82, 0.005)
    check("US NASS threshold-linear delta-AIC",
          round(float(nf["delta_aic"]), 1), 49.4, 0.05)
    check("US NASS threshold-linear AIC (kappa counted)",
          round(float(tl["aic_kappa_counted"]), 2), 32278.62, 0.01)
    check("US NASS threshold-linear RSS", round(float(tl["rss"]), 0),
          87646055, 5)
    check("US NASS unconstrained kappa (THI)",
          round(float(uh["kappa"]), 2), 63.1, 0.01)
    check("US NASS unconstrained pre slope (lbs/THI)",
          round(float(uh["slope_pre"]), 3), 2.203, 0.005)
    check("US NASS unconstrained post slope (lbs/THI)",
          round(float(uh["slope_post"]), 3), -9.989, 0.005)
    check("US NASS unconstrained delta-AIC",
          round(float(uh["delta_aic"]), 1), 54.8, 0.05)
    check("US NASS observations", int(nf["n"]), 3144, 0)
    check("US NASS linear AIC", round(float(nf["linear"]["aic"]), 2),
          32327.98, 0.01)

    pia = pia_data()
    check("PIA cattle share (%)", round(pia["tiers"][1]["pct_global_cattle"], 2),
          30.9, 0.05)
    ch = china()
    check("China total gross (B USD)", round(ch["total_gross_B"], 2), 20.09,
          0.02)
    mp = map_cattle_loss()
    check("map cattle total (B USD)", round(mp["map_total_B"], 2), 142.45, 0.01)
    return ok


# --------------------------------------------------------------------------
def main():
    ok = checks()
    print("=" * 72)
    print("ADOPTED-BASIS INTEGRITY CHECKS (2024, THI >= 72)")
    print("=" * 72)
    for c in ok:
        print(("PASS" if c["pass"] else "FAIL"), c["check"],
              "=", c["value"], "(target", c["target"], ")")
    nfail = sum(1 for c in ok if not c["pass"])
    lb = loss_basis()
    print("\nTotal gross %.2f B | avoidable %.2f B (%.2f%%) | residual %.2f B"
          % (lb["totals"]["gross_B"], lb["totals"]["avoidable_B"],
             lb["totals"]["avoidable_fraction"] * 100,
             lb["totals"]["residual_B"]))
    for sp in SPECIES_ORDER:
        s = lb["species"][sp]
        print("  %-9s %7.2f B  avoid %7.2f B (%5.1f%%)  heads %5.2f B  "
              "loss/head %6.2f USD  yield loss %5.1f%%"
              % (sp, s["gross_B"], s["avoidable_B"],
                 s["avoidable_fraction"] * 100, s["head_B"],
                 s["loss_per_head_usd"], s["yield_loss_fraction"] * 100))
    print("\nPayback horizons:")
    for tag in ("3yr", "5yr", "10yr"):
        p = lb["payback"][tag]
        print("  %-4s avoidable %6.2f B (%.2f%%) residual %6.2f B"
              % (tag, p["avoidable_B"], p["avoidable_fraction"] * 100,
                 p["residual_B"]))
    print("\n2000: gross %.2f B, heads %.2f B, per head %.3f USD"
          % (lb["y2000"]["gross_B"], lb["y2000"]["head_B"],
             lb["y2000"]["loss_per_head_usd"]))
    print("2024: gross %.2f B, heads %.2f B, per head %.3f USD"
          % (lb["totals"]["gross_B"], lb["totals"]["head_B"],
             lb["totals"]["loss_per_head_usd"]))
    print("growth gross %.1f%%, heads %.1f%%"
          % (lb["growth"]["gross_growth_pct"], lb["growth"]["head_growth_pct"]))
    cz = climate_zones()
    print("\nClimate zones (head share / loss share, %):")
    for z in cz["zones"]:
        print("  %-12s %5.1f / %5.1f" % (z["zone"], z["head_share_pct"],
                                         z["loss_share_pct"]))
    mp = map_cattle_loss()
    print("\nMap: %d cells, total %.2f B, max cell %.1f M USD"
          % (np.isfinite(mp["log10_loss_usd"]).sum(), mp["map_total_B"],
             mp["max_cell_usd"] / 1e6))
    dr = dose_response()
    print("\nDose response:")
    print("  Cyprus breakpoint %.1f (%.1f-%.1f), R2 %.3f vs %.3f, dAIC %.1f"
          % (dr["cyprus"]["breakpoint"], dr["cyprus"]["ci"][0],
             dr["cyprus"]["ci"][1], dr["cyprus"]["r2_piecewise"],
             dr["cyprus"]["r2_linear"], dr["cyprus"]["delta_aic"]))
    print("  Turkey threshold %.1f, slope %.2f %%/THI, R2 %.4f, "
          "zero crossing %.2f" % (dr["turkey"]["threshold"],
                                  dr["turkey"]["slope"],
                                  dr["turkey"]["r2"],
                                  dr["turkey"]["zero_crossing"]))
    for var in ("resp_rate", "rectal_temp"):
        vals = ", ".join("%s %.2f%s" % (r["group"], r["slope_ratio"],
                                        "*" if r["significant"] else "")
                         for r in dr["bangladesh"][var]["rows"])
        print("  Bangladesh %-12s %s" % (var, vals))
    print("\nMethane 2024: total %.4f Gt CO2eq (%.1f%% of 3.5 Gt)"
          % (methane()["total_GtCO2eq"],
             methane()["share_of_livestock_ch4_pct"]))
    cn = china()
    print("\nChina: total %.2f B, top province %s %.2f B"
          % (cn["total_gross_B"], cn["rows"][0]["province"],
             cn["rows"][0]["gross_B"]))
    print("\nCooling capacity zones:")
    for r in cooling_capacity_zones()["bands"]:
        print("  %-12s %.1f degC, head share %.1f%%, residual exposure %.1f%%"
              % (r["band"], r["capacity_degC"], r["head_share_pct"],
                 r["residual_exposure_pct"]))
    print("\nCHECKS: %d passed, %d failed" % (len(ok) - nfail, nfail))
    with open(os.path.join(FIG, "_numbers_check.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"checks": ok,
                   "loss": {k: lb[k] for k in ("totals", "growth")},
                   "n_fail": nfail}, fh, indent=1)
    return nfail


if __name__ == "__main__":
    raise SystemExit(0 if main() == 0 else 1)
