"""Build the derived data for Figure 4 panel (d): corrected US NASS models
with intercept + state + month + year fixed effects.

Primary model (Lead correction 2026-10-20, third revision): THRESHOLD-LINEAR,
i.e. zero slope below the breakpoint and a linear decline above it, matching
the threshold-linear dose-response form of the manuscript.  With the intercept
restored this reproduces the v4 qualitative claim:
  kappa = THI 66.2; post-threshold slope -14.82 lbs/cow/month per THI
  (SE 1.79, P < 0.001); RSS 87,646,055; Gaussian AIC 32,278.62
  (kappa counted) versus linear 32,327.98; delta-AIC 49.4; n = 3,144,
  24 states, 2013-2025.

Sensitivity (unconstrained two-segment hinge, not the primary model):
  kappa = THI 63.1; slope below +2.203 (SE 0.778, P = 0.005); slope above
  -9.989 (SE 1.889, P < 0.001); delta-AIC 54.8.

Output: 06_Scripts_v2/nass_fig4_data.json with
  * stats: threshold-linear (primary), unconstrained hinge (sensitivity) and
           linear model estimates plus fit statistics
  * bins: 2-THI-unit bins of state/month/year-adjusted milk yield
          (mean +/- SE, n), adjusted with the primary model's fixed effects
  * lines: linear, threshold-linear and unconstrained fitted values on a
           THI grid

The raw NASS/THI CSVs are read from the public-data cache; set the
environment variables NASS_MILK_CSV and NASS_THI_CSV to override.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "nass_fig4_data.json")

NASS = os.environ.get(
    "NASS_MILK_CSV",
    r"F:\20261011\畜牧行业热应激\p0_raw\A_repos"
    r"\20716020__nass_milk_production_2013_2026.csv")
THI = os.environ.get(
    "NASS_THI_CSV",
    r"F:\20261011\畜牧行业热应激\p0_raw\A_repos"
    r"\20716020__thi_monthly_by_state.csv")

KAPPA_GRID = np.arange(40.0, 80.01, 0.1)


def _p_two_sided(z):
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def main():
    nass = pd.read_csv(NASS)
    thi = pd.read_csv(THI)
    nass["mpc"] = nass["milk_production_mlbs"] * 1e6 / (
        nass["milk_cows_khead"] * 1e3)
    nass = nass[nass["mpc"].notna() & (nass["mpc"] > 0)]
    df = nass.merge(thi, on=["state_abbr", "year", "month"], how="inner")
    y = df["mpc"].to_numpy(float)
    t = df["thi_mean"].to_numpy(float)
    n = len(y)

    states = pd.get_dummies(df["state_abbr"], drop_first=True).to_numpy(float)
    months = pd.get_dummies(df["month"], drop_first=True).to_numpy(float)
    years = pd.get_dummies(df["year"], drop_first=True).to_numpy(float)
    fe = [states, months, years]
    fe_matrix = np.column_stack(fe)
    const = np.ones((n, 1))

    def fit(cols):
        X = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        return float(res @ res), int(X.shape[1]), beta, X

    def aic(rss, k):
        return n * np.log(rss / n) + 2 * k

    # linear model: intercept + THI + FE
    rss_l, k_l, beta_l, _ = fit([const, t.reshape(-1, 1)] + fe)
    aic_l = aic(rss_l, k_l)

    # primary threshold-linear model: intercept + hinge + FE
    # (zero slope below the breakpoint)
    best_tl = None
    for kap in KAPPA_GRID:
        hinge = np.maximum(0.0, t - kap).reshape(-1, 1)
        rss, k, beta, X = fit([const, hinge] + fe)
        if best_tl is None or rss < best_tl[0]:
            best_tl = (rss, float(kap), k, beta, X)
    rss_tl, kappa_tl, k_tl, beta_tl, X_tl = best_tl
    aic_tl = aic(rss_tl, k_tl)
    delta_aic_primary = aic_l - (aic_tl + 2.0)   # kappa counted

    res_tl = y - X_tl @ beta_tl
    dof_tl = n - X_tl.shape[1]
    cov_tl = (res_tl @ res_tl / dof_tl) * np.linalg.pinv(X_tl.T @ X_tl)
    se_tl = np.sqrt(np.diag(cov_tl))
    slope_post_tl = float(beta_tl[1])
    se_post_tl = float(se_tl[1])
    p_post_tl = _p_two_sided(slope_post_tl / se_post_tl)
    r2_tl = 1.0 - float(res_tl @ res_tl) / float(
        ((y - y.mean()) ** 2).sum())
    fe_part_tl = beta_tl[0] + fe_matrix @ beta_tl[2:]
    offset_tl = float(fe_part_tl.mean())
    adj = y - fe_part_tl + offset_tl

    # sensitivity: unconstrained two-segment hinge: intercept + THI + hinge + FE
    best_h = None
    for kap in KAPPA_GRID:
        hinge = np.maximum(0.0, t - kap).reshape(-1, 1)
        rss, k, beta, X = fit([const, t.reshape(-1, 1), hinge] + fe)
        if best_h is None or rss < best_h[0]:
            best_h = (rss, float(kap), k, beta, X)
    rss_h, kappa_h, k_h, beta_h, X_h = best_h
    aic_h = aic(rss_h, k_h)
    delta_aic_h = aic_l - (aic_h + 2.0)

    res_h = y - X_h @ beta_h
    dof_h = n - X_h.shape[1]
    cov_h = (res_h @ res_h / dof_h) * np.linalg.pinv(X_h.T @ X_h)
    se_h = np.sqrt(np.diag(cov_h))
    b1, b2 = float(beta_h[1]), float(beta_h[2])
    se_pre = float(se_h[1])
    se_post = math.sqrt(se_h[1] ** 2 + se_h[2] ** 2 + 2 * cov_h[1, 2])
    p_pre = _p_two_sided(b1 / se_pre)
    p_post = _p_two_sided((b1 + b2) / se_post)
    r2_h = 1.0 - float(res_h @ res_h) / float(((y - y.mean()) ** 2).sum())
    fe_part_h = beta_h[0] + fe_matrix @ beta_h[3:]
    offset_h = float(fe_part_h.mean())

    # fixed-effect offsets for the two fitted lines on the adjusted scale
    fe_part_l = beta_l[0] + fe_matrix @ beta_l[2:]
    offset_l = float(fe_part_l.mean())

    bins = np.floor(t / 2.0) * 2.0
    rows = []
    for b in np.unique(bins):
        m = bins == b
        if m.sum() < 5:
            continue
        rows.append({
            "center": float(b + 1.0),
            "mean": float(adj[m].mean()),
            "se": float(adj[m].std(ddof=1) / math.sqrt(m.sum())),
            "n": int(m.sum()),
            "thi_mean": float(t[m].mean()),
        })

    grid = np.arange(12.0, 84.01, 0.25)
    linear_line = offset_l + float(beta_l[1]) * grid
    threshold_line = offset_tl + slope_post_tl * np.maximum(
        0.0, grid - kappa_tl)
    hinge_line = offset_h + b1 * grid + b2 * np.maximum(0.0, grid - kappa_h)

    stats = {
        "n": int(n),
        "n_states": int(df["state_abbr"].nunique()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "primary": "threshold_linear",
        "threshold_linear": {
            "kappa": kappa_tl,
            "slope_post": slope_post_tl,
            "se_post": se_post_tl,
            "p_post": p_post_tl,
            "rss": rss_tl,
            "k": int(k_tl),
            "aic": float(aic_tl),
            "aic_kappa_counted": float(aic_tl + 2.0),
            "r2": r2_tl,
        },
        "unconstrained": {
            "kappa": kappa_h,
            "slope_pre": b1,
            "se_pre": se_pre,
            "p_pre": p_pre,
            "slope_post": b1 + b2,
            "se_post": se_post,
            "p_post": p_post,
            "rss": rss_h,
            "k": int(k_h),
            "aic": float(aic_h),
            "aic_kappa_counted": float(aic_h + 2.0),
            "delta_aic": float(delta_aic_h),
            "r2": r2_h,
        },
        "linear": {
            "rss": rss_l,
            "k": int(k_l),
            "aic": float(aic_l),
        },
        "delta_aic": float(delta_aic_primary),
        "offset_threshold_linear": offset_tl,
        "offset_unconstrained": offset_h,
        "offset_linear": offset_l,
        "adjustment": ("state + month + year fixed effects removed; partial "
                       "residuals shifted to the overall mean"),
        "raw_data": [os.path.basename(NASS), os.path.basename(THI)],
        "source": "public NASS milk production and monthly state THI data",
        "model": ("Gaussian models with intercept + state + month + year "
                  "fixed effects; primary model is threshold-linear (zero "
                  "slope below the breakpoint) with kappa profiled by least "
                  "squares and counted in the AIC; the unconstrained "
                  "two-segment hinge is a sensitivity analysis"),
    }

    # documented targets from the Lead's corrected reanalysis
    assert abs(kappa_tl - 66.2) < 0.05, kappa_tl
    assert abs(slope_post_tl - (-14.82)) < 0.01, slope_post_tl
    assert abs(rss_tl - 87646055.0) < 5.0, rss_tl
    assert abs((aic_tl + 2.0) - 32278.62) < 0.05, aic_tl + 2.0
    assert abs(delta_aic_primary - 49.4) < 0.1, delta_aic_primary
    assert abs(kappa_h - 63.1) < 0.05, kappa_h
    assert abs(b1 - 2.203) < 0.01, b1
    assert abs((b1 + b2) - (-9.989)) < 0.01, b1 + b2
    assert abs(delta_aic_h - 54.8) < 0.1, delta_aic_h
    assert n == 3144, n

    payload = {
        "stats": stats,
        "bins": rows,
        "lines": {
            "x": [round(float(v), 3) for v in grid],
            "linear": [round(float(v), 3) for v in linear_line],
            "threshold_linear": [round(float(v), 3) for v in threshold_line],
            "unconstrained": [round(float(v), 3) for v in hinge_line],
        },
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    print("n =", n, "states =", stats["n_states"], "years",
          stats["year_min"], stats["year_max"])
    print("PRIMARY threshold-linear: kappa = %.2f  post-slope %+.3f "
          "(SE %.3f, P %.2e)  RSS %.0f  AIC(kappa counted) %.2f  "
          "delta-AIC %.2f  R2 %.4f"
          % (kappa_tl, slope_post_tl, se_post_tl, p_post_tl, rss_tl,
             aic_tl + 2.0, delta_aic_primary, r2_tl))
    print("SENSITIVITY unconstrained: kappa = %.2f  slopes %+.3f / %+.3f  "
          "delta-AIC %.2f  R2 %.4f"
          % (kappa_h, b1, b1 + b2, delta_aic_h, r2_h))
    print("linear: RSS %.0f  AIC %.2f" % (rss_l, aic_l))
    print("bins:", len(rows), "  adjusted mean range: %.1f to %.1f"
          % (min(r["mean"] for r in rows), max(r["mean"] for r in rows)))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
