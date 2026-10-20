#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""US NASS reanalysis for the Nature Food revision (v5, 2026-10-20).

Reproduces the two-segment/fixed-effects analysis reported in Supplementary Methods
S1.9 for the US NASS state-month dairy panel, using the public NASS milk-production
file and the state-monthly mean THI table.

Model (continuous two-segment "hinge", with intercept):
  yield_it = alpha + b1*THI_it + b2*max(0, THI_it - kappa)
             + state FE + month FE + year FE + e_it
  slope below kappa = b1 ; slope above = b1 + b2 ; kappa selected by profile
  least squares on a 0.1-THI grid.

Reported statistics (n = 3,144): kappa = THI 63.1; b1 = +2.203 (SE 0.778,
P = 0.005); slope above = -9.989 (SE 1.889, P < 0.001); delta-AIC = 54.8
(Gaussian AIC, kappa counted as an estimated parameter); R2 = 0.953.

Outputs:
  Supplementary_Data_4_nass_state_month_panel.csv   merged analysis panel (n = 3,144)
  Supplementary_Data_5_nass_model_comparison.csv    linear vs two-segment comparisons
"""
import math
import numpy as np
import pandas as pd

NASS_FILE = r"F:\20261011\畜牧行业热应激\p0_raw\A_repos\20716020__nass_milk_production_2013_2026.csv"
THI_FILE = r"F:\20261011\畜牧行业热应激\p0_raw\A_repos\20716020__thi_monthly_by_state.csv"
OUT_PANEL = r"E:\1014deepseek\review_1019\v5_build\05_data\Supplementary_Data_4_nass_state_month_panel.csv"
OUT_MODELS = r"E:\1014deepseek\review_1019\v5_build\05_data\Supplementary_Data_5_nass_model_comparison.csv"
GRID = np.arange(40.0, 80.001, 0.1)


def load_panel():
    nass = pd.read_csv(NASS_FILE)
    thi = pd.read_csv(THI_FILE)
    nass["milk_per_cow_lbs"] = nass["milk_production_mlbs"] * 1e6 / (nass["milk_cows_khead"] * 1e3)
    nass = nass[nass["milk_per_cow_lbs"].notna() & (nass["milk_per_cow_lbs"] > 0)]
    return nass.merge(thi, on=["state_abbr", "year", "month"], how="inner")


def fe_matrix(df, kind="state+month+year"):
    parts = []
    if "state" in kind:
        parts.append(pd.get_dummies(df["state_abbr"], drop_first=True).to_numpy(float))
    if "month" in kind and "time" not in kind:
        parts.append(pd.get_dummies(df["month"], drop_first=True).to_numpy(float))
    if "year" in kind:
        parts.append(pd.get_dummies(df["year"], drop_first=True).to_numpy(float))
    if "time" in kind:
        ym = df.year.astype(str) + "-" + df.month.astype(str).str.zfill(2)
        parts.append(pd.get_dummies(ym, drop_first=True).to_numpy(float))
    return parts


def fit(y, t, fe, kappa=None):
    cols = [np.ones((len(y), 1)), t.reshape(-1, 1)]
    if kappa is not None:
        cols.append(np.maximum(0.0, t - kappa).reshape(-1, 1))
    cols.extend(fe)
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    k = X.shape[1]
    dof = len(y) - k
    cov = (rss / dof) * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return rss, k, beta, se, resid


def aic(rss, n, k):
    return n * np.log(rss / n) + 2 * k


def main():
    df = load_panel()
    y = df["milk_per_cow_lbs"].to_numpy(float)
    t = df["thi_mean"].to_numpy(float)
    n = len(y)

    rows = []
    for fe_name in ["state+month+year", "state+time", "state+year"]:
        fe = fe_matrix(df, fe_name)
        rss_l, k_l, b_l, se_l, _ = fit(y, t, fe)
        aic_l = aic(rss_l, n, k_l)
        best = None
        for kap in GRID:
            rss, k, b, se, resid = fit(y, t, fe, kappa=kap)
            if best is None or rss < best[0]:
                best = (rss, kap, k, b, se, resid)
        rss_b, kap_b, k_b, b_b, se_b, res_b = best
        rows.append(dict(fixed_effects=fe_name, model="linear", breakpoint_THI="", n=n, k=k_l,
                         RSS=round(rss_l, 1), AIC=round(aic_l, 2), delta_AIC=0.0,
                         slope_below=round(float(b_l[1]), 3), slope_above=round(float(b_l[1]), 3),
                         p_slope_change="", note="baseline"))
        # best two-segment model (kappa counted as an estimated parameter)
        Xb = np.column_stack([np.ones((n, 1)), t.reshape(-1, 1),
                              np.maximum(0.0, t - kap_b).reshape(-1, 1)] + fe)
        cov_b = (rss_b / (n - Xb.shape[1])) * np.linalg.pinv(Xb.T @ Xb)
        sa = math.sqrt(se_b[1] ** 2 + se_b[2] ** 2 + 2 * cov_b[1, 2])
        p_ch = 2 * (1 - 0.5 * (1 + math.erf(abs(b_b[2] / se_b[2]) / math.sqrt(2))))
        r2 = 1 - rss_b / float(((y - y.mean()) @ (y - y.mean())))
        rows.append(dict(fixed_effects=fe_name, model="two-segment (continuous hinge, kappa by profile LS)",
                         breakpoint_THI=round(float(kap_b), 1), n=n, k=k_b + 1,
                         RSS=round(rss_b, 1), AIC=round(aic(rss_b, n, k_b + 1), 2),
                         delta_AIC=round(aic_l - aic(rss_b, n, k_b + 1), 2),
                         slope_below=round(float(b_b[1]), 3),
                         slope_above=round(float(b_b[1] + b_b[2]), 3),
                         p_slope_change="%.2e" % p_ch,
                         note="kappa counted as estimated parameter; R2 = %.4f" % r2))
        for kap in (63.1, 63.2, 68.5):
            rss, k, b, se, _ = fit(y, t, fe, kappa=kap)
            rows.append(dict(fixed_effects=fe_name, model="two-segment, fixed breakpoint %.1f" % kap,
                             breakpoint_THI=kap, n=n, k=k, RSS=round(rss, 1),
                             AIC=round(aic(rss, n, k), 2), delta_AIC=round(aic_l - aic(rss, n, k), 2),
                             slope_below=round(float(b[1]), 3), slope_above=round(float(b[1] + b[2]), 3),
                             p_slope_change="", note="kappa fixed"))
    # primary threshold-linear model (zero pre-breakpoint slope) with state+month+year FE
    fe = fe_matrix(df, "state+month+year")
    rss_l, k_l, b_l, se_l, _ = fit(y, t, fe)
    aic_l = aic(rss_l, n, k_l)
    best0 = None
    for kap in GRID:
        X0 = np.column_stack([np.ones((n, 1)), np.maximum(0.0, t - kap).reshape(-1, 1)] + fe)
        beta0, *_ = np.linalg.lstsq(X0, y, rcond=None)
        r0 = y - X0 @ beta0
        rss0 = float(r0 @ r0)
        if best0 is None or rss0 < best0[0]:
            best0 = (rss0, kap, X0.shape[1], beta0)
    rss0, kap0, k0, b0 = best0
    X0 = np.column_stack([np.ones((n, 1)), np.maximum(0.0, t - kap0).reshape(-1, 1)] + fe)
    cov0 = (rss0 / (n - k0)) * np.linalg.pinv(X0.T @ X0)
    se0 = np.sqrt(np.diag(cov0))
    p0 = 2 * (1 - 0.5 * (1 + math.erf(abs(b0[1] / se0[1]) / math.sqrt(2))))
    rows.append(dict(fixed_effects="state+month+year", model="threshold-linear (zero pre-breakpoint slope)",
                     breakpoint_THI=round(float(kap0), 1), n=n, k=k0 + 1, RSS=round(rss0, 1),
                     AIC=round(aic(rss0, n, k0 + 1), 2), delta_AIC=round(aic_l - aic(rss0, n, k0 + 1), 2),
                     slope_below=0.0, slope_above=round(float(b0[1]), 3),
                     p_slope_change="%.2e" % p0,
                     note="threshold-linear; kappa counted as estimated parameter"))
    models = pd.DataFrame(rows)
    models.to_csv(OUT_MODELS, index=False)
    panel = df[["state_abbr", "year", "month", "milk_production_mlbs", "milk_cows_khead",
                "milk_per_cow_lbs", "thi_mean"]].sort_values(["state_abbr", "year", "month"])
    panel.to_csv(OUT_PANEL, index=False)
    print("panel:", panel.shape, "->", OUT_PANEL)
    print(models.to_string(index=False))


if __name__ == "__main__":
    main()
