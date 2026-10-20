"""
make_fig4.py -- Figure 4: piecewise dose-response validation.

(a) Cyprus dairy: binned milk yield vs THI maximum with the documented
    piecewise fit (breakpoint 68.5) and the linear fit.
(b) Bangladesh: post/pre-threshold slope ratios by genetic group for
    respiratory rate and rectal temperature, with P < 0.05 marks.
(c) Turkey: fertility loss vs THI with the documented threshold model
    (threshold 68, slope 0.92 %/THI).
(d) US NASS dairy threshold-linear fit (primary model, matching the
    manuscript's threshold-linear dose-response form: zero slope below the
    breakpoint). Binned state/month/year-adjusted milk yield (2-THI-unit
    bins, mean +/- SE) versus monthly mean THI, with the fitted linear model
    (dashed), the fitted threshold-linear model (solid; breakpoint THI 66.2,
    post-threshold slope -14.82 lbs/cow/month per THI, delta-AIC 49.4,
    n = 3,144) and, as a thin dotted sensitivity line, the unconstrained
    two-segment hinge (THI 63.1, delta-AIC 54.8). The corrected model
    includes the intercept that was missing from the first reanalysis;
    adding it reproduces the v4 qualitative claim of a breakpoint near THI 63
    to 66 with a negative post-threshold slope.

No confidence interval is fabricated: the only intervals shown are those
documented in the Supplementary Information (Cyprus 67.2 to 69.8, Turkey
66 to 70) and the literature range 68 to 72.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as S
import fig_numbers as F

OUT = S.FIG_MAIN
NAME = "Fig4_dose_response_validation"


def panel_cyprus(fig, dr):
    c = dr["cyprus"]
    ax = fig.add_axes([0.085, 0.560, 0.370, 0.352])
    x = c["thi"]
    y = c["mean"]
    sizes = 6.0 + 18.0 * (c["count"] / c["count"].max()) ** 0.5
    ax.scatter(x, y, s=sizes, facecolor="#9FB8C4", edgecolor="#3A5A６8"
               if False else "#3A5A68", linewidth=0.4, zorder=3)
    xx = np.linspace(46, 91, 400)
    bp = c["breakpoint"]
    ypw = (c["intercept_piecewise"] + c["slope1"] * xx
           + (c["slope2"] - c["slope1"]) * np.maximum(0.0, xx - bp))
    ylin = c["intercept_linear"] + c["linear_slope"] * xx
    ax.plot(xx, ypw, color=S.SERIES_A, lw=1.3, zorder=4,
            label="Piecewise fit")
    ax.plot(xx, ylin, color="#7A7A7A", lw=1.0, ls=(0, (4, 2)), zorder=4,
            label="Linear fit")
    ax.axvline(bp, color="#4D4D4D", lw=0.6, ls=(0, (1, 2)), zorder=2)
    ax.text(bp + 0.8, 32.3, "THI %.1f" % bp, fontsize=S.note_size(S.FS_ANNOT),
            color="#4D4D4D", ha="left", va="bottom")
    ax.text(49.5, 43.7,
            "piecewise: $\\it{R}$² = %.3f\nlinear: $\\it{R}$² = %.3f"
            "\nΔAIC = %.1f" % (c["r2_piecewise"], c["r2_linear"],
                               c["delta_aic"]),
            fontsize=S.note_size(S.FS_ANNOT), color="#1A1A1A", ha="left",
            va="top", linespacing=1.25)
    ax.text(47.0, 33.6, "n = %s test days, %d cows"
            % (format(c["n_records"], ","), c["n_cows"]),
            fontsize=S.note_size(S.FS_ANNOT), color="#4D4D4D", ha="left",
            va="bottom")
    ax.set_xlim(45, 92)
    ax.set_ylim(32, 44)
    ax.set_xticks([50, 60, 70, 80, 90])
    ax.set_yticks([33, 35, 37, 39, 41, 43])
    ax.set_xlabel("THI maximum")
    ax.set_ylabel("Milk yield (kg per cow per day)")
    ax.tick_params(axis="both", labelsize=S.note_size(S.FS_TICK))
    ax.legend(loc="upper right", fontsize=S.note_size(S.FS_LEGEND),
              handlelength=1.6, labelspacing=0.25, borderaxespad=0.3)
    S.panel_label(fig, ax, "a", dx=-0.105, dy=0.008)
    return ax


def panel_bangladesh(fig, dr):
    bd = dr["bangladesh"]
    ax = fig.add_axes([0.585, 0.560, 0.370, 0.352])
    groups = ["Local", "HF50", "HF62.5", "HF75", "HF87.5"]
    xpos = np.arange(len(groups))
    series = [("resp_rate", 0.0, "o", S.SERIES_A, "Respiratory rate"),
              ("rectal_temp", 0.0, "s", S.SERIES_B, "Rectal temperature")]
    offsets = {"resp_rate": -0.13, "rectal_temp": 0.13}
    for var, _, marker, col, label in series:
        rows = {r["group"]: r for r in bd[var]["rows"]}
        ys = [rows[g]["slope_ratio"] for g in groups]
        sig = [rows[g]["significant"] for g in groups]
        xo = xpos + offsets[var]
        ax.scatter(xo, ys, marker=marker, s=30, zorder=4,
                   facecolor=[col if s else "white" for s in sig],
                   edgecolor=col, linewidth=0.8)
        for xi, yi, s in zip(xo, ys, sig):
            if s:
                ax.annotate("*", (xi, yi), textcoords="offset points",
                            xytext=(0, 3.2), ha="center", va="bottom",
                            fontsize=S.note_size(S.FS_ANNOT) + 1.5,
                            color=col)
    ax.axhline(1.0, color="#4D4D4D", lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.text(0.0, 1.04, "no change (ratio = 1)", fontsize=S.note_size(
        S.FS_ANNOT), color="#4D4D4D", ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(0.35, 7.0)
    ax.set_yticks([0.5, 1, 2, 3, 4, 5])
    ax.set_yticklabels(["0.5", "1", "2", "3", "4", "5"])
    ax.set_xticks(xpos)
    ax.set_xticklabels(groups, fontsize=S.note_size(S.FS_TICK))
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_xlabel("Genetic group")
    ax.set_ylabel("Post/pre-threshold slope ratio (log scale)")
    ax.tick_params(axis="y", labelsize=S.note_size(S.FS_TICK))
    ax.text(0.985, 0.015, "filled: $\\it{P}$ < 0.05; open: not significant",
            transform=ax.transAxes, fontsize=S.note_size(S.FS_ANNOT),
            color="#4D4D4D", ha="right", va="bottom")
    handles = [
        Line2D([], [], marker="o", ls="", mfc=S.SERIES_A,
               mec=S.SERIES_A, markersize=4.5, label="Respiratory rate"),
        Line2D([], [], marker="s", ls="", mfc=S.SERIES_B,
               mec=S.SERIES_B, markersize=4.5, label="Rectal temperature"),
    ]
    ax.legend(handles=handles, loc="upper center", ncol=2,
              fontsize=S.note_size(S.FS_LEGEND), handletextpad=0.25,
              columnspacing=0.9, borderaxespad=0.15)
    S.panel_label(fig, ax, "b", dx=-0.105, dy=0.008)
    return ax


def panel_turkey(fig, dr):
    t = dr["turkey"]
    ax = fig.add_axes([0.085, 0.100, 0.320, 0.352])
    ax.errorbar(t["thi"], t["loss_pct"], yerr=t["std"], fmt="o", ms=4,
                mfc="white", mec=S.SERIES_A, mew=0.8, ecolor="#9FB8C4",
                elinewidth=0.9, capsize=2.2, capthick=0.6, zorder=3,
                label="Observed (mean ± SD)")
    xx = np.linspace(60, 78, 300)
    yy = np.maximum(0.0, t["slope"] * xx + t["intercept"])
    ax.plot(xx, yy, color=S.SERIES_B, lw=1.3, zorder=4, label="Threshold fit")
    ax.axvline(t["threshold"], color="#4D4D4D", lw=0.6, ls=(0, (1, 2)),
               zorder=2)
    ax.text(t["threshold"] - 0.5, 0.45, "THI %.0f" % t["threshold"],
            fontsize=S.note_size(S.FS_ANNOT), color="#4D4D4D", ha="right",
            va="bottom")
    ax.text(0.04, 0.97,
            "slope %.2f %% per THI unit\n$\\it{R}$² = %.4f"
            % (t["slope"], t["r2"]),
            transform=ax.transAxes, fontsize=S.note_size(S.FS_ANNOT),
            color="#1A1A1A", ha="left", va="top", linespacing=1.25)
    ax.set_xlim(60, 78)
    ax.set_ylim(-0.3, 7.2)
    ax.set_xticks([60, 64, 68, 72, 76])
    ax.set_yticks([0, 2, 4, 6])
    ax.set_xlabel("THI")
    ax.set_ylabel("Fertility loss (%)")
    ax.tick_params(axis="both", labelsize=S.note_size(S.FS_TICK))
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.80),
              fontsize=S.note_size(S.FS_LEGEND), handlelength=1.6,
              labelspacing=0.25, borderaxespad=0.0)
    S.panel_label(fig, ax, "c", dx=-0.100, dy=0.008)
    return ax


def panel_nass(fig, dr):
    nf = dr["nass"]
    st = nf["stats"]
    tl = st["threshold_linear"]
    uh = st["unconstrained"]
    b = nf["bins"]
    ln = nf["lines"]
    kappa = float(tl["kappa"])

    ax = fig.add_axes([0.565, 0.100, 0.395, 0.352])

    # binned state/month/year-adjusted means (+/- SE)
    ax.errorbar(b["center"], b["mean"], yerr=b["se"], fmt="o", ms=2.8,
                mfc="white", mec="#3A5A68", mew=0.7, ecolor="#B7BDC4",
                elinewidth=0.8, capsize=1.4, capthick=0.5, zorder=3,
                label="Binned means ± SE")
    ax.plot(ln["x"], ln["linear"], color="#7A7A7A", lw=1.0, ls=(0, (4, 2)),
            zorder=2, label="Linear fit")
    # sensitivity only: unconstrained two-segment hinge (thin dotted)
    ax.plot(ln["x"], ln["unconstrained"], color="#8C8C8C", lw=0.8,
            ls=(0, (1.5, 2.0)), zorder=3)
    # primary model: threshold-linear, zero slope below the breakpoint
    ax.plot(ln["x"], ln["threshold_linear"], color=S.SERIES_A, lw=1.5,
            zorder=4, label="Threshold-linear fit (66.2)")
    ax.axvline(kappa, color="#C9C9C9", lw=0.6, ls=(0, (1, 2)), zorder=1)

    y_kink = float(np.interp(kappa, ln["x"], ln["threshold_linear"]))
    ax.annotate("THI %.1f" % kappa, xy=(kappa, y_kink),
                xytext=(70.0, 2182.0), textcoords="data",
                fontsize=S.note_size(S.FS_ANNOT), color="#1A1A1A",
                ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#4D4D4D",
                                shrinkA=2, shrinkB=3), zorder=6)
    uh_kappa = float(uh["kappa"])
    y_uh = float(np.interp(uh_kappa, ln["x"], ln["unconstrained"]))
    ax.annotate("unconstrained\n%.1f" % uh_kappa, xy=(uh_kappa, y_uh),
                xytext=(63.0, 2135.0), textcoords="data",
                fontsize=S.note_size(S.FS_ANNOT), color="#7A7A7A",
                ha="right", va="bottom", linespacing=1.15,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#9A9A9A",
                                shrinkA=1, shrinkB=3), zorder=6)
    ax.text(0.02, 0.97,
            "ΔAIC = %.1f\nn = %s\nslope %.1f per THI"
            % (st["delta_aic"], format(st["n"], ","), tl["slope_post"]),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=S.note_size(S.FS_ANNOT), color="#1A1A1A",
            linespacing=1.25, zorder=6)

    ax.set_xlim(12, 84)
    ax.set_xticks([20, 40, 60, 80])
    ax.set_ylim(1800, 2240)
    ax.set_yticks([1800, 1900, 2000, 2100, 2200])
    ax.set_xlabel("Monthly mean THI")
    ax.set_ylabel("Adjusted milk yield (lbs/cow per month)")
    ax.tick_params(axis="both", labelsize=S.note_size(S.FS_TICK))
    ax.grid(axis="y", color="#EDEDED", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), ncol=1,
              fontsize=S.note_size(S.FS_LEGEND), frameon=False,
              handlelength=1.8, labelspacing=0.22, borderaxespad=0.0)
    S.panel_label(fig, ax, "d", dx=-0.140, dy=0.008)
    return ax


def main():
    S.setup_style()
    dr = F.dose_response()
    fig = plt.figure(figsize=(S.FULL_W, 6.05))
    panel_cyprus(fig, dr)
    panel_bangladesh(fig, dr)
    panel_turkey(fig, dr)
    panel_nass(fig, dr)
    svg, png = S.save_figure(fig, NAME, OUT)
    print("saved", os.path.basename(svg), os.path.getsize(svg) // 1024, "KB")
    print("saved", os.path.basename(png), os.path.getsize(png) // 1024, "KB")


if __name__ == "__main__":
    main()
