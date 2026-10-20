"""
make_fig3.py -- Figure 3: temporal trends 2000-2024.

(a) gross, avoidable and residual losses (stacked area + gross line).
(b) global livestock headcount by species (stacked area).
(c) loss per animal (2005 USD per head).
(d) avoidable fraction of gross loss, stable near 9 to 10 %.

All numbers come from fig_numbers.py.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as S
import fig_numbers as F

OUT = S.FIG_MAIN
NAME = "Fig3_temporal_trends"


def _year_axis(ax, label=True):
    ax.set_xlim(2000, 2024)
    ax.set_xticks([2000, 2010, 2020])
    ax.set_xticklabels(["2000", "2010", "2020"],
                       fontsize=S.note_size(S.FS_TICK))
    if label:
        ax.set_xlabel("Year")


def panel_losses(fig, ser):
    ax = fig.add_axes([0.075, 0.585, 0.885, 0.325])
    yr = np.array(ser["years"])
    resid = ser["residual_B"]
    avoid = ser["avoidable_B"]
    gross = ser["gross_B"]
    ax.fill_between(yr, 0, resid, color=S.RESID_COL, lw=0, zorder=1,
                    label="Residual")
    ax.fill_between(yr, resid, gross, color=S.AVOID_COL, lw=0, zorder=2,
                    label="Avoidable")
    ax.plot(yr, gross, color="#2B2B2B", lw=1.1, zorder=4, label="Gross")
    ax.plot([yr[0], yr[-1]], [gross[0], gross[-1]], "o", ms=3.2,
            color="#2B2B2B", zorder=5)
    ax.annotate("%.1f" % gross[0], (yr[0], gross[0]), textcoords="offset points",
                xytext=(4, 3), fontsize=S.note_size(S.FS_ANNOT),
                color="#1A1A1A", ha="left", va="bottom")
    ax.annotate("%.1f" % gross[-1], (yr[-1], gross[-1]),
                textcoords="offset points", xytext=(-4, 3),
                fontsize=S.note_size(S.FS_ANNOT), color="#1A1A1A",
                ha="right", va="bottom")
    ax.annotate("%.1f" % avoid[0], (yr[0], resid[0] + avoid[0] * 0.5),
                textcoords="offset points", xytext=(3, 0),
                fontsize=S.note_size(S.FS_ANNOT), color="#F4F7F4",
                ha="left", va="center")
    ax.annotate("%.1f" % avoid[-1], (yr[-1], resid[-1] + avoid[-1] * 0.5),
                textcoords="offset points", xytext=(-3, 0),
                fontsize=S.note_size(S.FS_ANNOT), color="#F4F7F4",
                ha="right", va="center")
    ax.text(0.62, 0.90, "gross loss +%.1f%%" % F.loss_basis()["growth"][
        "gross_growth_pct"], transform=ax.transAxes,
        fontsize=S.note_size(S.FS_ANNOT), color="#2B2B2B", ha="left",
        va="top")
    ax.set_ylim(0, 330)
    ax.set_yticks([0, 100, 200, 300])
    ax.set_ylabel("Loss (B USD, 2005)")
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    ax.tick_params(axis="y", labelsize=S.note_size(S.FS_TICK))
    _year_axis(ax, label=False)
    ax.legend(loc="lower right", ncol=3, fontsize=S.note_size(S.FS_LEGEND),
              handlelength=1.1, columnspacing=1.0, labelspacing=0.2,
              borderaxespad=0.2, bbox_to_anchor=(1.01, 0.02))
    S.panel_label(fig, ax, "a", dx=-0.098, dy=0.006)
    return ax


def panel_heads(fig, ser):
    ax = fig.add_axes([0.078, 0.088, 0.240, 0.380])
    yr = np.array(ser["years"])
    bottom = np.zeros_like(yr, dtype=float)
    order = ["chickens", "cattle", "sheep", "goats", "pigs", "buffalo"]
    for sp in order:
        v = ser["species_head_B"][sp]
        ax.fill_between(yr, bottom, bottom + v, lw=0,
                        color=S.SPECIES_COLORS[sp],
                        label=S.SPECIES_LABELS[sp], zorder=2)
        bottom = bottom + v
    ax.plot([yr[-1], yr[-1]], [0, bottom[-1]], color="#4D4D4D", lw=0.6)
    ax.annotate("32.12", (yr[-1], bottom[-1]), textcoords="offset points",
                xytext=(-3, 3), fontsize=S.note_size(S.FS_ANNOT),
                ha="right", va="bottom", color="#1A1A1A")
    ax.annotate("18.25", (yr[0], bottom[0]), textcoords="offset points",
                xytext=(4, 3), fontsize=S.note_size(S.FS_ANNOT),
                ha="left", va="bottom", color="#1A1A1A")
    ax.text(0.04, 0.96, "headcount +%.0f%%" % F.loss_basis()["growth"][
        "head_growth_pct"], transform=ax.transAxes,
        fontsize=S.note_size(S.FS_ANNOT), color="#2B2B2B", ha="left",
        va="top")
    ax.set_ylim(0, 38)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel("Headcount (B animals)")
    ax.tick_params(axis="y", labelsize=S.note_size(S.FS_TICK))
    _year_axis(ax)
    ax.legend(loc="lower left", ncol=2, fontsize=S.note_size(S.FS_LEGEND),
              handlelength=0.9, handletextpad=0.3, columnspacing=0.7,
              labelspacing=0.15, borderaxespad=0.15,
              frameon=True, framealpha=0.85, edgecolor="none",
              facecolor="white")
    S.panel_label(fig, ax, "b", dx=-0.075, dy=0.008)
    return ax


def panel_per_head(fig, ser):
    ax = fig.add_axes([0.393, 0.088, 0.240, 0.380])
    yr = np.array(ser["years"])
    v = np.array(ser["loss_per_head_usd"])
    ax.plot(yr, v, color=S.SERIES_A, lw=1.3, zorder=3)
    ax.plot(yr, v, "o", ms=2.4, mfc="white", mec=S.SERIES_A, mew=0.6,
            zorder=4)
    ax.annotate("%.2f" % v[0], (yr[0], v[0]), textcoords="offset points",
                xytext=(4, 4), fontsize=S.note_size(S.FS_ANNOT),
                ha="left", va="bottom", color="#1A1A1A")
    ax.annotate("%.2f" % v[-1], (yr[-1], v[-1]), textcoords="offset points",
                xytext=(-4, -10), fontsize=S.note_size(S.FS_ANNOT),
                ha="right", va="top", color="#1A1A1A")
    change = (v[-1] / v[0] - 1) * 100
    ax.text(0.04, 0.96, "%.1f%% since 2000" % change,
            transform=ax.transAxes, fontsize=S.note_size(S.FS_ANNOT),
            color=S.SERIES_A, ha="left", va="top")
    ax.set_ylim(7.5, 11.5)
    ax.set_yticks([8, 9, 10, 11])
    ax.set_ylabel("Loss per animal (USD per head)")
    ax.tick_params(axis="y", labelsize=S.note_size(S.FS_TICK))
    _year_axis(ax)
    S.panel_label(fig, ax, "c", dx=-0.075, dy=0.008)
    return ax


def panel_avoid_fraction(fig, ser):
    ax = fig.add_axes([0.708, 0.088, 0.240, 0.380])
    yr = np.array(ser["years"])
    v = np.array(ser["avoid_fraction_pct"])
    ax.axhspan(9.0, 10.0, color=S.GREY_L, alpha=0.55, lw=0, zorder=0)
    ax.plot(yr, v, color=S.SERIES_B, lw=1.3, zorder=3)
    ax.plot(yr, v, "o", ms=2.4, mfc="white", mec=S.SERIES_B, mew=0.6,
            zorder=4)
    ax.text(2000.6, 9.06, "9 to 10% band", fontsize=S.note_size(S.FS_ANNOT),
            color="#6E6E6E", ha="left", va="bottom", zorder=5)
    ax.annotate("%.1f" % v[0], (yr[0], v[0]), textcoords="offset points",
                xytext=(4, 4), fontsize=S.note_size(S.FS_ANNOT),
                ha="left", va="bottom", color="#1A1A1A")
    ax.annotate("%.1f" % v[-1], (yr[-1], v[-1]), textcoords="offset points",
                xytext=(-3, 8), fontsize=S.note_size(S.FS_ANNOT),
                ha="right", va="bottom", color="#1A1A1A")
    ax.set_ylim(8.4, 10.8)
    ax.set_yticks([8.5, 9.0, 9.5, 10.0, 10.5])
    ax.set_ylabel("Avoidable fraction (%)")
    ax.tick_params(axis="y", labelsize=S.note_size(S.FS_TICK))
    _year_axis(ax)
    S.panel_label(fig, ax, "d", dx=-0.075, dy=0.008)
    return ax


def main():
    S.setup_style()
    lb = F.loss_basis()
    ser = lb["series"]
    fig = plt.figure(figsize=(S.FULL_W, 5.55))
    panel_losses(fig, ser)
    panel_heads(fig, ser)
    panel_per_head(fig, ser)
    panel_avoid_fraction(fig, ser)
    svg, png = S.save_figure(fig, NAME, OUT)
    print("saved", os.path.basename(svg), os.path.getsize(svg) // 1024, "KB")
    print("saved", os.path.basename(png), os.path.getsize(png) // 1024, "KB")


if __name__ == "__main__":
    main()
