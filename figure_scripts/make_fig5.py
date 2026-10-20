"""
make_fig5.py -- Figure 5: heat-stress-induced increase in enteric methane
intensity and the additional methane emissions (2024).

Absolute additional-methane values are expressed on the methane-only budget
(enteric + manure, 3.5 Gt CO2eq per year): the source CSV is rescaled by
0.487 = 3.5/7.19 (Lead correction 2026-10-19).

(a) increase in methane intensity by species (%).
(b) additional methane by species (Gt CO2eq per year), total annotated.
(c) 2000-2024 trajectory of total additional methane with an inset giving the
    headcount-weighted exposure-day context.

Species order is identical in panels (a) and (b): 2024 additional methane,
descending.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as S
import fig_numbers as F

OUT = S.FIG_MAIN
NAME = "Fig5_methane_intensity"
CO2EQ = "CO$_2$eq"


def species_order(m):
    by = m["by_species"]
    return sorted(F.SPECIES, key=lambda sp: by[sp]["additional_GtCO2eq"],
                  reverse=True)


def panel_intensity(fig, m, order):
    ax = fig.add_axes([0.115, 0.615, 0.330, 0.330])
    by = m["by_species"]
    vals = np.array([by[sp]["increase_pct"] for sp in order])
    y = np.arange(len(order))[::-1]
    ax.barh(y, vals, height=0.60,
            color=[S.SPECIES_COLORS[sp] for sp in order],
            edgecolor="white", linewidth=0.5, zorder=2)
    for yi, v in zip(y, vals):
        ax.text(v + 0.07, yi, "%.2f" % v, va="center", ha="left",
                fontsize=S.note_size(S.FS_ANNOT), color="#1A1A1A")
    ax.set_yticks(y)
    ax.set_yticklabels([S.SPECIES_LABELS[sp] for sp in order],
                       fontsize=S.note_size(S.FS_TICK))
    ax.set_xlim(0, 4.3)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlabel("Increase in methane intensity (%)")
    ax.tick_params(axis="x", labelsize=S.note_size(S.FS_TICK))
    ax.tick_params(axis="y", length=0)
    S.panel_label(fig, ax, "a", dx=-0.100, dy=0.008)
    return ax


def panel_additional(fig, m, order):
    ax = fig.add_axes([0.605, 0.615, 0.345, 0.330])
    by = m["by_species"]
    vals = np.array([by[sp]["additional_GtCO2eq"] for sp in order])
    y = np.arange(len(order))[::-1]
    ax.barh(y, vals, height=0.60,
            color=[S.SPECIES_COLORS[sp] for sp in order],
            edgecolor="white", linewidth=0.5, zorder=2)
    for yi, v in zip(y, vals):
        ax.text(v + 0.0012, yi, "%.4f" % v, va="center", ha="left",
                fontsize=S.note_size(S.FS_ANNOT), color="#1A1A1A")
    ax.set_yticks(y)
    ax.set_yticklabels([S.SPECIES_LABELS[sp] for sp in order],
                       fontsize=S.note_size(S.FS_TICK))
    ax.set_xlim(0, 0.090)
    ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08])
    ax.set_xlabel("Additional methane (Gt " + CO2EQ + " per year)")
    ax.tick_params(axis="x", labelsize=S.note_size(S.FS_TICK))
    ax.tick_params(axis="y", length=0)
    ax.text(0.99, 0.04,
            "Total %.2f Gt %s per year\n(%.1f%% of livestock methane)"
            % (round(m["total_GtCO2eq"], 2), CO2EQ,
               m["share_of_livestock_ch4_pct"]),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=S.note_size(S.FS_ANNOT), color="#1A1A1A",
            linespacing=1.25)
    S.panel_label(fig, ax, "b", dx=-0.120, dy=0.008)
    return ax


def panel_trajectory(fig, m):
    ax = fig.add_axes([0.100, 0.085, 0.845, 0.395])
    yr = np.array(m["years"])
    vals = np.array(m["series_total_GtCO2eq"])
    ax.fill_between(yr, vals.min() * 0.55, vals, color=S.SEQ_COLORS[3],
                    alpha=0.35, lw=0, zorder=1)
    ax.plot(yr, vals, color=S.SERIES_A, lw=1.4, zorder=3)
    ax.plot(yr, vals, "o", ms=2.6, mfc="white", mec=S.SERIES_A, mew=0.6,
            zorder=4)
    ax.annotate("%.3f" % vals[0], (yr[0], vals[0]),
                textcoords="offset points", xytext=(4, 5),
                fontsize=S.note_size(S.FS_ANNOT), ha="left", va="bottom",
                color="#1A1A1A")
    ax.annotate("%.3f" % vals[-1], (yr[-1], vals[-1]),
                textcoords="offset points", xytext=(-4, 5),
                fontsize=S.note_size(S.FS_ANNOT), ha="right", va="bottom",
                color="#1A1A1A")
    change = (vals[-1] / vals[0] - 1) * 100
    ax.text(0.37, 0.955, "additional methane +%.1f%% since 2000" % change,
            transform=ax.transAxes, fontsize=S.note_size(S.FS_ANNOT),
            color=S.SERIES_A, ha="left", va="top")
    ax.set_xlim(1999.5, 2024.5)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2024])
    ax.set_xticklabels(["2000", "2005", "2010", "2015", "2020", "2024"],
                       fontsize=S.note_size(S.FS_TICK))
    ax.set_ylim(0.0800, 0.0950)
    ax.set_yticks([0.082, 0.084, 0.086, 0.088, 0.090, 0.092])
    ax.set_ylabel("Additional methane (Gt " + CO2EQ + " per year)")
    ax.set_xlabel("Year")
    ax.tick_params(axis="y", labelsize=S.note_size(S.FS_TICK))
    ax.grid(axis="y", color="#EDEDED", lw=0.4, zorder=0)
    ax.set_axisbelow(True)

    # inset: headcount-weighted mean exposure days (context for the trend)
    ins = ax.inset_axes([0.055, 0.600, 0.230, 0.380])
    expo = np.array(m["exposure_days_headweighted"])
    idx_years = [0, 12, 24]
    expo_vals = [expo[i] for i in idx_years]
    labels = ["2000", "2012", "2024"]
    xb = np.arange(len(idx_years))
    ins.bar(xb, expo_vals, width=0.6, color=S.SEQ_COLORS[5],
            edgecolor="white", linewidth=0.4, zorder=2)
    for xi, v in zip(xb, expo_vals):
        ins.text(xi, v + 4, "%.0f" % v, ha="center", va="bottom",
                 fontsize=S.note_size(S.FS_ANNOT), color="#1A1A1A")
    ins.set_xticks(xb)
    ins.set_xticklabels(labels, fontsize=S.note_size(S.FS_TICK))
    ins.set_ylim(0, 430)
    ins.set_yticks([0, 200])
    ins.tick_params(axis="y", labelsize=S.note_size(S.FS_TICK))
    ins.tick_params(axis="x", labelsize=S.note_size(S.FS_TICK))
    ins.text(0.5, 0.97, "Mean exposure days", transform=ins.transAxes,
             fontsize=S.note_size(S.FS_ANNOT), ha="center", va="top",
             color="#1A1A1A")
    ins.set_facecolor("none")
    for side in ("top", "right"):
        ins.spines[side].set_visible(False)
    S.panel_label(fig, ax, "c", dx=-0.078, dy=0.006)
    return ax


def main():
    S.setup_style()
    m = F.methane()
    order = species_order(m)
    fig = plt.figure(figsize=(S.FULL_W, 5.95))
    panel_intensity(fig, m, order)
    panel_additional(fig, m, order)
    panel_trajectory(fig, m)
    svg, png = S.save_figure(fig, NAME, OUT)
    print("saved", os.path.basename(svg), os.path.getsize(svg) // 1024, "KB")
    print("saved", os.path.basename(png), os.path.getsize(png) // 1024, "KB")


if __name__ == "__main__":
    main()
