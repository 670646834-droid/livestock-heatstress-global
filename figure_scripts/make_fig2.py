"""
make_fig2.py -- Figure 2: adaptation feasibility.

(a) avoidable and residual 2024 loss by species, as a 100 % composition bar
    with absolute totals annotated (adopted Table 1 basis).
(b) absolute avoidable loss and avoidable fraction for 3, 5 and 10 year
    payback horizons.
(c) cost-effectiveness of the six cooling measures implemented in the revised
    model (shade 65 USD/head 15 %, fans 19 USD 17 %, evaporative pads 150 USD
    50 %, sprinkler-fan systems 275 USD 65 %, smart sprinklers 300 USD 65 %,
    tunnel ventilation 2000 USD 80 %), with the five-year payback break-even
    line (cost = 5 x annual avoided loss, mean 2024 cattle loss 90 USD/head).

All numbers come from fig_numbers.py.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as S
import fig_numbers as F

OUT = S.FIG_MAIN
NAME = "Fig2_adaptation_feasibility"


def panel_species(fig, lb):
    ax = fig.add_axes([0.135, 0.565, 0.340, 0.345])
    rows = lb["table1"]           # descending gross loss
    n = len(rows)
    ys = np.arange(n)[::-1]
    for i, y in zip(range(n), ys):
        r = rows[i]
        frac = r["avoidable_fraction_pct"] / 100.0
        ax.barh(y, (1.0 - frac) * 100.0, height=0.62, color=S.RESID_COL,
                edgecolor="white", linewidth=0.5, zorder=2)
        ax.barh(y, frac * 100.0, left=(1.0 - frac) * 100.0, height=0.62,
                color=S.AVOID_COL, edgecolor="white", linewidth=0.5,
                zorder=3)
        ax.text(83.0, y, "%.2f" % r["gross_B"], va="center", ha="right",
                fontsize=S.note_size(S.FS_ANNOT), color="#1A1A1A", zorder=5)
    ax.set_yticks(ys)
    ax.set_yticklabels([S.SPECIES_LABELS[r["species"]] for r in rows],
                       fontsize=S.note_size(S.FS_TICK))
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of species gross loss (%)")
    ax.tick_params(axis="x", labelsize=S.note_size(S.FS_TICK))
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    fig.text(0.295, 0.925, "Gross loss (B USD, 2005)",
             ha="left", va="bottom", fontsize=S.note_size(S.FS_ANNOT),
             color="#4D4D4D")
    fig.legend(handles=[
        Patch(facecolor=S.RESID_COL, edgecolor="white", label="Residual"),
        Patch(facecolor=S.AVOID_COL, edgecolor="white", label="Avoidable"),
    ], loc="lower left", bbox_to_anchor=(0.105, 0.468), ncol=2,
        fontsize=S.note_size(S.FS_LEGEND), frameon=False,
        handlelength=1.1, handleheight=0.9, columnspacing=1.2,
        borderaxespad=0.0)
    S.panel_label(fig, ax, "a", dx=-0.100, dy=0.008)
    return ax


def panel_payback(fig, lb):
    ax = fig.add_axes([0.630, 0.565, 0.330, 0.345])
    tags = ["3yr", "5yr", "10yr"]
    labels = ["3 year", "5 year", "10 year"]
    vals = np.array([lb["payback"][t]["avoidable_B"] for t in tags])
    fr = np.array([lb["payback"][t]["avoidable_fraction"] * 100 for t in tags])
    x = np.arange(len(tags))
    ax.vlines(x, 0, vals, color="#8C8C8C", lw=0.9, zorder=1)
    ax.scatter(x, vals, s=34, facecolor=S.AVOID_COL, edgecolor="#24523B",
               linewidth=0.6, zorder=3)
    for i in range(len(tags)):
        ax.text(x[i], vals[i] + 1.1, "%.2f" % vals[i], ha="center",
                va="bottom", fontsize=S.note_size(S.FS_ANNOT),
                color="#1A1A1A")
        ax.text(x[i], vals[i] - 1.8, "%.1f%%" % fr[i], ha="center", va="top",
                fontsize=S.note_size(S.FS_ANNOT), color="#24523B")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=S.note_size(S.FS_TICK))
    ax.set_xlim(-0.55, len(tags) - 0.45)
    ax.set_ylim(0, 46)
    ax.set_ylabel("Avoidable loss (B USD, 2005)")
    ax.set_xlabel("Payback horizon")
    ax.tick_params(axis="y", labelsize=S.note_size(S.FS_TICK))
    ax.text(0.03, 0.97, "green: avoidable share",
            transform=ax.transAxes, fontsize=S.note_size(S.FS_ANNOT),
            color="#24523B", ha="left", va="top")
    S.panel_label(fig, ax, "b", dx=-0.090, dy=0.008)
    return ax


def panel_cost_curve(fig, cooling):
    ax = fig.add_axes([0.098, 0.085, 0.862, 0.360])
    L = cooling["loss_per_head_usd"]
    payback = cooling["payback_years"]
    measures = cooling["measures"]

    # 5-year payback break-even: cost = payback * loss_per_head * eff
    cost_grid = np.logspace(np.log10(8), np.log10(4000), 300)
    eff_break = 100.0 * cost_grid / (payback * L)
    ax.fill_between(cost_grid, eff_break, 90.0, where=eff_break <= 90.0,
                    color=S.AVOID_COL, alpha=0.10, lw=0, zorder=0)
    vis = eff_break <= 90.0
    ax.plot(cost_grid[vis], eff_break[vis], color="#2F6B4F", lw=1.0,
            ls=(0, (4, 2)), zorder=2)

    palette = [S.SEQ_COLORS[3], S.SEQ_COLORS[4], S.SEQ_COLORS[5],
               S.SEQ_COLORS[6], S.SEQ_COLORS[7], "#355468"]
    label_pos = {
        "Fans": (0.0, 7.0, "center", "bottom"),
        "Shade": (0.0, -9.0, "center", "top"),
        "Evaporative pads": (0.0, 7.0, "center", "bottom"),
        "Sprinkler-fan systems": (0.0, 7.0, "center", "bottom"),
        "Smart sprinklers": (8.0, -2.0, "left", "center"),
        "Tunnel ventilation": (-6.0, -8.0, "right", "top"),
    }
    for i, m in enumerate(measures):
        ax.scatter(m["cost"], m["eff"], s=38, facecolor=palette[i],
                   edgecolor="#2B2B2B", linewidth=0.7, zorder=4)
        dx, dy, ha, va = label_pos[m["name"]]
        ax.annotate(m["name"], (m["cost"], m["eff"]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va=va, fontsize=S.note_size(S.FS_ANNOT),
                    color="#1A1A1A", zorder=5)
    ax.text(9.2, 87.5,
            "above dashed line: 5-year payback met\n"
            "below: not cost-effective",
            fontsize=S.note_size(S.FS_ANNOT), color="#2F6B4F", ha="left",
            va="top", linespacing=1.25, zorder=5)

    ax.set_xscale("log")
    ax.set_xlim(8, 4000)
    ax.set_ylim(0, 90)
    ax.set_xticks([10, 20, 50, 100, 200, 500, 1000, 2000])
    ax.set_xticklabels(["10", "20", "50", "100", "200", "500", "1000",
                        "2000"])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_xlabel("Capital cost (USD per head, log scale)")
    ax.set_ylabel("Cooling effectiveness (%)")
    ax.tick_params(axis="both", labelsize=S.note_size(S.FS_TICK))
    ax.grid(axis="y", color="#EDEDED", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    S.panel_label(fig, ax, "c", dx=-0.088, dy=0.006)
    return ax


def main():
    S.setup_style()
    lb = F.loss_basis()
    cooling = F.cooling()
    fig = plt.figure(figsize=(S.FULL_W, 5.75))
    panel_species(fig, lb)
    panel_payback(fig, lb)
    panel_cost_curve(fig, cooling)
    svg, png = S.save_figure(fig, NAME, OUT)
    print("saved", os.path.basename(svg), os.path.getsize(svg) // 1024, "KB")
    print("saved", os.path.basename(png), os.path.getsize(png) // 1024, "KB")


if __name__ == "__main__":
    main()
