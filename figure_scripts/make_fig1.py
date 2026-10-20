"""
make_fig1.py -- Figure 1: global distribution of cattle heat-stress losses
(2024, THI >= 72, 2005 USD).

(a) Mollweide map of 2024 cattle gross loss per 0.5 degree cell, computed as
    weight = cattle_head * pw_loss_pct / 100 and scaled to the adopted cattle
    total of US$142.45 billion; Natural Earth 110m coastline overlay.
(b) cattle-head share versus loss share by climate zone (dumbbell chart).
(c) species composition of the 2024 gross loss with avoidable/residual split.

All numbers come from fig_numbers.py.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as S
import fig_numbers as F

OUT = S.FIG_MAIN
NAME = "Fig1_global_distribution"


def panel_map(fig):
    mp = F.map_cattle_loss()
    ax = fig.add_axes([0.049, 0.565, 0.902, 0.428])
    S.map_frame(ax)
    X, Y = S.grid_xy(mp["lat"], mp["lon"])
    Z = np.ma.masked_invalid(mp["log10_loss_usd"])
    S.add_simplified_contourf(ax, X, Y, Z, levels=mp["levels_log10"],
                              eps=0.012, extend="both", zorder=2)
    S.draw_coastline(ax)

    lon_ticks = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    cax = fig.add_axes([0.315, 0.526, 0.37, 0.0105])
    S.spec_colorbar(
        cax, mp["levels_log10"], S.SEQ_CMAP, extend="both",
        ticks=np.log10(lon_ticks),
        label="Cattle loss (million USD per 0.5\u00b0 cell, log scale)",
    )
    cax.set_xticklabels(["0.001", "0.01", "0.1", "1", "10", "100"])

    hot = [
        ("South Asia", 78.0, 24.0, 0.760, 0.380),
        ("sub-Saharan Africa", 12.0, 10.0, 0.440, 0.290),
        ("Latin America", -52.0, -12.0, 0.185, 0.300),
    ]
    for label, lon, lat, tx, ty in hot:
        hx, hy = S.mollweide(lon, lat)
        ax.annotate(
            label, xy=(float(hx), float(hy)),
            xycoords="data", xytext=(tx, ty), textcoords="axes fraction",
            fontsize=S.note_size(S.FS_ANNOT), ha="center", va="center",
            color="#1A1A1A",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#4D4D4D",
                            shrinkA=1.5, shrinkB=2.5),
            zorder=8,
        )
    S.panel_label(fig, ax, "a", dx=-0.030, dy=0.006)
    return ax


def panel_dumbbell(fig, cz):
    ax = fig.add_axes([0.100, 0.085, 0.335, 0.365])
    zones = cz["zones"]
    n = len(zones)
    ys = np.arange(n)[::-1]
    head = np.array([z["head_share_pct"] for z in zones])
    loss = np.array([z["loss_share_pct"] for z in zones])
    for i, y in zip(range(n), ys):
        ax.plot([head[i], loss[i]], [y, y], color="#8C8C8C", lw=0.9,
                solid_capstyle="round", zorder=1)
    ax.scatter(head, ys, s=26, facecolor="white", edgecolor="#4D4D4D",
               linewidth=0.7, zorder=3)
    ax.scatter(loss, ys, s=26, marker="s",
               c=[S.ZONE_COLORS[z["zone"]] for z in zones],
               edgecolors="#2F2F2F", linewidths=0.6, zorder=4)
    for i, y in zip(range(n), ys):
        hi, li = head[i], loss[i]
        ax.annotate("%.1f" % li, (li, y), textcoords="offset points",
                    xytext=(4, 3), ha="left", va="bottom",
                    fontsize=S.note_size(S.FS_ANNOT), color="#1A1A1A")
        ax.annotate("%.1f" % hi, (hi, y), textcoords="offset points",
                    xytext=(-4, -9), ha="right", va="top",
                    fontsize=S.note_size(S.FS_ANNOT), color="#4D4D4D")
    ax.set_yticks(ys)
    ax.set_yticklabels([S.ZONE_LABELS[z["zone"]] for z in zones],
                       fontsize=S.note_size(S.FS_TICK))
    ax.set_xlim(-2, 84)
    ax.set_ylim(-0.55, n - 0.45)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("Share of global total (%)")
    ax.tick_params(axis="x", labelsize=S.note_size(S.FS_TICK))
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E8E8E8", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    handles = [
        Line2D([], [], marker="o", ls="", markerfacecolor="white",
               markeredgecolor="#4D4D4D", markersize=4.5,
               label="Cattle head share"),
        Line2D([], [], marker="s", ls="", markerfacecolor="#8AAEAB",
               markeredgecolor="#2F2F2F", markersize=4.5,
               label="Loss share"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=S.note_size(
        S.FS_LEGEND), handletextpad=0.2, borderaxespad=0.1,
        bbox_to_anchor=(1.02, -0.03))
    S.panel_label(fig, ax, "b", dx=-0.080, dy=0.008)
    return ax


def panel_composition(fig, lb):
    ax = fig.add_axes([0.640, 0.085, 0.330, 0.365])
    rows = lb["table1"]  # descending gross loss
    n = len(rows)
    ys = np.arange(n)[::-1]
    for i, y in zip(range(n), ys):
        r = rows[i]
        frac = r["avoidable_fraction_pct"] / 100.0
        col = S.SPECIES_COLORS[r["species"]]
        light = tuple(0.42 * np.array(S.mcolors.to_rgb(col))
                      + 0.58 * np.ones(3))
        ax.barh(y, (1.0 - frac) * 100.0, height=0.62, color=light,
                edgecolor="white", linewidth=0.5, zorder=2)
        ax.barh(y, frac * 100.0, left=(1.0 - frac) * 100.0, height=0.62,
                color=col, edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(83.5, y, "%.1f" % r["gross_B"], va="center", ha="right",
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
    ax.legend(handles=[
        Patch(facecolor="#C9C9C9", edgecolor="white", label="Residual"),
        Patch(facecolor="#6E6E6E", edgecolor="white", label="Avoidable"),
    ], loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2,
        fontsize=S.note_size(S.FS_LEGEND), handlelength=1.0,
        handleheight=0.9, columnspacing=1.0, borderaxespad=0.0)
    fig.text(0.600, 0.462, "Bar totals (B USD)", ha="left", va="bottom",
             fontsize=S.note_size(S.FS_ANNOT), color="#4D4D4D")
    S.panel_label(fig, ax, "c", dx=-0.100, dy=0.008)
    return ax


def main():
    S.setup_style()
    lb = F.loss_basis()
    cz = F.climate_zones()
    fig = plt.figure(figsize=(S.FULL_W, 6.85))
    panel_map(fig)
    panel_dumbbell(fig, cz)
    panel_composition(fig, lb)
    svg, png = S.save_figure(fig, NAME, OUT)
    print("saved", os.path.basename(svg), os.path.getsize(svg) // 1024, "KB")
    print("saved", os.path.basename(png), os.path.getsize(png) // 1024, "KB")


if __name__ == "__main__":
    main()
