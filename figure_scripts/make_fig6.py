"""
make_fig6.py -- Figure 6: vulnerability and adaptation limits.

(a) share of global cattle in the physical incompatibility area (PIA) at
    >25 %, >50 % and >75 % of days with THI >= 72.
(b) species yield-loss fraction at THI >= 72 versus the avoidable fraction.
(c) attainable cooling capacity by latitude / humidity zone (tropical,
    subtropical, temperate, boreal), with the cattle share of each zone.

All numbers come from fig_numbers.py.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as S
import fig_numbers as F

OUT = S.FIG_MAIN
NAME = "Fig6_vulnerability_limits"


def panel_pia(fig, pia):
    ax = fig.add_axes([0.095, 0.585, 0.325, 0.350])
    tiers = pia["tiers"]
    labels = ["Above 25%", "Above 50%", "Above 75%"]
    vals = [t["pct_global_cattle"] for t in tiers]
    x = np.arange(len(tiers))
    cols = [S.SEQ_COLORS[2], S.SEQ_COLORS[4], S.SEQ_COLORS[6]]
    ax.bar(x, vals, width=0.62, color=cols, edgecolor="white", linewidth=0.5,
           zorder=2)
    for xi, v, t in zip(x, vals, tiers):
        ax.text(xi, v + 1.2, "%.1f%%" % v, ha="center", va="bottom",
                fontsize=S.note_size(S.FS_ANNOT), color="#1A1A1A")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=S.note_size(S.FS_TICK))
    ax.set_ylim(0, 62)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel("Share of global cattle in PIA (%)")
    ax.set_xlabel("Share of days with THI ≥ 72")
    ax.tick_params(axis="y", labelsize=S.note_size(S.FS_TICK))
    ax.text(0.02, 0.97, "PIA: physical incompatibility area",
            transform=ax.transAxes, fontsize=S.note_size(S.FS_ANNOT),
            color="#4D4D4D", ha="left", va="top")
    S.panel_label(fig, ax, "a", dx=-0.095, dy=0.008)
    return ax


def panel_scatter(fig, lb):
    ax = fig.add_axes([0.585, 0.585, 0.355, 0.350])
    species = F.SPECIES
    x = np.array([lb["species"][sp]["yield_loss_fraction"] * 100
                  for sp in species])
    y = np.array([lb["species"][sp]["avoidable_fraction"] * 100
                  for sp in species])
    ax.scatter(x, y, s=42, c=[S.SPECIES_COLORS[sp] for sp in species],
               edgecolors="#2B2B2B", linewidths=0.6, zorder=3)
    offs = {
        "cattle": (6, 4, "left"),
        "chickens": (7, 1, "left"),
        "buffalo": (-6, 3, "right"),
        "goats": (6, -1, "left"),
        "pigs": (-6, 5, "right"),
        "sheep": (7, -9, "left"),
    }
    for i, sp in enumerate(species):
        dx, dy, ha = offs[sp]
        ax.annotate(S.SPECIES_LABELS[sp], (x[i], y[i]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    va="center", fontsize=S.note_size(S.FS_ANNOT),
                    color="#1A1A1A")
    ax.set_xlim(8, 27.5)
    ax.set_ylim(-1.2, 16.5)
    ax.set_xticks([10, 15, 20, 25])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel("Yield loss fraction at THI ≥ 72 (%)")
    ax.set_ylabel("Avoidable fraction of gross loss (%)")
    ax.tick_params(axis="both", labelsize=S.note_size(S.FS_TICK))
    S.panel_label(fig, ax, "b", dx=-0.100, dy=0.008)
    return ax


def panel_capacity(fig, cap):
    ax = fig.add_axes([0.115, 0.095, 0.830, 0.375])
    bands = cap["bands"]
    y = np.arange(len(bands))[::-1]
    vals = [b["capacity_degC"] for b in bands]
    head = [b["head_share_pct"] for b in bands]
    resid = [b["residual_exposure_pct"] for b in bands]
    cols = [S.SEQ_COLORS[2], S.SEQ_COLORS[3], S.SEQ_COLORS[5],
            S.SEQ_COLORS[6]]
    ax.barh(y, vals, height=0.55, color=cols, edgecolor="white",
            linewidth=0.5, zorder=2)
    for yi, v, h, r in zip(y, vals, head, resid):
        ax.text(v + 0.18, yi, "%.1f °C" % v, va="center", ha="left",
                fontsize=S.note_size(S.FS_ANNOT), color="#1A1A1A")
        ax.text(13.5, yi, "%.1f%%" % h, va="center", ha="right",
                fontsize=S.note_size(S.FS_ANNOT), color="#4D4D4D")
        ax.text(16.5, yi, "%.1f%%" % r, va="center", ha="right",
                fontsize=S.note_size(S.FS_ANNOT), color="#4D4D4D")
    ax.text(13.5, len(bands) - 0.22, "Cattle\nshare", ha="right",
            va="bottom", fontsize=S.note_size(S.FS_ANNOT), color="#4D4D4D",
            linespacing=1.1)
    ax.text(16.5, len(bands) - 0.22, "Residual\nexposure", ha="right",
            va="bottom", fontsize=S.note_size(S.FS_ANNOT), color="#4D4D4D",
            linespacing=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([b["band"] for b in bands],
                       fontsize=S.note_size(S.FS_TICK))
    ax.set_xlim(0, 17.0)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("Attainable cooling capacity (°C)")
    ax.tick_params(axis="x", labelsize=S.note_size(S.FS_TICK))
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.65, len(bands) + 0.32)
    ax.text(0.35, len(bands) - 0.22,
            "Evaporative cooling saturates in humid (tropical) zones:\n"
            "3.5 °C attainable versus 10 °C in temperate zones",
            ha="left", va="bottom",
            fontsize=S.note_size(S.FS_ANNOT), color="#1A1A1A",
            linespacing=1.25)
    S.panel_label(fig, ax, "c", dx=-0.075, dy=0.006)
    return ax


def main():
    S.setup_style()
    lb = F.loss_basis()
    pia = F.pia_data()
    cap = F.cooling_capacity_zones()
    fig = plt.figure(figsize=(S.FULL_W, 5.80))
    panel_pia(fig, pia)
    panel_scatter(fig, lb)
    panel_capacity(fig, cap)
    svg, png = S.save_figure(fig, NAME, OUT)
    print("saved", os.path.basename(svg), os.path.getsize(svg) // 1024, "KB")
    print("saved", os.path.basename(png), os.path.getsize(png) // 1024, "KB")


if __name__ == "__main__":
    main()
