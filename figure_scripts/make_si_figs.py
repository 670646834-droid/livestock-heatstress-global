"""
make_si_figs.py -- Supplementary Figures 1-3.

SI_Fig1: physical incompatibility area (PIA) and adaptation limits.
    (a) PIA share of global cattle at above 25 %, 50 % and 75 % of days with
        THI >= 72 (share of animals).
    (b) global PIA map (above 50 % of days) on Mollweide, nested with the
        above 25 % and above 75 % tiers for context.
    (c) PIA share by species (share of animals; values documented in SI S2.5).
SI_Fig2: cooling-measure cost and effectiveness ranges (SI S2.2) with the
    5-year payback break-even annotation.
SI_Fig3: China provincial heat-stress losses in 2024, top 15 provinces,
    stacked avoidable / residual.

All numbers come from fig_numbers.py.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as S
import fig_numbers as F

OUT = S.FIG_SI
PIA_LABEL = "Physical incompatibility area (share of animals)"


# ==========================================================================
# SI Figure 1: PIA
# ==========================================================================
def si1_panel_a(fig, pia):
    ax = fig.add_axes([0.085, 0.665, 0.315, 0.285])
    tiers = pia["tiers"]
    labels = ["Above 25%", "Above 50%", "Above 75%"]
    vals = [t["pct_global_cattle"] for t in tiers]
    x = np.arange(len(tiers))
    cols = [S.SEQ_COLORS[2], S.SEQ_COLORS[4], S.SEQ_COLORS[6]]
    ax.bar(x, vals, width=0.62, color=cols, edgecolor="white",
           linewidth=0.5, zorder=2)
    for xi, v, t in zip(x, vals, tiers):
        ax.text(xi, v + 1.5, "%.1f%%\n%.2f B head" % (v, t["cattle_in_pia_B"]),
                ha="center", va="bottom", fontsize=S.note_size(S.FS_ANNOT),
                color="#1A1A1A", linespacing=1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=S.note_size(S.FS_TICK))
    ax.set_ylim(0, 68)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel("Share of global cattle (%)")
    ax.set_xlabel("Days per year with THI ≥ 72")
    ax.tick_params(axis="y", labelsize=S.note_size(S.FS_TICK))
    S.panel_label(fig, ax, "a", dx=-0.075, dy=0.008)
    return ax


def si1_panel_b(fig, pia):
    mp = pia["map"]
    lat = mp["lat"]
    lon = mp["lon"]
    cls = (mp["pia_25pct"] + mp["pia_50pct"] + mp["pia_75pct"]).astype(float)
    cls = np.where(cls > 0, cls, np.nan)
    # light cartographic simplification: draw the 3 nested tiers
    ax = fig.add_axes([0.215, 0.115, 0.570, 0.420])
    S.map_frame(ax)
    X, Y = S.grid_xy(lat, lon)
    cmap = plt.matplotlib.colors.ListedColormap(
        [S.SEQ_COLORS[3], S.SEQ_COLORS[5], S.SEQ_COLORS[7]])
    S.add_simplified_contourf(ax, X, Y, np.ma.masked_invalid(cls),
                              levels=[0.5, 1.5, 2.5, 3.5], cmap=cmap,
                              eps=0.010, extend="neither", zorder=2)
    S.draw_coastline(ax)
    ax.text(0.5, 1.005, PIA_LABEL, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=S.note_size(S.FS_ANNOT), color="#4D4D4D")
    handles = [
        Patch(facecolor=S.SEQ_COLORS[3], edgecolor="none",
              label="Above 25% of days"),
        Patch(facecolor=S.SEQ_COLORS[5], edgecolor="none",
              label="Above 50% of days"),
        Patch(facecolor=S.SEQ_COLORS[7], edgecolor="none",
              label="Above 75% of days"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.50, 0.015), ncol=3,
               fontsize=S.note_size(S.FS_LEGEND), frameon=False,
               handlelength=1.0, handleheight=0.9, columnspacing=1.2,
               borderaxespad=0.0)
    fig.text(0.50, 0.065, "Days per year with THI ≥ 72", ha="center",
             va="bottom", fontsize=S.note_size(S.FS_ANNOT), color="#4D4D4D")
    S.panel_label(fig, ax, "b", dx=-0.090, dy=0.020)
    return ax


def si1_panel_c(fig, pia):
    ax = fig.add_axes([0.590, 0.665, 0.360, 0.285])
    shares = pia["species_share_pct"]
    order = sorted(shares, key=lambda sp: shares[sp], reverse=True)
    y = np.arange(len(order))[::-1]
    vals = [shares[sp] for sp in order]
    ax.barh(y, vals, height=0.60,
            color=[S.SPECIES_COLORS[sp] for sp in order],
            edgecolor="white", linewidth=0.5, zorder=2)
    for yi, v in zip(y, vals):
        ax.text(v + 1.4, yi, "%.1f" % v, va="center", ha="left",
                fontsize=S.note_size(S.FS_ANNOT), color="#1A1A1A")
    ax.set_yticks(y)
    ax.set_yticklabels([S.SPECIES_LABELS[sp] for sp in order],
                       fontsize=S.note_size(S.FS_TICK))
    ax.set_xlim(0, 75)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_xlabel("Share of animals in physical\nincompatibility area (%)")
    ax.tick_params(axis="x", labelsize=S.note_size(S.FS_TICK))
    ax.tick_params(axis="y", length=0)
    S.panel_label(fig, ax, "c", dx=-0.090, dy=0.008)
    return ax


def make_si1():
    pia = F.pia_data()
    fig = plt.figure(figsize=(S.FULL_W, 5.35))
    si1_panel_a(fig, pia)
    si1_panel_c(fig, pia)
    si1_panel_b(fig, pia)
    svg, png = S.save_figure(fig, "SI_Fig1_PIA_adaptation_limits", OUT)
    print("saved", os.path.basename(svg), os.path.getsize(svg) // 1024, "KB")
    print("saved", os.path.basename(png), os.path.getsize(png) // 1024, "KB")


# ==========================================================================
# SI Figure 2: cooling-measure cost vs effectiveness (implemented model)
# ==========================================================================
def make_si2():
    cool = F.cooling()
    measures = cool["measures"]
    loss = cool["loss_per_head_usd"]
    fig = plt.figure(figsize=(S.FULL_W, 5.05))

    # panel (a): capital cost per head and the 5-year payback break-even cost
    ax1 = fig.add_axes([0.190, 0.575, 0.790, 0.365])
    ys = np.arange(len(measures))[::-1]
    for y, m in zip(ys, measures):
        ax1.plot([10, m["cost"]], [y, y], color="#4D4D4D", lw=1.0, zorder=2)
        ax1.scatter([m["cost"]], [y], s=26, facecolor=S.SERIES_A,
                    edgecolor="#2B2B2B", linewidth=0.5, zorder=3)
        ax1.scatter([m["breakeven_cost"]], [y], s=24, marker="D",
                    facecolor="white", edgecolor="#8A3B4A", linewidth=0.9,
                    zorder=4)
        ax1.text(m["cost"] * 1.13, y, "%d" % round(m["cost"]),
                 va="center", ha="left", fontsize=S.note_size(S.FS_ANNOT),
                 color="#1A1A1A")
    ax1.set_xscale("log")
    ax1.set_xlim(10, 6000)
    ax1.set_xticks([10, 20, 50, 100, 200, 500, 1000, 2000])
    ax1.set_xticklabels(["10", "20", "50", "100", "200", "500", "1000",
                         "2000"])
    ax1.set_yticks(ys)
    ax1.set_yticklabels([m["name"] for m in measures],
                        fontsize=S.note_size(S.FS_TICK))
    ax1.set_ylim(-0.75, len(measures) - 0.25)
    ax1.set_xlabel("Capital cost (USD per head, log scale)")
    ax1.tick_params(axis="x", labelsize=S.note_size(S.FS_TICK))
    ax1.tick_params(axis="y", length=0)
    ax1.grid(axis="x", color="#EDEDED", lw=0.4, zorder=0)
    ax1.set_axisbelow(True)
    ax1.text(0.985, 0.025,
             "diamond: 5-year payback break-even cost "
             "(5 × annual avoided loss)",
             transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=S.note_size(S.FS_ANNOT), color="#8A3B4A")
    S.panel_label(fig, ax1, "a", dx=-0.155, dy=0.008)

    # panel (b): cooling effectiveness
    ax2 = fig.add_axes([0.190, 0.095, 0.790, 0.365])
    for y, m in zip(ys, measures):
        ax2.plot([0, m["eff"]], [y, y], color="#4D4D4D", lw=1.0, zorder=2)
        ax2.scatter([m["eff"]], [y], s=26, facecolor=S.SERIES_B,
                    edgecolor="#2B2B2B", linewidth=0.5, zorder=3)
        ax2.text(m["eff"] + 1.6, y, "%d" % round(m["eff"]),
                 va="center", ha="left", fontsize=S.note_size(S.FS_ANNOT),
                 color="#1A1A1A")
    ax2.set_yticks(ys)
    ax2.set_yticklabels([m["name"] for m in measures],
                        fontsize=S.note_size(S.FS_TICK))
    ax2.set_ylim(-1.35, len(measures) - 0.25)
    ax2.set_xlim(0, 95)
    ax2.set_xticks([0, 20, 40, 60, 80])
    ax2.set_xlabel("Cooling effectiveness (% of heat-stress loss averted)")
    ax2.tick_params(axis="x", labelsize=S.note_size(S.FS_TICK))
    ax2.tick_params(axis="y", length=0)
    ax2.grid(axis="x", color="#EDEDED", lw=0.4, zorder=0)
    ax2.set_axisbelow(True)
    ax2.text(0.985, 0.025,
             "Break-even: cost = 5 × annual avoided loss\n"
             "(mean 2024 cattle loss %.0f USD per head per year)" % loss,
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=S.note_size(S.FS_ANNOT), color="#4D4D4D",
             linespacing=1.25)
    S.panel_label(fig, ax2, "b", dx=-0.155, dy=0.008)

    svg, png = S.save_figure(fig, "SI_Fig2_cooling_cost_effectiveness", OUT)
    print("saved", os.path.basename(svg), os.path.getsize(svg) // 1024, "KB")
    print("saved", os.path.basename(png), os.path.getsize(png) // 1024, "KB")


# ==========================================================================
# SI Figure 3: China provincial losses
# ==========================================================================
def make_si3():
    cn = F.china()
    rows = cn["rows"]
    n = len(rows)
    fig = plt.figure(figsize=(S.FULL_W, 4.55))
    ax = fig.add_axes([0.150, 0.095, 0.790, 0.855])
    y = np.arange(n)[::-1]
    gross = np.array([r["gross_B"] for r in rows])
    avoid = np.array([r["avoidable_B"] for r in rows])
    resid = np.array([r["residual_B"] for r in rows])
    ax.barh(y, resid, height=0.66, color=S.RESID_COL, edgecolor="white",
            linewidth=0.4, zorder=2, label="Residual")
    ax.barh(y, avoid, left=resid, height=0.66, color=S.AVOID_COL,
            edgecolor="white", linewidth=0.4, zorder=3, label="Avoidable")
    for yi, g in zip(y, gross):
        ax.text(g + 0.018, yi, "%.2f" % g, va="center", ha="left",
                fontsize=S.note_size(S.FS_ANNOT), color="#1A1A1A")
    ax.set_yticks(y)
    ax.set_yticklabels([r["province"] for r in rows],
                       fontsize=S.note_size(S.FS_TICK))
    ax.set_xlim(0, 2.55)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlabel("Heat-stress loss (B USD, 2005)")
    ax.tick_params(axis="x", labelsize=S.note_size(S.FS_TICK))
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#EDEDED", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=S.note_size(S.FS_LEGEND),
              handlelength=1.0, handleheight=0.9, labelspacing=0.25,
              borderaxespad=0.2, bbox_to_anchor=(1.0, 0.0))
    ax.text(0.985, 0.135,
            "China total %.2f B USD; top 15 provinces shown\n"
            "avoidable share %.1f%%"
            % (cn["total_gross_B"], cn["total_avoidable_fraction"] * 100),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=S.note_size(S.FS_ANNOT), color="#1A1A1A",
            linespacing=1.25)
    S.panel_label(fig, ax, "a", dx=-0.115, dy=0.006)
    svg, png = S.save_figure(fig, "SI_Fig3_china_provincial_losses", OUT)
    print("saved", os.path.basename(svg), os.path.getsize(svg) // 1024, "KB")
    print("saved", os.path.basename(png), os.path.getsize(png) // 1024, "KB")


def main():
    S.setup_style()
    make_si1()
    make_si2()
    make_si3()


if __name__ == "__main__":
    main()
