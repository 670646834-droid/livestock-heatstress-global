"""
figstyle.py -- shared plotting style, palette and geometry helpers for the
Nature Food revision figures (09_Figures/main and 09_Figures/si).

Style law enforced here:
  * Arial only, no bold anywhere, smallest text 8 pt at final printed size
    (axis labels 9 pt, ticks/annotations 8 pt, panel letters 9 pt).
  * No em/en dashes in any figure text.
  * Low-saturation colour-blind-safe palette (Paul Tol muted subset),
    at most six categorical species colours, consistent across figures.
  * SVG stays vector (svg.fonttype='none'), <= 500 KB; PNG 300 dpi, RGB.
  * A4 page: full width 165 mm, single column 80 mm.

Geometry helpers implement a Mollweide projection and a pure-Python
Natural Earth 110m coastline reader so that no cartopy/netCDF4 is needed.
"""
from __future__ import annotations

import os
import struct

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Ellipse
import matplotlib.colors as mcolors

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "05_Data")
SRC = os.path.join(ROOT, "11_Source", "data_prep")
FIG = os.path.join(ROOT, "09_Figures")
FIG_MAIN = os.path.join(FIG, "main")
FIG_SI = os.path.join(FIG, "si")
COAST_SHP = os.path.join(
    os.path.expanduser("~"),
    ".local", "share", "cartopy", "shapefiles", "natural_earth",
    "physical", "ne_110m_coastline.shp",
)

# --------------------------------------------------------------------------
# physical size helpers (A4)
# --------------------------------------------------------------------------
MM = 1.0 / 25.4
FULL_W = 165.0 * MM          # 6.496 in
COL_W = 80.0 * MM            # 3.150 in
HALF_W = 80.0 * MM

# text sizes (pt at final size); nothing below 8
FS_TICK = 8.0
FS_ANNOT = 8.0
FS_LABEL = 9.0
FS_LEGEND = 8.0
FS_PANEL = 9.0

# --------------------------------------------------------------------------
# palette: Paul Tol muted (colour-blind safe, low saturation)
# colour order is fixed to the 2024 gross-loss ranking so every figure uses
# the same species colour: cattle, chickens, buffalo, goats, pigs, sheep
# --------------------------------------------------------------------------
SPECIES_ORDER = ["cattle", "chickens", "buffalo", "goats", "pigs", "sheep"]
SPECIES_LABELS = {
    "cattle": "Cattle",
    "chickens": "Chickens",
    "buffalo": "Buffalo",
    "goats": "Goats",
    "pigs": "Pigs",
    "sheep": "Sheep",
}
SPECIES_COLORS = {
    "cattle": "#CC6677",    # Tol muted rose
    "chickens": "#DDCC77",  # Tol muted sand
    "buffalo": "#332288",   # Tol muted indigo
    "goats": "#999933",     # Tol muted olive
    "pigs": "#88CCEE",      # Tol muted cyan
    "sheep": "#44AA99",     # Tol muted teal
}
# semantic colours for the avoidable / residual decomposition
AVOID_COL = "#4F8A6B"       # muted sage green
RESID_COL = "#BFC4CA"       # muted light grey
GROSS_COL = "#4A4A4A"
# neutral accents
GREY_D = "#4D4D4D"
GREY_M = "#808080"
GREY_L = "#D9D9D9"
RULE_COL = "#7A7A7A"

# sequential map colormap: pale sand -> muted teal/blue, low saturation
SEQ_COLORS = [
    "#F4EFE4", "#E6DCC8", "#CFD6C6", "#AFC4BC",
    "#8AAEAB", "#66929A", "#4A7485", "#355468",
]
SEQ_CMAP = LinearSegmentedColormap.from_list("sand_teal", SEQ_COLORS)

# climate-zone colours (4 zones, sequential low-saturation blue)
ZONE_COLORS = {
    "tropical": "#8AAEAB",
    "subtropical": "#66929A",
    "temperate": "#AFC4BC",
    "boreal": "#D8DFD8",
}
ZONE_LABELS = {
    "tropical": "Tropical",
    "subtropical": "Subtropical",
    "temperate": "Temperate",
    "boreal": "Boreal",
}

# two-series accent colours (semantic, not species)
SERIES_A = "#4A7485"   # muted blue
SERIES_B = "#CC6677"   # muted rose


def setup_style():
    """Install the journal style; call once per figure script."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "font.size": FS_ANNOT,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "patch.linewidth": 0.5,
        "errorbar.capsize": 2.0,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",
        "mathtext.default": "it",
        "svg.fonttype": "none",          # keep text as text (selectable)
        "svg.hashsalt": "livestock-heatstress-2024-figs",
        "path.simplify": True,
        "path.simplify_threshold": 0.8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# --------------------------------------------------------------------------
# generic save + QA-friendly bookkeeping
# --------------------------------------------------------------------------
FONT_SIZES_USED = set()
QA_RECORDS = []


def note_size(size):
    """Record a font size that is used programmatically (for QA)."""
    FONT_SIZES_USED.add(float(size))
    return float(size)


def _to_rgb_png(path):
    """Force RGB (no alpha) and 300 dpi metadata on a saved PNG."""
    from PIL import Image
    with Image.open(path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        im.save(path, format="PNG", dpi=(300, 300), optimize=True)


def _bbox_area(bb):
    w = max(0.0, bb.x1 - bb.x0)
    h = max(0.0, bb.y1 - bb.y0)
    return w * h


def _intersect(a, b):
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _contains(outer, inner, tol=0.5):
    return (outer.x0 - tol <= inner.x0 and outer.y0 - tol <= inner.y0
            and outer.x1 + tol >= inner.x1 and outer.y1 + tol >= inner.y1)


def audit_figure(fig, name):
    """Programmatic layout audit: text bounds, text-text overlaps, panels."""
    from matplotlib.text import Text
    from matplotlib.axes import Axes

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bb = fig.bbox
    texts = [t for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()
             and t.get_text().strip() != ""]
    bounds_bad = []
    text_boxes = []
    for t in texts:
        try:
            bb = t.get_window_extent(renderer)
        except Exception:
            continue
        if (bb.x0 < fig_bb.x0 - 1.0 or bb.y0 < fig_bb.y0 - 1.0
                or bb.x1 > fig_bb.x1 + 1.0 or bb.y1 > fig_bb.y1 + 1.0):
            bounds_bad.append({"text": t.get_text()[:40],
                               "bbox": [round(v, 1) for v in bb.extents]})
        text_boxes.append((t, bb))

    overlaps = []
    for i in range(len(text_boxes)):
        ti, bi = text_boxes[i]
        for j in range(i + 1, len(text_boxes)):
            tj, bj = text_boxes[j]
            area = _intersect(bi, bj)
            if area > 3.0:
                overlaps.append({
                    "a": ti.get_text()[:32], "b": tj.get_text()[:32],
                    "area_px2": round(area, 1),
                })

    axes_list = list(fig.axes)
    axes_bb = [(a, a.get_window_extent(renderer)) for a in axes_list]
    axes_overlap = []
    for i in range(len(axes_bb)):
        ai, bi = axes_bb[i]
        for j in range(i + 1, len(axes_bb)):
            aj, bj = axes_bb[j]
            if _contains(bi, bj) or _contains(bj, bi):
                continue
            area = _intersect(bi, bj)
            if area > 4.0:
                axes_overlap.append({"ax1": ai.get_label(),
                                     "ax2": aj.get_label(),
                                     "area_px2": round(area, 1)})

    # text of one axes must not intrude into another (non-nested) axes
    intrude = []
    for t, bb in text_boxes:
        ta = getattr(t, "axes", None)
        for a, ab in axes_bb:
            if a is ta:
                continue
            if _contains(ab, bb) or _contains(bb, ab):
                continue
            area = _intersect(bb, ab)
            if area > 6.0:
                intrude.append({"text": t.get_text()[:32],
                                "axes": a.get_label(),
                                "area_px2": round(area, 1)})

    panel_texts = sorted(t.get_text().strip() for t, _ in text_boxes
                         if t.get_text().strip().startswith("(")
                         and t.get_text().strip().endswith(")")
                         and len(t.get_text().strip()) == 3)
    fontsizes = []
    bold_texts = []
    for t in texts:
        try:
            fontsizes.append(round(float(t.get_fontsize()), 2))
            if str(t.get_fontweight()).lower() in ("bold", "700", "heavy",
                                                   "black"):
                bold_texts.append(t.get_text()[:40])
        except Exception:
            pass
    rec = {
        "name": name,
        "fig_size_in": [round(fig.get_figwidth(), 4),
                        round(fig.get_figheight(), 4)],
        "panel_letters": panel_texts,
        "n_text": len(text_boxes),
        "min_fontsize": min(fontsizes) if fontsizes else None,
        "bold_texts": bold_texts,
        "text_out_of_bounds": bounds_bad,
        "text_overlaps": overlaps,
        "axes_overlaps": axes_overlap,
        "text_axes_intrusions": intrude,
        "palette_species": dict(SPECIES_COLORS),
        "palette_components": {"avoidable": AVOID_COL,
                               "residual": RESID_COL},
        "seq_cmap": list(SEQ_COLORS),
    }
    QA_RECORDS.append(rec)
    return rec


def save_figure(fig, name, outdir):
    """Audit, then save SVG (vector text) + PNG (300 dpi, RGB)."""
    os.makedirs(outdir, exist_ok=True)
    audit_figure(fig, name)
    svg = os.path.join(outdir, name + ".svg")
    png = os.path.join(outdir, name + ".png")
    fig.savefig(svg, format="svg", facecolor="white", edgecolor="white")
    fig.savefig(png, format="png", dpi=300, facecolor="white")
    _to_rgb_png(png)
    plt.close(fig)
    return svg, png


# --------------------------------------------------------------------------
# text helpers
# --------------------------------------------------------------------------
def panel_label(fig, ax, letter, dx=-0.045, dy=0.012):
    """(a), (b) ... 9 pt regular, placed at the top-left of a panel.

    Positions are clamped so that the label always stays inside the canvas.
    """
    pos = ax.get_position()
    x = max(pos.x0 + dx, 0.004)
    y = min(pos.y1 + dy, 0.977)
    fig.text(x, y, f"({letter})",
             fontsize=note_size(FS_PANEL), fontweight="normal",
             ha="left", va="bottom", color="#111111")


def zone_label(ax, x, y, text, **kw):
    """Plain 8 pt annotation text."""
    kw.setdefault("fontsize", note_size(FS_ANNOT))
    kw.setdefault("ha", "left")
    kw.setdefault("va", "bottom")
    kw.setdefault("color", "#222222")
    return ax.text(x, y, text, **kw)


# italic P value helper: '$P$' is italic through mathtext mathdefault it
def ptext(p):
    """Return a compact italic-P string with a comparison sign."""
    if p < 0.001:
        return r"$\it{P}$ < 0.001"
    return r"$\it{P}$ = %.3f" % p


def r2text(r2, label="R"):
    return r"$\it{%s}^{2}$ = %.3f" % (label, r2)


# --------------------------------------------------------------------------
# Mollweide projection (manual, no cartopy)
# --------------------------------------------------------------------------
def mollweide(lon_deg, lat_deg):
    """Mollweide projection of lon/lat in degrees; returns x, y (R = 1)."""
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    lam = np.radians(lon)
    phi = np.radians(lat)
    theta = np.array(phi, dtype=float)
    for _ in range(30):
        denom = 2.0 + 2.0 * np.cos(2.0 * theta)
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        theta = theta - (2.0 * theta + np.sin(2.0 * theta)
                         - np.pi * np.sin(phi)) / denom
    x = (2.0 * np.sqrt(2.0) / np.pi) * lam * np.cos(theta)
    y = np.sqrt(2.0) * np.sin(theta)
    return x, y


MOLL_XLIM = 2.0 * np.sqrt(2.0)
MOLL_YLIM = np.sqrt(2.0)


def map_frame(ax, graticule=True):
    """Draw the Mollweide outline (and optional light graticule)."""
    ax.set_xlim(-MOLL_XLIM, MOLL_XLIM)
    ax.set_ylim(-MOLL_YLIM, MOLL_YLIM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(False)
    if graticule:
        t = np.linspace(0, 2 * np.pi, 241)
        ax.plot(MOLL_XLIM * np.cos(t), MOLL_YLIM * np.sin(t),
                color="#9A9A9A", lw=0.45, zorder=5, clip_on=False)
        for lat0 in (-60, -30, 0, 30, 60):
            lons = np.linspace(-180, 180, 361)
            xs, ys = mollweide(lons, np.full_like(lons, lat0))
            ax.plot(xs, ys, color="#E2E2E2", lw=0.3, zorder=0)
        for lon0 in range(-120, 121, 60):
            lats = np.linspace(-89, 89, 181)
            xs, ys = mollweide(np.full_like(lats, lon0), lats)
            ax.plot(xs, ys, color="#E2E2E2", lw=0.3, zorder=0)


# --------------------------------------------------------------------------
# pure-python Natural Earth shapefile reader (polyline only)
# --------------------------------------------------------------------------
def _rdp(points, eps):
    """Ramer-Douglas-Peucker simplification of an (n, 2) polyline."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 3:
        return pts
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        p0, p1 = pts[i0], pts[i1]
        d = p1 - p0
        seg_len = np.hypot(d[0], d[1])
        if seg_len < 1e-12:
            dist = np.hypot(*(pts[i0 + 1:i1] - p0).T)
        else:
            dist = np.abs(d[0] * (p0[1] - pts[i0 + 1:i1, 1])
                          - (p0[0] - pts[i0 + 1:i1, 0]) * d[1]) / seg_len
        j = int(np.argmax(dist))
        if dist[j] > eps:
            k = i0 + 1 + j
            keep[k] = True
            stack.extend([(i0, k), (k, i1)])
    return pts[keep]


_COAST_CACHE = None


def load_coastline(tol=0.12):
    """Parse the Natural Earth 110m coastline .shp; return list of (lon,lat)
    arrays, split at the dateline and simplified with RDP."""
    global _COAST_CACHE
    if _COAST_CACHE is not None:
        return _COAST_CACHE
    segments = []
    if not os.path.exists(COAST_SHP):
        _COAST_CACHE = segments
        return segments
    with open(COAST_SHP, "rb") as fh:
        fh.read(100)  # file header
        while True:
            head = fh.read(8)
            if len(head) < 8:
                break
            _, clen = struct.unpack(">ii", head)
            content = fh.read(clen * 2)
            if len(content) < 4:
                break
            shptype = struct.unpack("<i", content[:4])[0]
            if shptype == 0:
                continue
            if shptype not in (3, 13, 23):  # polyline / z / m
                continue
            if shptype != 3:
                # not expected in the 110m coastline; skip
                continue
            nparts, npoints = struct.unpack("<ii", content[36:44])
            parts = struct.unpack("<%di" % nparts, content[44:44 + 4 * nparts])
            off = 44 + 4 * nparts
            pts = np.frombuffer(content, dtype="<f8", count=2 * npoints,
                                offset=off).reshape(npoints, 2)
            bounds = list(parts) + [npoints]
            for k in range(nparts):
                seg = pts[bounds[k]:bounds[k + 1]]
                if len(seg) < 2:
                    continue
                # split where the polyline crosses the dateline
                jump = np.abs(np.diff(seg[:, 0])) > 180.0
                if jump.any():
                    idx = np.flatnonzero(jump) + 1
                    for piece in np.split(seg, idx):
                        if len(piece) >= 2:
                            segments.append(_rdp(piece, tol))
                else:
                    segments.append(_rdp(seg, tol))
    _COAST_CACHE = segments
    return segments


def draw_coastline(ax, color="#4D4D4D", lw=0.35):
    for seg in load_coastline():
        xs, ys = mollweide(seg[:, 0], seg[:, 1])
        ax.plot(xs, ys, color=color, lw=lw, solid_joinstyle="round",
                zorder=6)


# --------------------------------------------------------------------------
# grid helpers
# --------------------------------------------------------------------------
def block_sum_2d(arr, factor):
    """Sum a (nlat, nlon) array over factor x factor blocks."""
    nlat, nlon = arr.shape
    if nlat % factor or nlon % factor:
        raise ValueError("shape not divisible by factor")
    return arr.reshape(nlat // factor, factor, nlon // factor,
                       factor).sum(axis=(1, 3))


def block_mean_2d(arr, factor):
    nlat, nlon = arr.shape
    if nlat % factor or nlon % factor:
        raise ValueError("shape not divisible by factor")
    return arr.reshape(nlat // factor, factor, nlon // factor,
                       factor).mean(axis=(1, 3))


def block_coords(lat, lon, factor):
    """Coordinates of block centres after block_sum_2d / block_mean_2d."""
    lat_c = lat.reshape(-1, factor).mean(axis=1)
    lon_c = lon.reshape(-1, factor).mean(axis=1)
    return lat_c, lon_c


def grid_xy(lat_c, lon_c):
    """2-D projected coordinates for a lat/lon centre grid."""
    LON, LAT = np.meshgrid(lon_c, lat_c)
    X, Y = mollweide(LON, LAT)
    return X, Y


# --------------------------------------------------------------------------
# contour path simplification (keeps SVG vector files small)
# --------------------------------------------------------------------------
def _simplify_ring(ring, eps):
    if ring.shape[0] < 4:
        return ring
    closed = np.allclose(ring[0], ring[-1])
    core = ring[:-1] if closed else ring
    core = _rdp(core, eps)
    if closed:
        core = np.vstack([core, core[0]])
    return core


def _compound_path_from_rings(rings, eps):
    """Simplify every ring of a compound contour path, keeping fill rule."""
    nv, nc = [], []
    for ring in rings:
        if len(ring) < 3:
            continue
        simp = _simplify_ring(ring, eps)
        if len(simp) < 3:
            continue
        nv.append(simp)
        nc.extend([Path.MOVETO] + [Path.LINETO] * (len(simp) - 2)
                  + [Path.CLOSEPOLY])
    if not nv:
        return None
    return Path(np.vstack(nv), np.array(nc, dtype=np.uint8))


def simplify_contourf(cs, ax, eps=0.012, zorder=2):
    """Replace a contourf ContourSet with RDP-simplified PathPatches.

    matplotlib >= 3.8 returns a single QuadContourSet artist whose
    ``get_paths()`` holds one compound path per colour band.  Each band is
    rebuilt as a compound path with simplified rings, preserving holes
    through the nonzero winding rule.
    """
    paths = cs.get_paths()
    faces = cs.get_facecolors()
    if len(faces) < len(paths):
        faces = np.resize(faces, (len(paths), 4))
    patches = []
    for path, face in zip(paths, faces):
        verts = path.vertices
        codes = path.codes
        if codes is None:
            continue
        rings = []
        start = 0
        for i in range(1, len(codes) + 1):
            if i == len(codes) or codes[i] == Path.MOVETO:
                seg = verts[start:i]
                if len(seg) >= 2:
                    rings.append(seg)
                start = i
        newpath = _compound_path_from_rings(rings, eps)
        if newpath is None:
            continue
        pat = PathPatch(newpath, facecolor=face, edgecolor="none",
                        linewidth=0, antialiased=True, zorder=zorder,
                        transform=ax.transData, clip_on=True)
        ax.add_patch(pat)
        patches.append(pat)
    cs.remove()
    return patches


def add_simplified_contourf(ax, X, Y, Z, levels, cmap=SEQ_CMAP,
                            eps=0.012, extend="max", zorder=2):
    """contourf -> simplified patches; returns (patches, levels, cmap)."""
    cs = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend=extend,
                     antialiased=False, zorder=zorder)
    patches = simplify_contourf(cs, ax, eps=eps, zorder=zorder)
    return patches, list(levels)


def spec_colorbar(cax, levels, cmap, extend="max", ticks=None, label=None):
    """Compact horizontal colourbar for a simplified contour map."""
    from matplotlib.colorbar import ColorbarBase
    norm = mcolors.BoundaryNorm(levels, cmap.N, extend=extend)
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal",
                      extend=extend, ticks=ticks)
    if label:
        cax.set_xlabel(label, fontsize=note_size(FS_ANNOT), labelpad=1.5)
    cax.tick_params(length=2, width=0.5, labelsize=note_size(FS_TICK),
                    pad=1.5)
    cb.outline.set_linewidth(0.4)
    return cb
