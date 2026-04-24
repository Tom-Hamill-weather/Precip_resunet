"""
plot_patch_selection.py

Usage:
    python plot_patch_selection.py cyyyymmddhh clead

Example:
    python plot_patch_selection.py 2025021700 12   # Feb 17 2025 – widespread SE flooding

Illustrates the non-overlapping 96×96 patch sampling strategy.
Uses a stride-96 tiled grid with a date-seeded random global shift to prevent
terrain over-learning.  Wet patches (blue) are sampled by precip^1.5 weight;
dry patches (red) are sampled uniformly.
"""

import os
import sys
import warnings
import numpy as np
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
import pygrib
from netCDF4 import Dataset
from configparser import ConfigParser
from dateutils import dateshift

warnings.filterwarnings("ignore")

PATCH_HALF = 48    # 96×96 patch → ±48 grid cells from centre

# ─── Environment detection ────────────────────────────────────────────────────

def detect_config():
    """Return (config_file, aws_base_path) for the current host."""
    if os.path.exists('/data/resnet_data'):
        return 'config_aws.ini', '/data/resnet_data'
    if os.path.exists('/data2/resnet_data'):
        return 'config_aws.ini', '/data2/resnet_data'
    if os.path.exists('/storage2/library/archive/grid'):
        return 'config_hdo.ini', None
    return 'config_laptop.ini', None

# ─── Patch selection: non-overlapping tiled grid with random global shift ─────

def select_patches_nonoverlapping(precip_graf, quality_mrms, ny, nx, cyyyymmddhh):
    """
    Non-overlapping 96×96 patch selection.

    Algorithm:
      1. Seed from the date string so each forecast day gets a unique shift.
      2. Draw (shift_y, shift_x) uniformly from [0, 96) — shifts the tile grid
         so terrain never falls at the same patch-local position across days.
      3. Tile the valid domain (matching the original y/x bounds) with stride=96.
      4. Exclude tiles where >10% of MRMS pixels are bad quality.
      5. Classify each remaining tile as wet (patch max GRAF >= 0.5 mm) or dry.
      6. Sample wet tiles weighted by mean_precip^1.5:
           n_wet = 35 if domain mean > 0.15 mm, 25 if >= 0.10, 10 otherwise.
      7. Sample 5-8 dry tiles uniformly at random.
         (No padding to a fixed total — on dry days, only dry tiles are taken.)
    Returns (j_sel, i_sel, is_wet) where is_wet[k] is True for wet patches.
    """
    seed = int(cyyyymmddhh) % (2**31)
    rng = np.random.default_rng(seed)

    shift_y = int(rng.integers(0, 96))
    shift_x = int(rng.integers(0, 96))

    # Valid center range (same domain bounds as original code)
    y_min = ny // 8 + 65
    y_max = ny * 4 // 5
    x_min = nx // 10
    x_max = 9 * nx // 10

    centers_y = np.arange(y_min + shift_y, y_max, 96)
    centers_x = np.arange(x_min + shift_x, x_max, 96)

    # Ensure full 96×96 patch fits within the array
    centers_y = centers_y[(centers_y - 48 >= 0) & (centers_y + 48 < ny)]
    centers_x = centers_x[(centers_x - 48 >= 0) & (centers_x + 48 < nx)]

    yy, xx = np.meshgrid(centers_y, centers_x, indexing='ij')
    fy, fx = yy.ravel(), xx.ravel()

    if len(fy) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=bool)

    # Per-patch statistics (value at [j,i] = statistic over 96×96 block)
    patch_mean = ndimage.uniform_filter(precip_graf.astype(float), size=96)
    patch_max  = ndimage.maximum_filter(precip_graf.astype(float), size=96)
    bad_frac   = ndimage.uniform_filter((quality_mrms <= 0.01).astype(float), size=96)

    pmean = patch_mean[fy, fx]
    pmax  = patch_max[fy, fx]
    bfrac = bad_frac[fy, fx]

    # Drop patches with too much bad MRMS data (training masks individual bad
    # pixels via ignore_index=-1, so a 50% threshold is reasonable)
    valid = bfrac <= 0.50
    fy, fx, pmean, pmax = fy[valid], fx[valid], pmean[valid], pmax[valid]

    if len(fy) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=bool)

    # Wet: any pixel in the patch reaches 0.5 mm; weights by patch mean for sampling
    wet_mask = pmax >= 0.5
    dry_mask = ~wet_mask

    fy_wet, fx_wet = fy[wet_mask], fx[wet_mask]
    fy_dry, fx_dry = fy[dry_mask], fx[dry_mask]
    pm_wet = pmean[wet_mask]

    domain_mean = float(precip_graf.mean())
    n_wet = 35 if domain_mean > 0.15 else (25 if domain_mean >= 0.10 else 10)
    n_dry = int(rng.integers(5, 9))  # 5–8 inclusive

    j_out, i_out, wet_flag = [], [], []

    if len(fy_wet) > 0:
        w = pm_wet ** 1.5
        w /= w.sum()
        n_take = min(n_wet, len(fy_wet))
        idx = rng.choice(len(fy_wet), size=n_take, replace=False, p=w)
        j_out.extend(fy_wet[idx]);  i_out.extend(fx_wet[idx])
        wet_flag.extend([True] * n_take)

    if len(fy_dry) > 0:
        n_take = min(n_dry, len(fy_dry))
        idx = rng.choice(len(fy_dry), size=n_take, replace=False)
        j_out.extend(fy_dry[idx]);  i_out.extend(fx_dry[idx])
        wet_flag.extend([False] * n_take)

    return (np.array(j_out, dtype=int),
            np.array(i_out, dtype=int),
            np.array(wet_flag, dtype=bool))

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_patch_selection.py cyyyymmddhh clead")
        sys.exit(1)

    cyyyymmddhh = sys.argv[1]
    clead       = sys.argv[2]
    config_file, aws_base = detect_config()

    config = ConfigParser()
    config.read(config_file)
    dirs = config["DIRECTORIES"]

    def d(key):
        """Look up a config directory key, remapping /data/resnet_data if on AWS."""
        val = dirs[key]
        if aws_base:
            val = val.replace('/data/resnet_data', aws_base)
        return val

    # ── Build GRAF file path ──────────────────────────────────────────────────
    il          = int(clead)
    cyyyymmdd   = cyyyymmddhh[:8]
    chh         = cyyyymmddhh[8:10]
    valid       = dateshift(cyyyymmddhh, il)
    yyyymmdd_v  = valid[:8]
    hh_v        = valid[8:10]

    graf_trans = int(config["PARAMETERS"].get("GRAF_transition_date", "2024040512"))
    if int(cyyyymmddhh) > graf_trans:
        graf_dir = d("grafdatadir_conus_new")
        prefix   = 'grid.hdo-graf_conus.'
    else:
        graf_dir = d("grafdatadir_conus_old")
        prefix   = 'grid.hdo-graflr_conus.'

    graf_path = os.path.join(
        graf_dir, cyyyymmdd, chh,
        f"{prefix}{yyyymmdd_v}T{hh_v}0000Z."
        f"{cyyyymmdd}T{chh}0000Z.PT{clead}H.CONUS@4km.APCP.SFC.grb2"
    )
    print(f"GRAF : {graf_path}")
    if not os.path.exists(graf_path):
        sys.exit(f"ERROR: GRAF file not found:\n  {graf_path}")

    with pygrib.open(graf_path) as gf:
        grb          = gf.select(endStep=il)[0]
        lats, lons   = grb.latlons()
        precip_graf  = np.clip(grb.values, 0.0, 75.0)

    ny, nx = lats.shape
    print(f"Grid: {ny} × {nx},  max GRAF precip = {precip_graf.max():.2f} mm")

    # ── Read MRMS quality (for patch exclusion only) ──────────────────────────
    mrms_path = os.path.join(
        d("mrms_data_directory"), valid[:6],
        f"MRMS_1h_pamt_and_data_qual_{valid}.nc"
    )
    print(f"MRMS: {mrms_path}")
    if os.path.exists(mrms_path):
        with Dataset(mrms_path, 'r') as nc:
            qual = np.array(nc.variables['data_quality'][:, :])
            qual = np.where(qual > 1.0, -1.0, qual)
    else:
        print("  MRMS file not found — assuming all pixels good quality.")
        qual = np.ones((ny, nx), dtype=np.float32)

    # ── Run patch selection ───────────────────────────────────────────────────
    domain_mean = float(precip_graf.mean())
    print(f"Domain mean = {domain_mean:.4f} mm")

    j_sel, i_sel, is_wet = select_patches_nonoverlapping(
        precip_graf, qual, ny, nx, cyyyymmddhh)
    n_wet = int(is_wet.sum())
    n_dry = int((~is_wet).sum())
    print(f"Selected {len(j_sel)} patches ({n_wet} wet, {n_dry} dry)")

    # ── Basemap ───────────────────────────────────────────────────────────────
    # Use explicit CONUS-focused bounds so the data fills the figure; the full
    # GRAF domain extends well into Canada and Mexico, leaving wide triangular
    # whitespace at the corners when the grid corners are used directly.
    m = Basemap(
        rsphere=(6378137.00, 6356752.3142),
        resolution='l', area_thresh=1000., projection='lcc',
        lat_1=35., lat_2=45., lat_0=40., lon_0=-100.,
        llcrnrlon=-126.0, llcrnrlat=22.0,
        urcrnrlon=-64.5,  urcrnrlat=50.5,
    )
    xg, yg = m(lons, lats)

    # ── Colourmap (matches plot_GRAF_MRMS.py) ─────────────────────────────────
    colorst = ['White', '#E4FFFF', '#C4E8FF', '#8FB3FF', '#D8F9D8',
               '#A6ECA6', '#42F742', 'Yellow', 'Gold', 'Orange',
               '#FCD5D9', '#F6A3AE', '#f17484']
    clevs = np.array([0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 25, 50])
    cmap  = mpl.colors.LinearSegmentedColormap.from_list("", colorst,
                                                          N=len(colorst))
    norm  = mcolors.BoundaryNorm(clevs, len(colorst), clip=True)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 10))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.13, top=0.93)

    ax.set_title(
        f"GRAF 1-h precipitation — non-overlapping 96×96 training patches\n"
        f"Init: {cyyyymmddhh}   Valid: {valid}   Lead: +{clead} h   "
        f"N = {len(j_sel)} patches  ({n_wet} wet, {n_dry} dry)",
        fontsize=18
    )

    CS = m.pcolormesh(xg, yg, precip_graf, cmap=cmap, norm=norm,
                      shading='nearest', ax=ax)
    m.drawcoastlines(linewidth=0.8,  color='Gray',  ax=ax)
    m.drawcountries(linewidth=0.6,   color='Gray',  ax=ax)
    m.drawstates(linewidth=0.3,      color='Gray',  ax=ax)

    # Colorbar
    cax = fig.add_axes([0.08, 0.05, 0.84, 0.026])
    cb  = plt.colorbar(CS, orientation='horizontal', cax=cax,
                        drawedges=True, ticks=clevs, format='%g', extend='max')
    cb.ax.tick_params(labelsize=13)
    cb.set_label('1-h accumulated precipitation (mm)', fontsize=14)

    # ── Patch rectangles (blue = wet, red = dry) ──────────────────────────────
    r = PATCH_HALF
    for color_flag, edge_color in [(True, '#0044CC'), (False, '#CC2200')]:
        polys = []
        mask = is_wet if color_flag else ~is_wet
        for jy, ix in zip(j_sel[mask], i_sel[mask]):
            j0 = max(jy - r, 0);    j1 = min(jy + r, ny - 1)
            i0 = max(ix - r, 0);    i1 = min(ix + r, nx - 1)
            clat = [lats[j0, i0], lats[j0, i1], lats[j1, i1], lats[j1, i0]]
            clon = [lons[j0, i0], lons[j0, i1], lons[j1, i1], lons[j1, i0]]
            cx, cy = m(clon, clat)
            polys.append(Polygon(list(zip(cx, cy)), closed=True))
        if polys:
            pc = PatchCollection(polys, facecolor='none',
                                 edgecolor=edge_color, linewidth=1.0,
                                 alpha=0.85, zorder=5)
            ax.add_collection(pc)

    # Mark patch centres
    if len(j_sel):
        cx_cen, cy_cen = m(lons[j_sel, i_sel], lats[j_sel, i_sel])
        ax.plot(cx_cen, cy_cen, 'k.', markersize=2.5, zorder=6, alpha=0.7)

    # Simple legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='#0044CC', lw=1.5, label='Wet patch (mean≥0.1 mm)'),
        Line2D([0], [0], color='#CC2200', lw=1.5, label='Dry patch (mean<0.1 mm)'),
    ]
    ax.legend(handles=legend_handles, loc='lower left', fontsize=11,
              framealpha=0.7, edgecolor='gray')

    outfile = f"patch_selection_{cyyyymmddhh}_{clead}h.png"
    fig.savefig(outfile, dpi=200)
    print(f"Saved: {outfile}")
    plt.close()


if __name__ == "__main__":
    main()
