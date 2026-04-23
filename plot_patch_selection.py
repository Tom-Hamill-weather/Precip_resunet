"""
plot_patch_selection.py

Usage:
    python plot_patch_selection.py cyyyymmddhh clead

Example:
    python plot_patch_selection.py 2025021700 12   # Feb 17 2025 – widespread SE flooding

Illustrates the 96×96 patch sampling used in save_patched_GRAF_MRMS_GFS.py.
Reads raw GRAF precipitation and MRMS quality data, runs the micro-sampling
logic (weighted by smoothed precipitation, quality-filtered), and plots the
GRAF field on a CONUS basemap with each selected patch shown as a rectangle.
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

# ─── Patch selection (matches save_patched_GRAF_MRMS_GFS.py exactly) ─────────

def select_patches(precip_graf, quality_mrms, ny, nx, nsamps=35, seed=42):
    """
    Micro-sampling: weighted random selection of 96×96 patch centres.

    Candidates come from a dense strided grid (stride=24).  Each candidate
    is weighted by smoothed_precip² so wetter regions are preferred.
    Candidates where >10% of MRMS pixels are flagged bad are excluded.
    """
    np.random.seed(seed)
    smoothed = ndimage.gaussian_filter(precip_graf, sigma=30)

    stride = 24
    yy, xx = np.meshgrid(
        np.arange(ny // 8 + 65, ny * 4 // 5, stride),
        np.arange(nx // 10,      9 * nx // 10, stride),
        indexing='ij'
    )
    fy, fx = yy.ravel(), xx.ravel()

    vals = smoothed[fy, fx]
    pmax = float(np.max(vals ** 2)) or 1.0
    weights = 1e-4 + (1.0 - 1e-4) * (vals ** 2) / pmax

    bad_frac = ndimage.uniform_filter((quality_mrms <= 0.01).astype(float),
                                      size=96)
    weights[bad_frac[fy, fx] > 0.10] = 0.0

    wsum = weights.sum()
    if wsum == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    probs   = weights / wsum
    chosen  = np.random.choice(len(fy), size=min(nsamps, len(fy)),
                               replace=False, p=probs)
    return fy[chosen], fx[chosen]

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
    nsamps = 50 if domain_mean > 0.15 else (28 if domain_mean < 0.10 else 35)
    print(f"Domain mean = {domain_mean:.4f} mm  →  nsamps = {nsamps}")

    j_sel, i_sel = select_patches(precip_graf, qual, ny, nx, nsamps=nsamps)
    print(f"Selected {len(j_sel)} patches")

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
        f"GRAF 1-h precipitation and selected 96×96 training patches\n"
        f"Init: {cyyyymmddhh}   Valid: {valid}   Lead: +{clead} h   "
        f"N = {len(j_sel)} patches",
        fontsize=20
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

    # ── Patch rectangles ──────────────────────────────────────────────────────
    # Build polygon corners for each selected patch in map coordinates.
    # The GRAF grid is Lambert Conformal so patches are square in grid space
    # but may appear slightly non-rectangular on the lat/lon-based map; we use
    # the actual four corner lat/lons to be exact.
    polys = []
    r = PATCH_HALF
    for jy, ix in zip(j_sel, i_sel):
        j0 = max(jy - r, 0);    j1 = min(jy + r, ny - 1)
        i0 = max(ix - r, 0);    i1 = min(ix + r, nx - 1)
        clat = [lats[j0, i0], lats[j0, i1], lats[j1, i1], lats[j1, i0]]
        clon = [lons[j0, i0], lons[j0, i1], lons[j1, i1], lons[j1, i0]]
        cx, cy = m(clon, clat)
        polys.append(Polygon(list(zip(cx, cy)), closed=True))

    pc = PatchCollection(polys, facecolor='none',
                          edgecolor='black', linewidth=0.9, alpha=0.80,
                          zorder=5)
    ax.add_collection(pc)

    # Also mark patch centres for clarity
    if len(j_sel):
        cx_cen, cy_cen = m(lons[j_sel, i_sel], lats[j_sel, i_sel])
        ax.plot(cx_cen, cy_cen, 'k.', markersize=2.5, zorder=6, alpha=0.7)

    outfile = f"patch_selection_{cyyyymmddhh}_{clead}h.png"
    fig.savefig(outfile, dpi=200)
    print(f"Saved: {outfile}")
    plt.close()


if __name__ == "__main__":
    main()
