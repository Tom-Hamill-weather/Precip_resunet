"""
plot_patch_selection_256_example.py

Usage:
    python plot_patch_selection_256_example.py cyyyymmddhh clead

Example:
    python plot_patch_selection_256_example.py 2025021700 12   # SE flooding case

Illustrative version of plot_patch_selection.py: draws sample wet (blue) and
dry (red) patch box outlines over raw GRAF 1-h precipitation, using an
oversized 256x256 patch for visual clarity (NOT the actual 96x96 training
patch size). No title or legend is drawn.
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
from configparser import ConfigParser
from dateutils import dateshift

warnings.filterwarnings("ignore")

PATCH_HALF = 128    # 256x256 illustrative patch -> +/-128 grid cells from centre
TILE_SIZE  = 2 * PATCH_HALF

# --- Environment detection --------------------------------------------------

def detect_config():
    """Return (config_file, aws_base_path) for the current host."""
    if os.path.exists('/data/resnet_data'):
        return 'config_aws.ini', '/data/resnet_data'
    if os.path.exists('/data2/resnet_data'):
        return 'config_aws.ini', '/data2/resnet_data'
    if os.path.exists('/storage2/library/archive/grid'):
        return 'config_hdo.ini', None
    return 'config_laptop.ini', None

# --- Patch selection: non-overlapping tiled grid with random global shift --

def select_patches_nonoverlapping(precip_graf, ny, nx, cyyyymmddhh, n_total=None):
    """
    Non-overlapping TILE_SIZE x TILE_SIZE patch selection (illustrative sizing).
    Same logic as plot_patch_selection.py, but scaled to the larger patch size
    above so the example is easy to see on a CONUS map. This illustrative
    version does not require MRMS coverage/quality, so tiles outside the
    MRMS domain (e.g. over Canada/Mexico) are eligible too.
    """
    seed = int(cyyyymmddhh) % (2**31)
    rng = np.random.default_rng(seed)

    shift_y = int(rng.integers(0, TILE_SIZE))
    shift_x = int(rng.integers(0, TILE_SIZE))

    y_min = ny // 8 + 65
    y_max = ny * 4 // 5
    x_min = nx // 10
    x_max = 9 * nx // 10

    centers_y = np.arange(y_min + shift_y, y_max, TILE_SIZE)
    centers_x = np.arange(x_min + shift_x, x_max, TILE_SIZE)

    centers_y = centers_y[(centers_y - PATCH_HALF >= 0) & (centers_y + PATCH_HALF < ny)]
    centers_x = centers_x[(centers_x - PATCH_HALF >= 0) & (centers_x + PATCH_HALF < nx)]

    yy, xx = np.meshgrid(centers_y, centers_x, indexing='ij')
    fy, fx = yy.ravel(), xx.ravel()

    if len(fy) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=bool)

    patch_mean = ndimage.uniform_filter(precip_graf.astype(float), size=TILE_SIZE)
    patch_max  = ndimage.maximum_filter(precip_graf.astype(float), size=TILE_SIZE)

    pmean = patch_mean[fy, fx]
    pmax  = patch_max[fy, fx]

    # With a 256x256 illustrative tile (vs. the real 96x96 training patch),
    # "any pixel >= 0.5 mm" is true almost everywhere near a widespread
    # system, so wet/dry is defined on patch-mean / patch-max instead to
    # keep a clear visual contrast: wet = substantial mean rain in the
    # tile, dry = essentially no rain anywhere in the tile.
    wet_mask = pmean >= 0.5
    dry_mask = pmax < 0.5

    fy_wet, fx_wet = fy[wet_mask], fx[wet_mask]
    fy_dry, fx_dry = fy[dry_mask], fx[dry_mask]
    pm_wet = pmean[wet_mask]

    if n_total is None:
        domain_mean = float(precip_graf.mean())
        n_wet = 8 if domain_mean > 0.15 else (6 if domain_mean >= 0.10 else 3)
        n_dry = int(rng.integers(4, 7))
    else:
        n_dry = min(len(fy_dry), max(1, n_total // 3))
        n_wet = min(len(fy_wet), n_total - n_dry)
        n_dry = min(len(fy_dry), n_total - n_wet)

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

# --- Main --------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_patch_selection_256_example.py cyyyymmddhh clead [n_total_patches]")
        sys.exit(1)

    cyyyymmddhh = sys.argv[1]
    clead       = sys.argv[2]
    n_total     = int(sys.argv[3]) if len(sys.argv) > 3 else None
    config_file, aws_base = detect_config()

    config = ConfigParser()
    config.read(config_file)
    dirs = config["DIRECTORIES"]

    def d(key):
        val = dirs[key]
        if aws_base:
            val = val.replace('/data/resnet_data', aws_base)
        return val

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
    print(f"Grid: {ny} x {nx},  max GRAF precip = {precip_graf.max():.2f} mm")

    domain_mean = float(precip_graf.mean())
    print(f"Domain mean = {domain_mean:.4f} mm")

    j_sel, i_sel, is_wet = select_patches_nonoverlapping(
        precip_graf, ny, nx, cyyyymmddhh, n_total=n_total)
    n_wet = int(is_wet.sum())
    n_dry = int((~is_wet).sum())
    print(f"Selected {len(j_sel)} patches ({n_wet} wet, {n_dry} dry)")

    # Use GRAF's own native LCC projection (spherical earth, lat_1=50,
    # lat_2=20, lat_0=50, lon_0=255 == -105) rather than a guessed one --
    # only with the exact native projection is the regular row/column grid
    # (and therefore each patch box) an axis-aligned rectangle in map space.
    m = Basemap(
        rsphere=6371229.0,
        resolution='l', area_thresh=1000., projection='lcc',
        lat_1=50., lat_2=20., lat_0=50., lon_0=255.0 - 360.0,
        llcrnrlon=lons.min(), llcrnrlat=lats.min(),
        urcrnrlon=lons.max(), urcrnrlat=lats.max(),
    )
    xg, yg = m(lons, lats)

    colorst = ['White', '#E4FFFF', '#C4E8FF', '#8FB3FF', '#D8F9D8',
               '#A6ECA6', '#42F742', 'Yellow', 'Gold', 'Orange',
               '#FCD5D9', '#F6A3AE', '#f17484']
    clevs = np.array([0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 25, 50])
    cmap  = mpl.colors.LinearSegmentedColormap.from_list("", colorst,
                                                          N=len(colorst))
    norm  = mcolors.BoundaryNorm(clevs, len(colorst), clip=True)

    fig, ax = plt.subplots(figsize=(13, 10))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

    CS = m.pcolormesh(xg, yg, precip_graf, cmap=cmap, norm=norm,
                      shading='nearest', ax=ax)
    m.drawcoastlines(linewidth=0.8,  color='Gray',  ax=ax)
    m.drawcountries(linewidth=0.6,   color='Gray',  ax=ax)
    m.drawstates(linewidth=0.3,      color='Gray',  ax=ax)

    # Draw box corners straight from the projected grid (xg, yg) rather than
    # re-projecting lat/lon -- since m now uses GRAF's exact native LCC
    # projection, this makes each patch a perfectly axis-aligned rectangle.
    r = PATCH_HALF
    for color_flag, edge_color in [(True, '#0044CC'), (False, '#CC2200')]:
        polys = []
        mask = is_wet if color_flag else ~is_wet
        for jy, ix in zip(j_sel[mask], i_sel[mask]):
            j0 = max(jy - r, 0);    j1 = min(jy + r, ny - 1)
            i0 = max(ix - r, 0);    i1 = min(ix + r, nx - 1)
            x0, x1 = xg[jy, i0], xg[jy, i1]
            y0, y1 = yg[j0, ix], yg[j1, ix]
            polys.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                                  closed=True))
        if polys:
            pc = PatchCollection(polys, facecolor='none',
                                 edgecolor=edge_color, linewidth=1.5,
                                 alpha=0.9, zorder=5)
            ax.add_collection(pc)

    # Crop tightly to the exact projected extent of the GRAF grid so no
    # whitespace/triangular corners remain and every patch box (always a
    # subset of the grid) is guaranteed to render fully within view.
    ax.set_xlim(xg.min(), xg.max())
    ax.set_ylim(yg.min(), yg.max())
    ax.set_aspect('equal')
    ax.set_axis_off()

    outfile = f"patch_selection_256_example_{cyyyymmddhh}_{clead}h.png"
    fig.savefig(outfile, dpi=200)
    print(f"Saved: {outfile}")
    plt.close()


if __name__ == "__main__":
    main()
