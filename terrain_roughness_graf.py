"""
python terrain_roughness_graf.py

Computes local standard deviation of raw GRAF terrain elevation on GRAF's
fixed CONUS-plus-surroundings grid (1308y x 1524x), for Tom's terrain-
roughness stratification of the BSS-vs-lead-time figure (July 2026 port of
the HRRRcal terrain-roughness analysis in ~/HRRRcal/terrain_roughness.py to
this repo's GRAF verification pipeline).

Local std uses the same Gaussian-moments trick HRRRcal used:

  local_std = sqrt(gaussian_filter(terrain**2, sigma) - gaussian_filter(terrain, sigma)**2)

sigma=14 grid points (Tom's answer, 2026-07-20): GRAF's grid spacing is
~4.3 km (measured directly from lats/lons, not the ~3.2-4.0 km originally
guessed), so 14 grid points gives the same ~60-km physical smoothing scale
as HRRRcal's sigma=20 grid points on HRRR's 3-km grid.

Unlike HRRRcal, there is no interior-border-clip / CONUS-only concept here:
GRAF's domain is much bigger than CONUS (Canada/Mexico/Caribbean included),
and Tom's answer (2026-07-20) was to use the naive candidate-point set (all
finite terrain points, no quality or bounding-box restriction) for the
90th-percentile threshold on this first pass -- matching HRRRcal's own
development history (naive mask first, climo-quality filter added later
only after the naive result was inspected).

Writes terrain_roughness_mask_graf.nc (local_std, top10_mask, bottom90_mask
-- same variable names as HRRRcal's terrain_roughness_mask.nc, so downstream
code needs no new names) plus a 2-panel QC PNG.

Usage:
  python terrain_roughness_graf.py
"""
import os
import numpy as np
import netCDF4
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.basemap import Basemap

# --- Auto-detect environment (AWS vs local) ---
def detect_environment():
    aws_paths = ['/data/resnet_data', '/data2/resnet_data']
    for path in aws_paths:
        if os.path.exists(path):
            print(f"Detected AWS environment (found {path})")
            return 'aws', path
    print("Detected local laptop environment")
    return 'laptop', None

ENVIRONMENT, AWS_BASE_PATH = detect_environment()

# --------------------------------------------------------------

def read_config_file(config_file, directory_object_name):
    from configparser import ConfigParser
    config_object = ConfigParser()
    config_object.read(config_file)
    directory = config_object[directory_object_name]

    if "GRAFdatadir_conus_laptop" in directory:
        terrain_file = directory.get("terrain_file", None)
    else:
        base_dir = directory.get("resnet_data_directory", AWS_BASE_PATH or "/data/resnet_data")
        terrain_file = directory.get("terrain_file", f"{base_dir}/terrain/GRAF_CONUS_terrain_info.nc")

    print(f"  Terrain file: {terrain_file}")
    return terrain_file

# --------------------------------------------------------------

SIGMA = 14.0
PCTL = 90.0

OUT_NC = 'terrain_roughness_mask_graf.nc'
OUT_PLOT = 'GRAF_terrain_roughness_QC.png'


def local_std(terrain, sigma):
    t = terrain.astype(np.float64)
    m1 = ndimage.gaussian_filter(t, sigma)
    m2 = ndimage.gaussian_filter(t * t, sigma)
    return np.sqrt(np.clip(m2 - m1 * m1, 0.0, None))


def _basemap(ax, lon, lat):
    m = Basemap(rsphere=(6378137.00, 6356752.3142),
                resolution='l', area_thresh=1000., projection='lcc',
                lat_1=35., lat_2=45, lat_0=40., lon_0=-97.5,
                llcrnrlon=-125.0, llcrnrlat=24.0,
                urcrnrlon=-66.0, urcrnrlat=50.0, ax=ax)
    m.drawcoastlines(linewidth=0.6, color='Gray')
    m.drawcountries(linewidth=0.5, color='Gray')
    m.drawstates(linewidth=0.4, color='Gray')
    return m


def _panel_field(fig, ax, lon, lat, field, cmap, norm, title):
    m = _basemap(ax, lon, lat)
    x, y = m(lon, lat)
    mesh = ax.pcolormesh(x, y, field, cmap=cmap, norm=norm, shading='nearest')
    ax.set_title(title, fontsize=11)
    fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, fraction=0.045)


def main():
    if ENVIRONMENT == 'aws':
        config_file_name = 'config_aws.ini'
    else:
        config_file_name = 'config_laptop.ini'
    terrain_file = read_config_file(config_file_name, 'DIRECTORIES')

    nc = netCDF4.Dataset(terrain_file, 'r')
    terrain = np.asarray(nc.variables['terrain_height'][:], np.float64)
    lat = np.asarray(nc.variables['lats'][:], np.float64)
    lon = np.asarray(nc.variables['lons'][:], np.float64)
    nc.close()
    ny, nx = terrain.shape

    rough = local_std(terrain, SIGMA)

    # Naive candidate set: all finite points, no border/quality/bbox restriction
    # (Tom's decision, 2026-07-20 -- GRAF's domain is far bigger than CONUS,
    # so an interior-border-clip doesn't apply here the way it did for HRRR).
    candidates = np.isfinite(rough) & np.isfinite(terrain)

    thr = float(np.percentile(rough[candidates], PCTL))
    top10_mask = candidates & (rough >= thr)
    bottom90_mask = candidates & (rough < thr)

    with netCDF4.Dataset(OUT_NC, 'w') as ds:
        ds.createDimension('y', ny)
        ds.createDimension('x', nx)
        kw = dict(zlib=True, complevel=4)
        ds.createVariable('local_std', 'f4', ('y', 'x'), **kw)[:] = rough.astype(np.float32)
        ds.createVariable('top10_mask', 'i1', ('y', 'x'), **kw)[:] = top10_mask.astype(np.int8)
        ds.createVariable('bottom90_mask', 'i1', ('y', 'x'), **kw)[:] = bottom90_mask.astype(np.int8)
        ds.createVariable('lats', 'f4', ('y', 'x'), **kw)[:] = lat.astype(np.float32)
        ds.createVariable('lons', 'f4', ('y', 'x'), **kw)[:] = lon.astype(np.float32)
        ds.sigma_gridpoints = SIGMA
        ds.percentile = PCTL
        ds.threshold_m = thr
        ds.candidate_filter = 'naive: all finite terrain points, no border/quality/bbox restriction'
        ds.n_top10 = int(top10_mask.sum())
        ds.n_bottom90 = int(bottom90_mask.sum())
        ds.description = ('Local std of raw GRAF terrain elevation (gaussian_filter '
                           'moments trick, sigma=14 grid pts, ~60-km physical scale on '
                           "GRAF's ~4.3-km grid) used to stratify the GRAF BSS-vs-lead-"
                           'time verification into top-10%-roughest vs bottom-90% '
                           'terrain, ported from HRRRcal terrain_roughness.py.')

    print(f'[write] {OUT_NC}  sigma={SIGMA}gp  thr={thr:.1f} m  '
          f'n_top10={int(top10_mask.sum())}  n_bottom90={int(bottom90_mask.sum())}')

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    _panel_field(fig, axes[0], lon, lat, rough, 'terrain',
                 mcolors.Normalize(0.0, float(np.nanpercentile(rough[candidates], 99))),
                 f'Local std of terrain elevation (m), Gaussian sigma={SIGMA:.0f} grid pts')
    _panel_field(fig, axes[1], lon, lat, top10_mask.astype(np.float32),
                 'Reds', mcolors.Normalize(0, 1),
                 f'Top 10% roughest terrain  (threshold={thr:.0f} m)')
    fig.suptitle('GRAF terrain-roughness stratification for BSS-vs-lead-time analysis',
                 fontsize=14)
    plt.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[write] {OUT_PLOT}')


if __name__ == '__main__':
    main()
