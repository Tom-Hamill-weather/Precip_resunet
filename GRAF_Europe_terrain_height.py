"""
python GRAF_Europe_terrain_height.py

Generate GRAF_Europe_terrain_info.nc from the GRAF reforecast MPAS static file.

Source data:
  /tmp/rpm4km.static.nc  (s3://twc-graf-reforecast/rpm4km.static.nc)
  Variables: latCell, lonCell (radians), ter (meters) on MPAS unstructured mesh

Target grid:
  723 x 666 Lambert Conformal Conic, ~4 km, Europe domain
  Derived from a European GRAF GRIB2 file.

Interpolation method: Linear Delaunay triangulation (barycentric) in LCC
projected coordinates (metres).  This is the method used by MPAS's own
convert_mpas utility and ECMWF's MIR package for remapping real-valued
fields from an unstructured Voronoi mesh to a regular grid.  Operating in
projected metre-space rather than degree-space avoids the distortion
introduced by the non-uniform aspect ratio of geographic coordinates at
European latitudes (~40–62 N).

References:
  - MPAS convert_mpas: barycentric interpolation for all real-valued fields
    https://github.com/mgduda/convert_mpas
  - ECMWF MIR: finite-element linear on triangular (unstructured) source meshes
    https://www.ecmwf.int/en/newsletter/152/computing/new-ecmwf-interpolation-package-mir
  - CDO forum: bilinear preferred over conservative for orography
    https://code.mpimet.mpg.de/boards/1/topics/274?r=288
  - Taylor 2024 GMD: conservative remapping mesh-imprinting artefacts
    https://gmd.copernicus.org/articles/17/415/2024/

Tom Hamill, Apr 2026
"""

import numpy as np
import os, sys
import pygrib
import scipy.ndimage as ndimage
from scipy.interpolate import LinearNDInterpolator
from netCDF4 import Dataset
import pyproj
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------

STATIC_FILE = '/tmp/rpm4km.static.nc'
EURO_GRIB_EXAMPLE = (
    '/data/resnet_data/GRAF/hdo-graf_europe/20260215/00/'
    'grid.hdo-graf_europe.20260215T010000Z.20260215T000000Z'
    '.PT1H.EUROPE@4km.APCP.SFC.grb2'
)
OUTFILE = '/data/resnet_data/terrain/GRAF_Europe_terrain_info.nc'

# Gaussian smoothing sigma (grid points) — same as CONUS script
SIGMA_PRIMARY = 15.0
SIGMAS = [15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0]

# Bounding-box padding around Europe target domain (degrees)
# Wider margins avoid edge artefacts in LinearNDInterpolator
LAT_PAD = 4.0
LON_PAD = 6.0

# ---------------------------------------------------------------

def read_mpas_terrain(static_file):
    """
    Read terrain height and cell-centre coordinates from the MPAS static file.
    latCell / lonCell are stored in radians; lonCell is 0-360 east.
    Returns arrays in degrees with lonCell in -180..180 convention.
    """
    print(f'Reading MPAS static file: {static_file}')
    nc = Dataset(static_file, 'r')
    lat_rad = nc.variables['latCell'][:]
    lon_rad = nc.variables['lonCell'][:]
    ter = nc.variables['ter'][:]
    nc.close()

    lat_deg = np.degrees(lat_rad)
    lon_deg = np.degrees(lon_rad)
    lon_deg = np.where(lon_deg > 180.0, lon_deg - 360.0, lon_deg)

    print(f'  Total MPAS cells: {len(ter):,}')
    print(f'  ter range: {ter.min():.1f} to {ter.max():.1f} m')
    return lat_deg, lon_deg, ter


def read_euro_target_grid(grib_file):
    """
    Read the European GRAF target grid lat/lon from a GRIB2 file.
    Returns 2-D arrays (ny, nx) and projection parameters.
    """
    print(f'Reading European target grid from: {grib_file}')
    if not os.path.exists(grib_file):
        print(f'ERROR: GRIB file not found: {grib_file}')
        sys.exit(1)

    f = pygrib.open(grib_file)
    grb = f.read(1)[0]
    lats, lons = grb.latlons()
    proj = grb.projparams
    f.close()

    ny, nx = lats.shape
    print(f'  Grid shape: {ny} x {nx}')
    print(f'  Lat range: {lats.min():.3f} to {lats.max():.3f}')
    print(f'  Lon range: {lons.min():.3f} to {lons.max():.3f}')
    print(f'  Projection: {proj}')
    return lats, lons, proj


def filter_mpas_to_bbox(lat_deg, lon_deg, ter,
                        lat_min, lat_max, lon_min, lon_max):
    """
    Return MPAS points within an extended bounding box.
    Using a padded box ensures LinearNDInterpolator has source triangles
    covering all target-grid edge/corner cells.
    """
    mask = ((lat_deg >= lat_min) & (lat_deg <= lat_max) &
            (lon_deg >= lon_min) & (lon_deg <= lon_max))
    print(f'  MPAS points inside bounding box: {mask.sum():,} / {len(ter):,}')
    return lat_deg[mask], lon_deg[mask], ter[mask]


def build_lcc_projector(proj_params):
    """
    Build a pyproj Transformer: geographic (lon, lat) -> LCC (x, y in metres).
    Uses the sphere radius from the MPAS/GRAF projection parameters.
    """
    a = proj_params.get('a', 6371229)
    b = proj_params.get('b', a)
    lon_0 = proj_params['lon_0']
    lat_0 = proj_params['lat_0']
    lat_1 = proj_params['lat_1']
    lat_2 = proj_params['lat_2']

    lcc_crs = pyproj.CRS.from_dict({
        'proj': 'lcc',
        'a': a, 'b': b,
        'lon_0': lon_0, 'lat_0': lat_0,
        'lat_1': lat_1, 'lat_2': lat_2,
        'x_0': 0, 'y_0': 0,
        'units': 'm',
    })
    geo_crs = pyproj.CRS.from_epsg(4326)   # WGS84 lon/lat
    return pyproj.Transformer.from_crs(geo_crs, lcc_crs, always_xy=True)


def interpolate_terrain(src_lon, src_lat, src_ter,
                        tgt_lons, tgt_lats, transformer):
    """
    Interpolate terrain from unstructured MPAS points to the target grid using
    linear Delaunay triangulation (barycentric) in LCC projected coordinates.

    Working in projected metre-space (rather than geographic degree-space)
    avoids the aspect-ratio distortion at ~40-62 N and ensures the
    triangulation reflects true physical distances — matching the approach
    used by MPAS convert_mpas and ECMWF MIR for unstructured-to-structured
    remapping of real-valued fields.
    """
    print('Projecting source MPAS points to LCC coordinates...')
    src_x, src_y = transformer.transform(src_lon, src_lat)

    print('Projecting target grid to LCC coordinates...')
    ny, nx = tgt_lons.shape
    tgt_x, tgt_y = transformer.transform(tgt_lons.ravel(), tgt_lats.ravel())

    print(f'Building Delaunay triangulation on {len(src_x):,} source points...')
    interp = LinearNDInterpolator(
        np.column_stack([src_x, src_y]),
        src_ter,
        fill_value=np.nan,
    )

    print(f'Interpolating to {ny * nx:,} target points...')
    ter_flat = interp(tgt_x, tgt_y)
    ter_grid = ter_flat.reshape(ny, nx)

    n_nan = np.isnan(ter_grid).sum()
    if n_nan > 0:
        print(f'  WARNING: {n_nan} target points fell outside source triangulation.')
        print('  Filling with nearest-source fallback...')
        from scipy.spatial import cKDTree
        src_pts = np.column_stack([src_x, src_y])
        tgt_pts = np.column_stack([tgt_x, tgt_y])
        tree = cKDTree(src_pts)
        _, idx = tree.query(tgt_pts[np.isnan(ter_flat)])
        ter_flat[np.isnan(ter_flat)] = src_ter[idx]
        ter_grid = ter_flat.reshape(ny, nx)

    print(f'  Interpolated terrain range: {np.nanmin(ter_grid):.1f} '
          f'to {np.nanmax(ter_grid):.1f} m')
    return ter_grid


def terrain_slopes(data, lons, lats, ny, nx):
    """
    Compute dterrain/dlon and dterrain/dlat (m/m) using centred differences
    in physical (metre) distance.  Fully vectorised over the (ny, nx) grid.
    """
    mpdlat = 111000.0  # metres per degree latitude
    coslat = np.cos(lats * np.pi / 180.0)   # (ny, nx)
    mpdlon = mpdlat * coslat                # (ny, nx)

    dterrain_dlon = np.zeros((ny, nx), dtype=float)
    dterrain_dlat = np.zeros((ny, nx), dtype=float)

    # ---- interior (centred differences) ----
    dy_int = (lats[2:, 1:-1] - lats[:-2, 1:-1]) * mpdlat / 2.0
    dx_int = (lons[1:-1, 2:] - lons[1:-1, :-2]) * mpdlon[1:-1, 1:-1] / 2.0
    dterrain_dlon[1:-1, 1:-1] = (data[1:-1, 2:] - data[1:-1, :-2]) / (2.0 * dx_int)
    dterrain_dlat[1:-1, 1:-1] = (data[2:, 1:-1] - data[:-2, 1:-1]) / (2.0 * dy_int)

    # ---- west boundary (forward difference in x) ----
    dy_w = (lats[2:, 0] - lats[:-2, 0]) * mpdlat / 2.0
    dx_w = (lons[1:-1, 1] - lons[1:-1, 0]) * mpdlon[1:-1, 0]
    dterrain_dlon[1:-1, 0] = (data[1:-1, 1] - data[1:-1, 0]) / dx_w
    dterrain_dlat[1:-1, 0] = (data[2:, 0]   - data[:-2, 0])  / (2.0 * dy_w)

    # ---- east boundary (backward difference in x) ----
    dy_e = (lats[2:, -1] - lats[:-2, -1]) * mpdlat / 2.0
    dx_e = (lons[1:-1, -1] - lons[1:-1, -2]) * mpdlon[1:-1, -1]
    dterrain_dlon[1:-1, -1] = (data[1:-1, -1] - data[1:-1, -2]) / dx_e
    dterrain_dlat[1:-1, -1] = (data[2:, -1]   - data[:-2, -1]) / (2.0 * dy_e)

    # ---- south boundary (forward difference in y) ----
    dy_s = (lats[1, 1:-1] - lats[0, 1:-1]) * mpdlat
    dx_s = (lons[0, 2:] - lons[0, :-2]) * mpdlon[0, 1:-1] / 2.0
    dterrain_dlon[0, 1:-1] = (data[0, 2:] - data[0, :-2]) / (2.0 * dx_s)
    dterrain_dlat[0, 1:-1] = (data[1, 1:-1] - data[0, 1:-1]) / dy_s

    # ---- north boundary (backward difference in y) ----
    dy_n = (lats[-1, 1:-1] - lats[-2, 1:-1]) * mpdlat
    dx_n = (lons[-1, 2:] - lons[-1, :-2]) * mpdlon[-1, 1:-1] / 2.0
    dterrain_dlon[-1, 1:-1] = (data[-1, 2:] - data[-1, :-2]) / (2.0 * dx_n)
    dterrain_dlat[-1, 1:-1] = (data[-1, 1:-1] - data[-2, 1:-1]) / dy_n

    # ---- corners (average of three adjacent edge values) ----
    dterrain_dlon[0, 0]   = (dterrain_dlon[0, 1]   + dterrain_dlon[1, 0]   + dterrain_dlon[1, 1])   / 3.0
    dterrain_dlat[0, 0]   = (dterrain_dlat[0, 1]   + dterrain_dlat[1, 0]   + dterrain_dlat[1, 1])   / 3.0
    dterrain_dlon[-1, 0]  = (dterrain_dlon[-1, 1]  + dterrain_dlon[-2, 0]  + dterrain_dlon[-2, 1])  / 3.0
    dterrain_dlat[-1, 0]  = (dterrain_dlat[-1, 1]  + dterrain_dlat[-2, 0]  + dterrain_dlat[-2, 1])  / 3.0
    dterrain_dlon[-1, -1] = (dterrain_dlon[-2, -2] + dterrain_dlon[-2, -1] + dterrain_dlon[-1, -2]) / 3.0
    dterrain_dlat[-1, -1] = (dterrain_dlat[-2, -2] + dterrain_dlat[-2, -1] + dterrain_dlat[-1, -2]) / 3.0
    dterrain_dlon[0, -1]  = (dterrain_dlon[1, -2]  + dterrain_dlon[1, -1]  + dterrain_dlon[0, -2])  / 3.0
    dterrain_dlat[0, -1]  = (dterrain_dlat[1, -2]  + dterrain_dlat[1, -1]  + dterrain_dlat[0, -2])  / 3.0

    return dterrain_dlon, dterrain_dlat


def write_to_netcdf(outfile, ny, nx, nsigma, lons, lats,
                    sigmas, terrain_height, terrain_height_smoothed,
                    terrain_height_local_difference,
                    dterrain_dlon, dterrain_dlat,
                    dterrain_dlon_smoothed, dterrain_dlat_smoothed,
                    dterrain_dlon_difference, dterrain_dlat_difference,
                    terrain_height_smoothed_multisigma):
    """Write terrain fields to netCDF — same structure as GRAF_CONUS_terrain_info.nc."""
    print(f'Writing to {outfile}')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    nc = Dataset(outfile, 'w', format='NETCDF4_CLASSIC')
    nc.createDimension('ny', ny)
    nc.createDimension('nx', nx)
    nc.createDimension('nsigma', nsigma)

    def mkvar(name, dims, zlib=True, lsd=None, **attrs):
        kw = dict(zlib=zlib)
        if lsd is not None:
            kw['least_significant_digit'] = lsd
        v = nc.createVariable(name, 'f4', dims, **kw)
        for k, val in attrs.items():
            setattr(v, k, val)
        return v

    mkvar('lons', ('ny', 'nx'), long_name='longitude', units='degrees_east')[:] = lons
    mkvar('lats', ('ny', 'nx'), long_name='latitude',  units='degrees_north')[:] = lats
    mkvar('sigmas', ('nsigma',),
          long_name='smoothing length scale in GRAF grid pts',
          units='number of grid points')[:] = sigmas

    mkvar('terrain_height', ('ny', 'nx'), lsd=2,
          units='m',
          long_name='Terrain height for GRAF Europe (m)',
          valid_range=[-90., 13000.],
          missing_value=-99.99)[:] = terrain_height

    mkvar('terrain_height_smoothed', ('ny', 'nx'), lsd=2,
          units='m',
          long_name='Smoothed terrain height for GRAF Europe, '
                    '15 grid point sigma Gaussian convolve (m)',
          valid_range=[-90., 13000.],
          missing_value=-99.99)[:] = terrain_height_smoothed

    mkvar('terrain_height_local_difference', ('ny', 'nx'), lsd=2,
          units='m',
          long_name='Raw minus smoothed terrain height difference for '
                    'GRAF Europe, 15 grid point sigma Gaussian convolve',
          valid_range=[-3000., 3000.],
          missing_value=-9999.99)[:] = terrain_height_local_difference

    for name, data, ln in [
        ('dterrain_dlon',           dterrain_dlon,           'change in terrain height with longitude per metre horizontal'),
        ('dterrain_dlat',           dterrain_dlat,           'change in terrain height with latitude per metre horizontal'),
        ('dterrain_dlon_smoothed',  dterrain_dlon_smoothed,  'change in (smoothed) terrain height with longitude per metre horizontal'),
        ('dterrain_dlat_smoothed',  dterrain_dlat_smoothed,  'change in (smoothed) terrain height with latitude per metre horizontal'),
        ('dterrain_dlon_difference',dterrain_dlon_difference,'change in (raw-smoothed) terrain height with longitude per metre horizontal'),
        ('dterrain_dlat_difference',dterrain_dlat_difference,'change in (raw-smoothed) terrain height with latitude per metre horizontal'),
    ]:
        mkvar(name, ('ny', 'nx'), lsd=5, units='m/m', long_name=ln,
              valid_range=[-100., 100.], missing_value=-999.99)[:] = data

    mkvar('terrain_height_smoothed_multisigma', ('nsigma', 'ny', 'nx'), lsd=5,
          units='m',
          long_name='Terrain height smoothed at multiple sigmas',
          valid_range=[-90., 13000.],
          missing_value=-999.99)[:] = terrain_height_smoothed_multisigma

    nc.title = 'GRAF Europe terrain information: height, smoothed, deviations, gradients'
    nc.history = 'Created by GRAF_Europe_terrain_height.py; Apr 2026; Tom Hamill'
    nc.institution = 'The Weather Company'
    nc.platform = 'The Weather Company GRAF Europe'
    nc.interpolation_method = (
        'Linear Delaunay triangulation (barycentric) in LCC projected '
        'coordinates, matching MPAS convert_mpas and ECMWF MIR approach '
        'for real-valued fields on unstructured Voronoi meshes'
    )
    nc.source_data = 's3://twc-graf-reforecast/rpm4km.static.nc (MPAS static file)'
    nc.close()
    print('Done.')


# =======================================================

import time
t0 = time.time()

# 1. Read MPAS global terrain
lat_deg, lon_deg, ter = read_mpas_terrain(STATIC_FILE)

# 2. Read European target grid from GRIB
tgt_lats, tgt_lons, proj_params = read_euro_target_grid(EURO_GRIB_EXAMPLE)
ny, nx = tgt_lats.shape

# 3. Filter MPAS points to padded European bounding box
lat_min = tgt_lats.min() - LAT_PAD
lat_max = tgt_lats.max() + LAT_PAD
lon_min = tgt_lons.min() - LON_PAD
lon_max = tgt_lons.max() + LON_PAD
print(f'\nFiltering MPAS to bbox: lat [{lat_min:.1f}, {lat_max:.1f}],'
      f' lon [{lon_min:.1f}, {lon_max:.1f}]')
src_lat, src_lon, src_ter = filter_mpas_to_bbox(
    lat_deg, lon_deg, ter, lat_min, lat_max, lon_min, lon_max)

# 4. Build LCC projector
transformer = build_lcc_projector(proj_params)

# 5. Interpolate: barycentric linear in LCC projected coordinates
print('\nInterpolating terrain...')
terrain_height = interpolate_terrain(
    src_lon, src_lat, src_ter, tgt_lons, tgt_lats, transformer)

print(f'Interpolation complete in {time.time() - t0:.1f} s')

# 6. Smooth with primary sigma
print(f'\nSmoothing with sigma={SIGMA_PRIMARY} grid points...')
terrain_height_smoothed = ndimage.gaussian_filter(terrain_height, SIGMA_PRIMARY)
terrain_height_local_difference = terrain_height - terrain_height_smoothed

# 7. Smooth at multiple sigmas
nsigma = len(SIGMAS)
terrain_height_smoothed_multisigma = np.zeros((nsigma, ny, nx), dtype=float)
for i, sigma in enumerate(SIGMAS):
    print(f'  Smoothing sigma={sigma}...')
    terrain_height_smoothed_multisigma[i] = ndimage.gaussian_filter(terrain_height, sigma)

# 8. Terrain gradients (raw and smoothed)
print('\nComputing terrain slopes...')
dterrain_dlon, dterrain_dlat = terrain_slopes(terrain_height, tgt_lons, tgt_lats, ny, nx)
dterrain_dlon_smoothed, dterrain_dlat_smoothed = terrain_slopes(
    terrain_height_smoothed, tgt_lons, tgt_lats, ny, nx)
dterrain_dlon_difference, dterrain_dlat_difference = terrain_slopes(
    terrain_height_local_difference, tgt_lons, tgt_lats, ny, nx)

# 9. Write output
write_to_netcdf(
    OUTFILE, ny, nx, nsigma, tgt_lons, tgt_lats, SIGMAS,
    terrain_height, terrain_height_smoothed, terrain_height_local_difference,
    dterrain_dlon, dterrain_dlat,
    dterrain_dlon_smoothed, dterrain_dlat_smoothed,
    dterrain_dlon_difference, dterrain_dlat_difference,
    terrain_height_smoothed_multisigma)

print(f'\nTotal time: {time.time() - t0:.1f} s')
print(f'Output: {OUTFILE}')
