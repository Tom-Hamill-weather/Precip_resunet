"""graf_precip_climo.py

Shared loader for the GRAF-grid blended monthly precipitation climatology
(PRISM core + WorldClim non-CONUS land + ERA5 offshore, built by
build_precip_climo_graf.py -> /data/resnet_data/static/precip_climo_graf.nc,
(12, ny, nx) mm/month on the exact GRAF grid). Added as a static
'precip_climo' input channel to both the gamma_mixture_season and
gamma_climatology_season trainers/inference scripts (2026-09-23), on Tom's
read that without an explicit climatology feature these models can't
reproduce PRISM-sharp forecast/observed climatology from sin/cos(doy) +
local terrain-gradient channels alone. Centralized here so all four
call sites (2 trainers, 2 inference scripts) share one loading/
normalization convention.

Normalization matches HRRRcal's graf7clim precedent exactly
(HRRRcal/pytorch_train_hrrr_gamma_mixture.py:165,466): log1p(mm/month)
scaled by log1p(1500), clipped to [0,1] -- already in the model's expected
[0,1] input range, so NORM_BOUNDS['precip_climo'] = (0.0, 1.0) downstream
(the generic (x-lo)/(hi-lo) normalizer in each trainer becomes a no-op for
this channel, same convention already used for sin_sh/cos_sh).
"""
import numpy as np
import netCDF4

PRECIP_CLIMO_NC = '/data/resnet_data/static/precip_climo_graf.nc'
GRAF_TERRAIN_NC = '/home/thamill/resnet/GRAF_CONUS_terrain_info.nc'
PRECIP_CLIMO_LOG_MAX = float(np.log1p(1500.0))  # ~7.31, same ceiling as HRRRcal

_CACHE = {}


def _lognorm(mm):
    return np.clip(np.log1p(np.clip(mm, 0.0, None)) / PRECIP_CLIMO_LOG_MAX,
                   0.0, 1.0).astype(np.float32)


def _load():
    if 'climo' not in _CACHE:
        ds = netCDF4.Dataset(PRECIP_CLIMO_NC)
        climo = np.array(ds['precip_climo'][:], dtype=np.float32)  # (12, ny, nx) mm/month
        ds.close()
        tds = netCDF4.Dataset(GRAF_TERRAIN_NC)
        lats = np.array(tds['lats'][:], dtype=np.float64)
        lons = np.array(tds['lons'][:], dtype=np.float64)
        tds.close()
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([lats.ravel(), lons.ravel()]))
        _CACHE['climo'] = climo
        _CACHE['tree'] = tree
        _CACHE['shape'] = lats.shape
    return _CACHE['climo'], _CACHE['tree'], _CACHE['shape']


def preload():
    """Call once before spawning DataLoader workers so forked workers
    inherit the loaded array + KDTree via copy-on-write instead of each
    separately rebuilding it (same fork-safe-globals convention as
    build_patch_pools_graf.py's _W_TERRAIN)."""
    _load()


def sample_climo_patch(month, center_lat, center_lon, half=48):
    """Nearest-grid-index lookup of (center_lat, center_lon) into the GRAF
    grid, then a (2*half, 2*half) log1p-normalized window of that month's
    climatology -- exact per-pixel spatial alignment with the terrain/GRAF/
    MRMS training patches, since they were cut from the same grid. A
    lookup is needed (rather than a stored offset) because the patch pool
    (build_patch_pools_graf.py) only stores each patch's CENTER lat/lon
    (meta_lat/meta_lon), not the row/col it was cut from."""
    climo, tree, (ny, nx) = _load()
    _, idx = tree.query([center_lat, center_lon])
    jy, ix = divmod(int(idx), nx)
    y0, y1 = jy - half, jy + half
    x0, x1 = ix - half, ix + half
    y0c, y1c = max(0, y0), min(ny, y1)
    x0c, x1c = max(0, x0), min(nx, x1)
    patch = np.zeros((2 * half, 2 * half), dtype=np.float32)
    patch[y0c - y0:y0c - y0 + (y1c - y0c),
          x0c - x0:x0c - x0 + (x1c - x0c)] = climo[month - 1, y0c:y1c, x0c:x1c]
    return _lognorm(patch)


def fulldomain_climo(month):
    """Full-domain (ny, nx) log1p-normalized climatology slice for one
    calendar month -- for whole-domain inference, where the target grid IS
    the GRAF grid precip_climo_graf.nc was built on (no lookup needed)."""
    climo, _, _ = _load()
    return _lognorm(climo[month - 1])


def fulldomain_climo_coarse4x(month, factor=4):
    """4x-block-averaged version of fulldomain_climo, for the whole-domain
    training prototype at 4x coarser resolution (2026-09-24). Averages the
    RAW mm/month field first, then log1p-normalizes the coarse-cell mean --
    not the other way around (averaging already-log1p values would bias
    toward the geometric rather than arithmetic mean of the sub-cell
    values, which isn't what "typical accumulation in this coarse cell"
    should mean)."""
    climo, _, (ny, nx) = _load()
    raw = climo[month - 1]
    coarse = raw.reshape(ny // factor, factor, nx // factor, factor).mean(axis=(1, 3))
    return _lognorm(coarse)
