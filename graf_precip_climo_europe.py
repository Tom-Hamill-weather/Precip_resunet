"""graf_precip_climo_europe.py

European-grid counterpart to graf_precip_climo.py: loader for the
WorldClim-only monthly precip climatology built by
build_precip_climo_europe.py -> /data/resnet_data/static/precip_climo_europe.nc,
(12, 723, 666) mm/month on the GRAF Europe grid. Same log1p normalization
convention as the CONUS module so 'precip_climo' channel values are
directly comparable/consistent with the checkpoint's expected [0,1] range.

Inference-only (no training on Europe data), so only the full-domain
month-indexed slice is needed — no per-patch KDTree lookup like the CONUS
module's sample_climo_patch.
"""
import numpy as np
import netCDF4

PRECIP_CLIMO_NC = '/data/resnet_data/static/precip_climo_europe.nc'
PRECIP_CLIMO_LOG_MAX = float(np.log1p(1500.0))  # same ceiling as CONUS module

_CACHE = {}


def _lognorm(mm):
    return np.clip(np.log1p(np.clip(mm, 0.0, None)) / PRECIP_CLIMO_LOG_MAX,
                   0.0, 1.0).astype(np.float32)


def _load():
    if 'climo' not in _CACHE:
        ds = netCDF4.Dataset(PRECIP_CLIMO_NC)
        climo = np.array(ds['precip_climo'][:], dtype=np.float32)  # (12, ny, nx) mm/month
        ds.close()
        _CACHE['climo'] = climo
    return _CACHE['climo']


def fulldomain_climo(month):
    """Full-domain (ny, nx) log1p-normalized climatology slice for one
    calendar month, on the GRAF Europe grid."""
    climo = _load()
    return _lognorm(climo[month - 1])
