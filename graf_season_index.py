"""graf_season_index.py

Index-building and FiLM-conditioning helpers for season-pooled, lead-pooled
GRAF ResUNet training. Mirrors the HRRR calibration recipe
(/home/thamill/HRRRcal/pytorch_train_hrrr_gamma_mixture.py:228-412) applied to
the GRAF zarr patch pools written by build_patch_pools_graf.py.

Patch pool contract (one zarr group per YYYYMM, written by
build_patch_pools_graf.py):
  arrays   GRAF, MRMS, MRMS_qual, terrain_diff, dt_dlon, dt_dlat, GFS_r
           each (n_patches, 96, 96) float32
  meta     meta_day   (n_patches,) int32   YYYYMMDD of the INIT time
           meta_cycle (n_patches,) int32   init hour (0, 6, 12, or 18)
           meta_lead  (n_patches,) int32   forecast lead time (h)
           meta_lat   (n_patches,) float32 patch-center latitude
           meta_lon   (n_patches,) float32 patch-center longitude (-180..180)
  attrs    n_patches (int), complete (bool)
"""

import datetime
import os

import numpy as np
import zarr

SEASON_MONTHS = {
    'DJF': {12, 1, 2},
    'MAM': {3, 4, 5},
    'JJA': {6, 7, 8},
    'SON': {9, 10, 11},
}

LEAD_MAX = 72  # matches build_patch_pools_graf.py's LEADS = range(3, 73, 3)


def _yyyymm_list(patches_dir):
    """Sorted list of YYYYMM strings for complete zarr pools."""
    months = []
    if not os.path.isdir(patches_dir):
        return months
    for name in sorted(os.listdir(patches_dir)):
        if not name.endswith('.zarr'):
            continue
        path = os.path.join(patches_dir, name)
        try:
            g = zarr.open_group(path, mode='r')
            if g.attrs.get('complete'):
                months.append(name[:-5])
        except Exception:
            pass
    return months


def _index_from_zarr(zarr_path, lead_filter=None):
    """Return list of (zarr_path, patch_idx, day, cycle, lead, lat, lon) for one store."""
    g = zarr.open_group(zarr_path, mode='r')
    n = g.attrs['n_patches']
    days   = np.array(g['meta_day'][:n],   dtype=np.int32)
    cycles = np.array(g['meta_cycle'][:n], dtype=np.int32)
    leads  = np.array(g['meta_lead'][:n],  dtype=np.int32)
    lats   = np.array(g['meta_lat'][:n],   dtype=np.float32)
    lons   = np.array(g['meta_lon'][:n],   dtype=np.float32)
    indices = np.arange(n, dtype=np.int32)
    if lead_filter is not None:
        lead_set = {lead_filter} if isinstance(lead_filter, int) else set(lead_filter)
        mask = np.isin(leads, list(lead_set))
        indices, days, cycles, leads, lats, lons = (
            indices[mask], days[mask], cycles[mask], leads[mask], lats[mask], lons[mask])
    return [(zarr_path, int(idx), int(d), int(c), int(l), float(la), float(lo))
            for idx, d, c, l, la, lo in zip(indices, days, cycles, leads, lats, lons)]


def _day_of_year(day):
    yr, mo, dom = day // 10000, (day // 100) % 100, day % 100
    return datetime.date(yr, mo, dom).timetuple().tm_yday


def build_seasonal_index(patches_dir, season, holdout=False, year=None, lead_filter=None):
    """All patches for a season, pooled across every available year.

    Leakage-safe split: every 4th 7-day block of the year is held out
    (holdout=True -> that 20%), matching
    HRRRcal/pytorch_train_hrrr_gamma_mixture.py:300-333.
    """
    if season not in SEASON_MONTHS:
        raise ValueError(f'season must be one of {list(SEASON_MONTHS)}')
    target_months = SEASON_MONTHS[season]
    entries = []
    for yyyymm in _yyyymm_list(patches_dir):
        mm, yyyy = int(yyyymm[4:]), int(yyyymm[:4])
        if mm not in target_months:
            continue
        if year is not None and yyyy != year:
            continue
        path = os.path.join(patches_dir, f'{yyyymm}.zarr')
        for entry in _index_from_zarr(path, lead_filter=lead_filter):
            _, _, day, _, _, _, _ = entry
            block = (_day_of_year(day) - 1) // 7
            is_holdout = (block % 4 == 3)
            if is_holdout == holdout:
                entries.append(entry)
    return entries


def make_cond(day, cycle, lead, center_lon, lead_max=LEAD_MAX):
    """5-scalar FiLM conditioning vector: [sin_doy, cos_doy, sin_solar_hour,
    cos_solar_hour, lead_norm]. Mirrors
    HRRRcal/pytorch_train_hrrr_gamma_mixture.py:685-704."""
    doy = _day_of_year(day)
    sin_doy = np.sin(2 * np.pi * (doy - 1) / 365.0)
    cos_doy = np.cos(2 * np.pi * (doy - 1) / 365.0)

    utc_valid = (cycle + lead) % 24
    solar_hour = (utc_valid + center_lon / 15.0) % 24
    sin_sh = np.sin(2 * np.pi * solar_hour / 24.0)
    cos_sh = np.cos(2 * np.pi * solar_hour / 24.0)

    lead_norm = lead / float(lead_max)
    return np.array([sin_doy, cos_doy, sin_sh, cos_sh, lead_norm], dtype=np.float32)
