"""build_patch_pools_graf.py

Usage:
    python build_patch_pools_graf.py start_yyyymm end_yyyymm [--workers N] [--pools-dir DIR]

Builds season/lead-pooled GRAF/MRMS/GFS/terrain patch pools, one zarr store
per calendar month, covering every available init time (00/06/12/18Z) and
every 3-hourly lead time 3-72h. This replaces the current per-(IC-date,lead)
pickle scheme (save_patched_GRAF_MRMS_GFS2.py) for season-pooled, FiLM
lead-pooled training (pytorch_train_resunet_gamma_mixture_season.py):
training by season needs many years of patches pooled together rather than
one ~8-month recency window, and FiLM lead-pooling needs every lead time
represented per init time rather than one pickle per lead.

Architecture mirrors HRRRcal/build_patch_pools_hrrr.py: one zarr group per
YYYYMM, `complete` attr marks a finished month so a killed/resumed run just
skips finished months and rebuilds the (single) interrupted one.

Reuses GRAFDataProcessor (GRIB/MRMS/GFS reading, terrain, non-overlapping
patch selection) from save_patched_GRAF_MRMS_GFS2.py unchanged -- only the
accumulation/storage strategy is new (zarr instead of one in-RAM pickle per
lead, and looping over every lead per init time instead of one lead per
script invocation).

Patch pool contract: see graf_season_index.py docstring.
"""

import argparse
import calendar
import os
import shutil
import time
from multiprocessing import Pool

import numpy as np
import zarr

from dateutils import dateshift
from save_patched_GRAF_MRMS_GFS2 import GRAFDataProcessor, detect_config

CYCLES = ['00', '06', '12', '18']
LEADS = list(range(3, 73, 3))  # 3..72h every 3h (24 leads)
PATCH_HALF = 48  # 96x96 patches

_PATCH_VARS = ['GRAF', 'MRMS', 'MRMS_qual', 'terrain_diff', 'dt_dlon', 'dt_dlat', 'GFS_r']
_META_VARS = ['meta_day', 'meta_cycle', 'meta_lead', 'meta_lat', 'meta_lon']
_META_DTYPES = {'meta_day': 'i4', 'meta_cycle': 'i4', 'meta_lead': 'i4',
                'meta_lat': 'f4', 'meta_lon': 'f4'}

# Per-worker-process globals set by _worker_init (fork-safe on Linux).
_W_PROCESSOR = None
_W_TERRAIN = None
_W_LATLONS = None


def _worker_init(config_file):
    global _W_PROCESSOR, _W_TERRAIN, _W_LATLONS
    import warnings
    warnings.filterwarnings('ignore')
    _W_PROCESSOR = GRAFDataProcessor(config_file)
    _W_TERRAIN = _W_PROCESSOR.read_terrain()
    _W_LATLONS = None


def _process_ic(ic):
    """Process one init time (10-char YYYYMMDDHH) across all LEADS.

    Returns a dict of lists (patch pool columns) or None if no patches.
    """
    global _W_PROCESSOR, _W_TERRAIN, _W_LATLONS
    processor = _W_PROCESSOR
    terrain_diff, terr_dlon, terr_dlat = _W_TERRAIN
    date, cycle = ic[:8], ic[8:10]

    batch = {k: [] for k in _PATCH_VARS + _META_VARS}
    r = PATCH_HALF

    for lead in LEADS:
        graf_file, _, _ = processor.get_filenames(ic, lead)
        need_ll = (_W_LATLONS is None)
        istat, precip_graf, lats, lons, _ = processor.read_grib_precip(
            graf_file, lead, compute_latlons=need_ll)
        if istat != 0:
            continue
        if need_ll:
            _W_LATLONS = (lats, lons)
        lats, lons = _W_LATLONS

        valid = dateshift(ic, lead)
        istat, precip_mrms, quality_mrms = processor.read_mrms(valid)
        if istat != 0:
            continue

        istat, gfs_data = processor.read_gfs(ic, lead)
        if istat != 0:
            continue

        j_idx, i_idx = processor.select_patches_nonoverlapping(
            precip_graf, quality_mrms, lats.shape[0], lats.shape[1], ic)
        if len(j_idx) == 0:
            continue

        gfs_patches = processor.interpolate_gfs_to_patches(gfs_data, lats, lons, j_idx, i_idx)

        for k, (jy, ix) in enumerate(zip(j_idx, i_idx)):
            y_sl, x_sl = slice(jy - r, jy + r), slice(ix - r, ix + r)
            batch['GRAF'].append(precip_graf[y_sl, x_sl].astype(np.float32))
            batch['MRMS'].append(precip_mrms[y_sl, x_sl].astype(np.float32))
            batch['MRMS_qual'].append(quality_mrms[y_sl, x_sl].astype(np.float32))
            batch['terrain_diff'].append(terrain_diff[y_sl, x_sl].astype(np.float32))
            batch['dt_dlon'].append(terr_dlon[y_sl, x_sl].astype(np.float32))
            batch['dt_dlat'].append(terr_dlat[y_sl, x_sl].astype(np.float32))
            batch['GFS_r'].append(gfs_patches[k]['r'].astype(np.float32))
            batch['meta_day'].append(int(date))
            batch['meta_cycle'].append(int(cycle))
            batch['meta_lead'].append(int(lead))
            batch['meta_lat'].append(float(lats[jy, ix]))
            batch['meta_lon'].append(float(lons[jy, ix]))

    if not batch['GRAF']:
        return None
    return batch


def _create_zarr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    g = zarr.open_group(path, mode='w')
    zst = zarr.codecs.ZstdCodec(level=4)
    for name in _PATCH_VARS:
        g.create_array(name, shape=(0, 96, 96), chunks=(64, 96, 96), dtype='f4', compressors=zst)
    for name in _META_VARS:
        g.create_array(name, shape=(0,), chunks=(4096,), dtype=_META_DTYPES[name])
    return g


def _append(g, batch, start):
    n = len(batch['GRAF'])
    if n == 0:
        return start
    end = start + n
    for name in _PATCH_VARS:
        arr = g[name]
        arr.resize((end,) + arr.shape[1:])
        arr[start:end] = np.stack(batch[name])
    for name in _META_VARS:
        arr = g[name]
        arr.resize((end,))
        arr[start:end] = np.array(batch[name], dtype=_META_DTYPES[name])
    return end


def _month_range(start_yyyymm, end_yyyymm):
    y, m = int(start_yyyymm[:4]), int(start_yyyymm[4:6])
    ey, em = int(end_yyyymm[:4]), int(end_yyyymm[4:6])
    months = []
    while (y, m) <= (ey, em):
        months.append(f'{y:04d}{m:02d}')
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def build_month(processor, config_file, yyyymm, zpath, workers):
    year, month = int(yyyymm[:4]), int(yyyymm[4:6])
    ndays = calendar.monthrange(year, month)[1]
    days = [f'{year:04d}{month:02d}{d:02d}' for d in range(1, ndays + 1)]
    ic_list = [day + cycle for day in days for cycle in CYCLES]

    g = _create_zarr(zpath)
    n_total = 0
    t0 = time.time()
    with Pool(workers, initializer=_worker_init, initargs=(config_file,)) as pool:
        for i, batch in enumerate(pool.imap_unordered(_process_ic, ic_list, chunksize=1)):
            if batch is not None:
                n_total = _append(g, batch, n_total)
            if (i + 1) % 20 == 0 or (i + 1) == len(ic_list):
                elapsed = time.time() - t0
                print(f'  {yyyymm}: {i + 1}/{len(ic_list)} init times, '
                      f'{n_total} patches, {elapsed:.0f}s', flush=True)

    g.attrs['n_patches'] = n_total
    g.attrs['yyyymm'] = yyyymm
    g.attrs['complete'] = True
    print(f'{yyyymm}: DONE, {n_total} patches in {time.time() - t0:.0f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('start_yyyymm')
    ap.add_argument('end_yyyymm')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument('--pools-dir', default=None)
    args = ap.parse_args()

    config_file = detect_config()
    processor = GRAFDataProcessor(config_file)
    base = processor.aws_base_path or '../resnet_data'
    pools_dir = args.pools_dir or os.path.join(base, 'graf_season_pools')
    os.makedirs(pools_dir, exist_ok=True)

    months = _month_range(args.start_yyyymm, args.end_yyyymm)
    print(f'Building {len(months)} months of patch pools into {pools_dir} '
          f'({len(LEADS)} leads x {len(CYCLES)} cycles/day, {args.workers} workers)')

    for yyyymm in months:
        zpath = os.path.join(pools_dir, f'{yyyymm}.zarr')
        if os.path.exists(zpath):
            try:
                g = zarr.open_group(zpath, mode='r')
                if g.attrs.get('complete'):
                    print(f'{yyyymm}: already complete, skipping')
                    continue
            except Exception:
                pass
        build_month(processor, config_file, yyyymm, zpath, args.workers)


if __name__ == '__main__':
    main()
