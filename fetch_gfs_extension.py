"""fetch_gfs_extension.py

Backfills forecast hours 51-72h (3h steps) for the existing gfs_subset_*.nc
archive at /data/resnet_data/gfs/, which currently only covers steps 0-48h.
Needed so the season+FiLM ResUNet training (pytorch_train_resunet_gamma_mixture_season.py)
can pool patches out to 72h lead.

Source: s3://noaa-gfs-bdp-pds (NOAA GFS 0.25-deg pgrb2, public, byte-range
HTTP GET on .idx-located GRIB2 messages -- same pattern already used for
on-the-fly RH download in resunet_inference_gamma_mixture_optimized_europe.py).
Confirmed reachable back to 2021-02-26.

Region/fields match the existing gfs_subset_*.nc files exactly (verified via
their stored GRIB_* attrs): lat 60->10N, lon 220->305E at 0.25 deg (201x341),
fields PWAT/entire atmosphere, RH/entire atmosphere (single layer),
CAPE/surface, UGRD+VGRD/10 m above ground, UGRD+VGRD/700 mb.

Usage:
    # single test case, prints values, does not write a file
    python fetch_gfs_extension.py --test 2025120100 51

    # backfill one init time, all of steps 51..72, writes
    # {GFS_DIR}/{yyyymm}/gfs_subset_{cyyyymmddhh}_ext5172.nc
    python fetch_gfs_extension.py --init 2025120100

    # backfill every existing gfs_subset_*.nc under GFS_DIR in [start,end]
    python fetch_gfs_extension.py --range 202301 202512 --workers 16
"""

import argparse
import glob
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

GFS_S3_BASE = 'https://noaa-gfs-bdp-pds.s3.amazonaws.com'
GFS_DIR = '/data/resnet_data/gfs'

EXT_STEPS = list(range(51, 73, 3))   # 51,54,...,72 -- extends the existing 0..48 archive
ALL_STEPS = list(range(0, 73, 3))    # 0,3,...,72 -- full build for months with no base file at all
CYCLES = ['00', '06', '12', '18']

# (output variable name, .idx search substring, GRIB_typeOfLevel for sanity check)
FIELDS = [
    ('pwat', 'PWAT:entire atmosphere', 'atmosphereSingleLayer'),
    ('r',    'RH:entire atmosphere (considered as a single layer)', 'atmosphereSingleLayer'),
    ('cape', 'CAPE:surface', 'surface'),
    ('u10',  'UGRD:10 m above ground', 'heightAboveGround'),
    ('v10',  'VGRD:10 m above ground', 'heightAboveGround'),
    ('u',    'UGRD:700 mb', 'isobaricInhPa'),
    ('v',    'VGRD:700 mb', 'isobaricInhPa'),
]

# Existing gfs_subset_*.nc region: lat 60->10N, lon 220->305E at 0.25 deg.
# Full GFS 0.25-deg grid: lat 90->-90 (721 pts), lon 0->359.75 (1440 pts).
LAT0, LON0 = 90.0, 0.0
DLAT, DLON = 0.25, 0.25
LAT_START, LAT_END = 60.0, 10.0   # inclusive, descending
LON_START, LON_END = 220.0, 305.0  # inclusive, ascending


def _box_slices():
    j0 = round((LAT0 - LAT_START) / DLAT)
    j1 = round((LAT0 - LAT_END) / DLAT) + 1
    i0 = round((LON_START - LON0) / DLON)
    i1 = round((LON_END - LON0) / DLON) + 1
    return slice(j0, j1), slice(i0, i1)


def _gfs_url(cyyyymmddhh, forecast_hour):
    cyyyymmdd, chh = cyyyymmddhh[:8], cyyyymmddhh[8:10]
    fhr = f'{int(forecast_hour):03d}'
    base = f'{GFS_S3_BASE}/gfs.{cyyyymmdd}/{chh}/atmos/gfs.t{chh}z.pgrb2.0p25.f{fhr}'
    return base, base + '.idx'


def _parse_index(idx_text, search_str):
    lines = [l for l in idx_text.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        if search_str in line:
            byte_start = int(line.split(':')[1])
            byte_end = int(lines[i + 1].split(':')[1]) - 1 if i + 1 < len(lines) else None
            return byte_start, byte_end
    return None, None


def _download_range(data_url, byte_start, byte_end, session):
    range_hdr = f'bytes={byte_start}-{byte_end}' if byte_end is not None else f'bytes={byte_start}-'
    resp = session.get(data_url, headers={'Range': range_hdr}, timeout=60)
    if resp.status_code not in (200, 206):
        raise IOError(f'HTTP {resp.status_code} fetching {data_url} range {range_hdr}')
    return resp.content


def fetch_one(cyyyymmddhh, forecast_hour, session=None):
    """Download all FIELDS for one (init_time, lead), cropped to the CONUS box.

    Returns dict {varname: (201,341) float32 array} or None on failure.
    """
    import pygrib

    session = session or requests.Session()
    data_url, idx_url = _gfs_url(cyyyymmddhh, forecast_hour)

    idx_resp = session.get(idx_url, timeout=30)
    if idx_resp.status_code != 200:
        print(f'  [{cyyyymmddhh} f{forecast_hour:03d}] index not found: HTTP {idx_resp.status_code}')
        return None
    idx_text = idx_resp.text

    jsl, isl = _box_slices()
    out = {}
    for varname, search_str, _ in FIELDS:
        byte_start, byte_end = _parse_index(idx_text, search_str)
        if byte_start is None:
            print(f'  [{cyyyymmddhh} f{forecast_hour:03d}] field not found in index: {search_str}')
            return None
        try:
            raw = _download_range(data_url, byte_start, byte_end, session)
        except Exception as e:
            print(f'  [{cyyyymmddhh} f{forecast_hour:03d}] download failed for {varname}: {e}')
            return None

        with tempfile.NamedTemporaryFile(suffix='.grb2', delete=False) as tmp:
            tmp.write(raw)
            tmpname = tmp.name
        try:
            f = pygrib.open(tmpname)
            grb = f.read(1)[0]
            vals_global = grb.values.astype(np.float32)  # (721, 1440), lat 90->-90
            f.close()
        finally:
            os.unlink(tmpname)

        vals_global = np.where(np.isnan(vals_global), 0.0, vals_global)
        out[varname] = vals_global[jsl, isl].copy()

    return out


def backfill_init_time(cyyyymmddhh, workers=4, overwrite=False, steps=None, out_path=None):
    """Fetch `steps` (default EXT_STEPS) for one init time and write a netCDF file.

    Default (steps=None): extension mode, writes
    {GFS_DIR}/{yyyymm}/gfs_subset_{ic}_ext5172.nc alongside the existing 0-48h base file.
    Pass steps=ALL_STEPS and out_path=.../gfs_subset_{ic}.nc for a full 0-72h build
    (months with no base file at all, e.g. 2026 which has none yet).
    """
    steps = steps or EXT_STEPS
    yyyymm = cyyyymmddhh[:6]
    if out_path is None:
        out_path = os.path.join(GFS_DIR, yyyymm, f'gfs_subset_{cyyyymmddhh}_ext5172.nc')
    if os.path.exists(out_path) and not overwrite:
        return out_path, True

    session = requests.Session()
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_one, cyyyymmddhh, fh, session): fh for fh in steps}
        for fut in as_completed(futures):
            fh = futures[fut]
            data = fut.result()
            if data is None:
                print(f'{cyyyymmddhh}: FAILED at step {fh}h -- aborting this init time')
                return out_path, False
            results[fh] = data

    _write_netcdf(out_path, cyyyymmddhh, results)
    print(f'{cyyyymmddhh}: wrote {out_path}')
    return out_path, True


def full_build_range(start_yyyymmdd, end_yyyymmdd, workers=8):
    """Build complete 0-72h gfs_subset_{ic}.nc files for every day/cycle in
    [start_yyyymmdd, end_yyyymmdd] that doesn't already have a base file.
    For months with no existing archive at all (e.g. 2026)."""
    from dateutils import dateshift

    day = start_yyyymmdd
    ics = []
    while day <= end_yyyymmdd:
        for cycle in CYCLES:
            ics.append(day + cycle)
        day = dateshift(day + '00', 24)[:8]

    print(f'Full 0-72h build: {len(ics)} init times in [{start_yyyymmdd}, {end_yyyymmdd}]')
    for i, ic in enumerate(ics):
        yyyymm = ic[:6]
        out_path = os.path.join(GFS_DIR, yyyymm, f'gfs_subset_{ic}.nc')
        if os.path.exists(out_path):
            print(f'{ic}: base file already exists, skipping')
            continue
        backfill_init_time(ic, workers=workers, steps=ALL_STEPS, out_path=out_path)
        if (i + 1) % 20 == 0:
            print(f'  progress: {i + 1}/{len(ics)}')


def _write_netcdf(out_path, cyyyymmddhh, results):
    from netCDF4 import Dataset

    jsl, isl = _box_slices()
    lat_full = LAT0 - DLAT * np.arange(721)
    lon_full = LON0 + DLON * np.arange(1440)
    lats = lat_full[jsl]
    lons = lon_full[isl]
    steps = sorted(results.keys())

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with Dataset(out_path, 'w') as nc:
        nc.createDimension('step', len(steps))
        nc.createDimension('latitude', len(lats))
        nc.createDimension('longitude', len(lons))

        v = nc.createVariable('latitude', 'f4', ('latitude',)); v[:] = lats
        v = nc.createVariable('longitude', 'f4', ('longitude',)); v[:] = lons
        v = nc.createVariable('step', 'i4', ('step',)); v[:] = steps

        for varname, _, _ in FIELDS:
            v = nc.createVariable(varname, 'f4', ('step', 'latitude', 'longitude'))
            v[:] = np.stack([results[s][varname] for s in steps])

        nc.description = f'GFS 51-72h extension for {cyyyymmddhh}, source s3://noaa-gfs-bdp-pds'


def _existing_init_times(start_yyyymm, end_yyyymm):
    ics = []
    for path in sorted(glob.glob(os.path.join(GFS_DIR, '*', 'gfs_subset_*.nc'))):
        base = os.path.basename(path)
        if '_ext' in base:
            continue
        ic = base.replace('gfs_subset_', '').replace('.nc', '')
        if start_yyyymm <= ic[:6] <= end_yyyymm:
            ics.append(ic)
    return sorted(ics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', nargs=2, metavar=('CYYYYMMDDHH', 'FORECAST_HOUR'))
    ap.add_argument('--init', metavar='CYYYYMMDDHH')
    ap.add_argument('--range', nargs=2, metavar=('START_YYYYMM', 'END_YYYYMM'))
    ap.add_argument('--full-range', nargs=2, metavar=('START_YYYYMMDD', 'END_YYYYMMDD'),
                    help='full 0-72h build for days with no existing base gfs_subset file '
                         '(e.g. 2026, which has none yet)')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    if args.test:
        ic, fh = args.test
        data = fetch_one(ic, int(fh))
        if data is None:
            print('FAILED')
            return
        for k, v in data.items():
            print(f'{k}: shape={v.shape} min={v.min():.2f} max={v.max():.2f} mean={v.mean():.2f}')
        return

    if args.init:
        backfill_init_time(args.init, workers=args.workers)
        return

    if args.range:
        start_yyyymm, end_yyyymm = args.range
        ics = _existing_init_times(start_yyyymm, end_yyyymm)
        print(f'Backfilling {len(ics)} init times in [{start_yyyymm}, {end_yyyymm}]')
        for i, ic in enumerate(ics):
            backfill_init_time(ic, workers=args.workers)
            if (i + 1) % 20 == 0:
                print(f'  progress: {i + 1}/{len(ics)}')
        return

    if args.full_range:
        start_yyyymmdd, end_yyyymmdd = args.full_range
        full_build_range(start_yyyymmdd, end_yyyymmdd, workers=args.workers)
        return

    ap.print_help()


if __name__ == '__main__':
    main()
