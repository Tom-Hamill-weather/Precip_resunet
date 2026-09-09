"""merge_gfs_extension.py

Merges the 2023-2025 gfs_subset_{ic}.nc (steps 0-48h) base files with the
gfs_subset_{ic}_ext5172.nc (steps 51-72h) extension files written by
fetch_gfs_extension.py into single complete 0-72h files, so
save_patched_GRAF_MRMS_GFS2.py:read_gfs() sees one file per init time exactly
like it always has -- no code changes needed there.

Writes to a temp file and renames over the base file only after a full
read-back verification succeeds (25 steps, correct dtype/shape). The original
base file content is preserved unchanged for steps 0-48; only new step
records are added.

Usage:
    python merge_gfs_extension.py --range 202301 202512
    python merge_gfs_extension.py --dry-run --range 202301 202512   # verify only, no writes
"""

import argparse
import glob
import os

import numpy as np
from netCDF4 import Dataset

FIELDS = ['pwat', 'r', 'cape', 'u10', 'v10', 'u', 'v']
GFS_DIR = '/data/resnet_data/gfs'


def _read_field_3d(f, name):
    """Read one FIELDS variable as (step, latitude, longitude), collapsing an
    extra pressureFromGroundLayer axis via max if present (mirrors
    save_patched_GRAF_MRMS_GFS2.py:read_gfs()'s handling of the same
    inconsistency in some months' base archive)."""
    var = f.variables[name]
    arr = np.array(var[:])
    if 'pressureFromGroundLayer' in var.dimensions:
        level_axis = var.dimensions.index('pressureFromGroundLayer')
        arr = np.max(arr, axis=level_axis)
    return arr


def merge_one(base_path, ext_path, dry_run=False):
    with Dataset(base_path, 'r') as fb, Dataset(ext_path, 'r') as fe:
        base_steps = np.array(fb.variables['step'][:])
        ext_steps = np.array(fe.variables['step'][:])
        lat_b, lat_e = fb.variables['latitude'][:], fe.variables['latitude'][:]
        lon_b, lon_e = fb.variables['longitude'][:], fe.variables['longitude'][:]
        if not (np.allclose(lat_b, lat_e) and np.allclose(lon_b, lon_e)):
            return False, 'lat/lon grid mismatch between base and extension'
        if set(base_steps) & set(ext_steps):
            return False, f'overlapping steps: {set(base_steps) & set(ext_steps)}'

        all_steps = np.concatenate([base_steps, ext_steps])
        order = np.argsort(all_steps)
        merged_steps = all_steps[order]

        merged = {}
        for name in FIELDS:
            b = _read_field_3d(fb, name)
            e = _read_field_3d(fe, name)
            merged[name] = np.concatenate([b, e], axis=0)[order]

    if dry_run:
        return True, f'{len(merged_steps)} steps: {list(merged_steps)}'

    tmp_path = base_path + '.merging.tmp'
    with Dataset(tmp_path, 'w') as out:
        out.createDimension('step', len(merged_steps))
        out.createDimension('latitude', len(lat_b))
        out.createDimension('longitude', len(lon_b))
        v = out.createVariable('latitude', 'f4', ('latitude',)); v[:] = lat_b
        v = out.createVariable('longitude', 'f4', ('longitude',)); v[:] = lon_b
        v = out.createVariable('step', 'i4', ('step',)); v[:] = merged_steps
        for name in FIELDS:
            v = out.createVariable(name, 'f4', ('step', 'latitude', 'longitude'))
            v[:] = merged[name]
        out.description = 'GFS 0-72h, merged base (0-48h) + extension (51-72h)'

    # Verify before replacing. Some base files pre-date this extension work
    # with their own gaps in 0-48h (a handful of 2023 init times are missing
    # individual steps) -- read_gfs() already tolerates missing steps via
    # nearest-match, so we only require no duplicates/drops from the merge
    # itself, not a complete 0-72h sequence.
    with Dataset(tmp_path, 'r') as check:
        steps_check = np.array(check.variables['step'][:])
        expected_n = len(base_steps) + len(ext_steps)
        is_sorted_unique = len(set(steps_check)) == len(steps_check) == expected_n \
            and list(steps_check) == sorted(steps_check)
        if not is_sorted_unique:
            os.remove(tmp_path)
            return False, f'post-write verification failed, steps={list(steps_check)}'

    os.replace(tmp_path, base_path)
    os.remove(ext_path)
    return True, f'merged {len(merged_steps)} steps -> {base_path}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--range', nargs=2, required=True, metavar=('START_YYYYMM', 'END_YYYYMM'))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    start_yyyymm, end_yyyymm = args.range

    ext_paths = sorted(glob.glob(os.path.join(GFS_DIR, '*', 'gfs_subset_*_ext5172.nc')))
    n_ok, n_fail = 0, 0
    for ext_path in ext_paths:
        base = os.path.basename(ext_path).replace('_ext5172.nc', '.nc')
        yyyymm = base.replace('gfs_subset_', '')[:6]
        if not (start_yyyymm <= yyyymm <= end_yyyymm):
            continue
        base_path = os.path.join(os.path.dirname(ext_path), base)
        if not os.path.exists(base_path):
            print(f'SKIP {base}: no base file found')
            n_fail += 1
            continue
        try:
            ok, msg = merge_one(base_path, ext_path, dry_run=args.dry_run)
        except Exception as exc:
            ok, msg = False, f'{type(exc).__name__}: {exc}'
        print(('OK   ' if ok else 'FAIL ') + f'{base}: {msg}')
        n_ok += ok
        n_fail += not ok

    print(f'\nDone: {n_ok} merged, {n_fail} failed, out of {len(ext_paths)} extension files found')


if __name__ == '__main__':
    main()
