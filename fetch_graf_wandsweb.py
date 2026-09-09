"""fetch_graf_wandsweb.py

Backfills the local hdo-graf_conus GRIB2 mirror at /data/resnet_data/GRAF/hdo-graf_conus/
from the WANDS archive, closing the gap left by the fact that this AWS box has no route
to the Cray (10.66.63.22) or to wandsweb.wxsci.qa.fcst.weather.com over HTTPS (both time
out at the network level -- confirmed 2026-08-29). Paul Bayer pointed at a plain-HTTP
internal address, http://10.233.22.117, which IS reachable from this box and serves the
same /archive/grid/... tree (confirmed 2026-08-31: hdo-graf_conus data present through
today, 20260831).

Local archive currently stops at 2026-02-23 -- this fills 2026-02-24 onward so
build_patch_pools_graf.py can build 2026 season pools (see project_hourly_resunet_season_film
memory for the rest of that pipeline).

Only APCP.SFC.grb2 files are fetched (the only file type get_filenames() in
save_patched_GRAF_MRMS_GFS2.py ever reads -- the companion GOV.SFC.grb2 files are
unused by any script in this repo, so they're skipped to save bandwidth/disk).

Usage:
    # backfill a date range (inclusive), all 4 cycles, leads 1-72h
    python3 fetch_graf_wandsweb.py --start 20260224 --end 20260831 --workers 8
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from urllib.parse import quote

import requests

WANDSWEB_BASE = 'http://10.233.22.117/archive/grid/hdo-graf_conus'
GRAF_DIR = '/data/resnet_data/GRAF/hdo-graf_conus'
CYCLES = ['00', '06', '12', '18']
LEADS = list(range(1, 73))  # PT1H .. PT72H, matches existing archive's hourly coverage

_session = requests.Session()
_session.mount('http://', requests.adapters.HTTPAdapter(pool_maxsize=32))


def build_filename(cyyyymmdd, chh, lead):
    init_dt = datetime.strptime(cyyyymmdd + chh, '%Y%m%d%H')
    fcst_dt = init_dt + timedelta(hours=lead)
    return (f"grid.hdo-graf_conus.{fcst_dt:%Y%m%d}T{fcst_dt:%H}0000Z."
            f"{cyyyymmdd}T{chh}0000Z.PT{lead}H.CONUS@4km.APCP.SFC.grb2")


def fetch_one(url, local_path):
    resp = _session.get(url, timeout=30)
    if resp.status_code == 200:
        tmp_path = local_path + '.part'
        with open(tmp_path, 'wb') as f:
            f.write(resp.content)
        os.rename(tmp_path, local_path)
        return 'ok'
    elif resp.status_code == 404:
        return 'missing'
    else:
        return f'http_{resp.status_code}'


def fetch_cycle(cyyyymmdd, chh):
    local_dir = os.path.join(GRAF_DIR, cyyyymmdd, chh)
    os.makedirs(local_dir, exist_ok=True)
    counts = {'ok': 0, 'skip': 0, 'missing': 0, 'error': 0}
    for lead in LEADS:
        fname = build_filename(cyyyymmdd, chh, lead)
        local_path = os.path.join(local_dir, fname)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            counts['skip'] += 1
            continue
        url = f"{WANDSWEB_BASE}/{cyyyymmdd}/{chh}/{quote(fname)}"
        try:
            result = fetch_one(url, local_path)
        except requests.RequestException as e:
            print(f'  WARNING {cyyyymmdd}/{chh} lead {lead}h: {e}', flush=True)
            counts['error'] += 1
            continue
        if result == 'ok':
            counts['ok'] += 1
        elif result == 'missing':
            counts['missing'] += 1
        else:
            print(f'  WARNING {cyyyymmdd}/{chh} lead {lead}h: {result}', flush=True)
            counts['error'] += 1
    return cyyyymmdd, chh, counts


def daterange(start, end):
    d = datetime.strptime(start, '%Y%m%d')
    dend = datetime.strptime(end, '%Y%m%d')
    while d <= dend:
        yield d.strftime('%Y%m%d')
        d += timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True, help='YYYYMMDD, inclusive')
    ap.add_argument('--end', required=True, help='YYYYMMDD, inclusive')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    tasks = [(d, c) for d in daterange(args.start, args.end) for c in CYCLES]
    print(f'Backfilling {len(tasks)} date/cycle combos from {args.start} to {args.end} '
          f'({args.workers} workers)', flush=True)

    total = {'ok': 0, 'skip': 0, 'missing': 0, 'error': 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(fetch_cycle, d, c) for d, c in tasks]
        done = 0
        for fut in as_completed(futures):
            cyyyymmdd, chh, counts = fut.result()
            for k in total:
                total[k] += counts[k]
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f'[{done}/{len(tasks)}] {cyyyymmdd}/{chh} done -- '
                      f'running totals: {total}', flush=True)

    print(f'FINISHED. totals: {total}', flush=True)


if __name__ == '__main__':
    main()
