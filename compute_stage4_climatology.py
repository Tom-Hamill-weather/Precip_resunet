#!/usr/bin/env python3
"""
compute_stage4_climatology.py

Downloads NCEP Stage IV 1-h QPE from the Iowa Environmental Mesonet
archive (2020–2024) and accumulates exceedance counts to produce
climatological exceedance probabilities on the native Stage IV 4-km
polar-stereographic grid.

Output    : /data/resnet_data/stage4_climo_2020_2024.nc
Checkpoint: /data/resnet_data/stage4_climo_checkpoint.npz   (resume-safe)
Log       : /data/resnet_data/stage4_climo.log

Usage:
    python compute_stage4_climatology.py            # full run
    python compute_stage4_climatology.py --test     # 6 test days and exit
"""

import os, sys, signal, tempfile, logging, time
from datetime import datetime, timedelta, date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pygrib
import requests
from netCDF4 import Dataset

# ── configuration ─────────────────────────────────────────────────
START_DATE  = date(2020, 1, 1)
END_DATE    = date(2024, 12, 31)
THRESHOLDS  = [0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]   # mm
NT, NM, NH  = len(THRESHOLDS), 12, 24
NY, NX      = 881, 1121          # Stage IV polar-stereo grid
MISSING_VAL = 9999.0
MAX_WORKERS = 8                  # parallel download threads
IEM_BASE    = "https://mesonet.agron.iastate.edu/archive/data"
OUTPUT_FILE = Path("/data/resnet_data/stage4_climo_2020_2024.nc")
CHECKPOINT  = Path("/data/resnet_data/stage4_climo_checkpoint.npz")
LOG_FILE    = Path("/data/resnet_data/stage4_climo.log")
SAVE_EVERY  = 30     # days between checkpoint saves

# ── logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── graceful shutdown on SIGINT / SIGTERM ─────────────────────────
_stop = False
def _handle_sig(sig, frame):
    global _stop
    log.info(f"Signal {sig} received — will checkpoint after current day.")
    _stop = True
signal.signal(signal.SIGINT,  _handle_sig)
signal.signal(signal.SIGTERM, _handle_sig)


# ─────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────

def _hour_url(dt: datetime) -> str:
    return (f"{IEM_BASE}/{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/stage4/"
            f"ST4.{dt:%Y%m%d%H}.01h.grib")


def fetch_hour(dt: datetime, session: requests.Session):
    """Download one hourly GRIB file.  Returns (dt, bytes) or (dt, None)."""
    url = _hour_url(dt)
    for attempt in range(3):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200:
                return dt, r.content
            return dt, None          # 404 = missing hour in archive
        except requests.RequestException as exc:
            if attempt == 2:
                log.warning(f"  fetch failed {url}: {exc}")
                return dt, None
            time.sleep(2 ** attempt)


# ─────────────────────────────────────────────────────────────────
# GRIB decode
# ─────────────────────────────────────────────────────────────────

def decode_grib(content: bytes, dt: datetime):
    """Parse GRIB bytes → (precip float32 array, lats, lons) or (None,…)."""
    fd, path = tempfile.mkstemp(suffix=".grib")
    try:
        os.write(fd, content)
        os.close(fd)
        grbs = pygrib.open(path)
        msg  = grbs.select(name="Total Precipitation")[0]
        data = msg.values.astype(np.float32)
        lats, lons = (a.astype(np.float32) for a in msg.latlons())
        grbs.close()
        data[data >= MISSING_VAL] = np.nan
        return data, lats, lons
    except Exception as exc:
        log.warning(f"  GRIB decode error {dt}: {exc}")
        try: os.close(fd)
        except OSError: pass
        return None, None, None
    finally:
        try: os.unlink(path)
        except OSError: pass


# ─────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────────────────────────

def checkpoint_save(count_exc, count_tot, lats, lons, done_dates):
    # np.savez_compressed appends .npz automatically if not already present,
    # so use a tmp name that already ends in .npz to avoid a doubled suffix.
    tmp = Path(str(CHECKPOINT).replace('.npz', '_tmp.npz'))
    np.savez_compressed(
        str(tmp),
        count_exc  = count_exc,
        count_tot  = count_tot,
        lats       = lats,
        lons       = lons,
        done_dates = np.array(sorted(done_dates), dtype='U10'),
    )
    tmp.rename(CHECKPOINT)   # atomic replace — safe against partial writes
    log.info(f"  checkpoint saved ({len(done_dates)} days done)")


def checkpoint_load():
    if not CHECKPOINT.exists():
        return None
    log.info(f"Loading checkpoint: {CHECKPOINT}")
    c = np.load(CHECKPOINT, allow_pickle=True)
    return (c['count_exc'], c['count_tot'], c['lats'], c['lons'],
            set(c['done_dates'].tolist()))


# ─────────────────────────────────────────────────────────────────
# NetCDF output
# ─────────────────────────────────────────────────────────────────

def write_netcdf(count_exc, count_tot, lats, lons):
    log.info(f"Writing output: {OUTPUT_FILE}")
    # count_tot[month, hour] = number of valid files in that bin
    # broadcast to (NT, NM, NH, NY, NX) for division
    denom = count_tot[np.newaxis, :, :, np.newaxis, np.newaxis]
    with np.errstate(invalid='ignore', divide='ignore'):
        prob = np.where(denom > 0, count_exc / denom, np.nan).astype(np.float32)

    ds = Dataset(OUTPUT_FILE, 'w')
    ds.createDimension('threshold', NT)
    ds.createDimension('month',     NM)
    ds.createDimension('hour',      NH)
    ds.createDimension('y',         NY)
    ds.createDimension('x',         NX)
    kw = dict(zlib=True, complevel=5)

    v = ds.createVariable('climo_prob', 'f4',
                          ('threshold','month','hour','y','x'), **kw)
    v[:] = prob
    v.long_name = '1-h precipitation exceedance climatological probability'
    v.units     = '1'
    v.comment   = ('P(ST4 >= threshold[t] | month=m, utc_hour=h); '
                   'computed from Stage IV 1-h QPE 2020-2024')

    vt = ds.createVariable('threshold', 'f4', ('threshold',))
    vt[:] = np.array(THRESHOLDS, dtype=np.float32)
    vt.units = 'mm'; vt.long_name = 'Exceedance threshold'

    vm = ds.createVariable('month', 'i4', ('month',))
    vm[:] = np.arange(1, 13)
    vm.long_name = 'Month of year (1=Jan … 12=Dec)'

    vh = ds.createVariable('hour', 'i4', ('hour',))
    vh[:] = np.arange(0, 24)
    vh.long_name = 'UTC hour of 1-h accumulation valid time'

    vla = ds.createVariable('lat', 'f4', ('y','x'), **kw)
    vla[:] = lats;  vla.units = 'degrees_north'

    vlo = ds.createVariable('lon', 'f4', ('y','x'), **kw)
    vlo[:] = lons;  vlo.units = 'degrees_east'

    vct = ds.createVariable('count_total', 'f4', ('month','hour'), **kw)
    vct[:] = count_tot.astype(np.float32)
    vct.long_name = 'Valid file count per month-hour bin'

    ds.source  = 'Iowa Environmental Mesonet Stage IV archive (mesonet.agron.iastate.edu)'
    ds.period  = f'{START_DATE} to {END_DATE}'
    ds.created = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    ds.close()
    log.info("  netCDF written.")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    test_mode = '--test' in sys.argv

    # ── init or resume from checkpoint ────────────────────────────
    chk = checkpoint_load()
    if chk:
        count_exc, count_tot, lats, lons, done_dates = chk
        log.info(f"  Resuming: {len(done_dates)} days already processed")
    else:
        # count_exc[threshold, month, hour, y, x]  float32 ~7.4 GB
        count_exc  = np.zeros((NT, NM, NH, NY, NX), dtype=np.float32)
        count_tot  = np.zeros((NM, NH),              dtype=np.float32)
        lats = lons = None
        done_dates  = set()

    # ── build list of pending days ─────────────────────────────────
    all_days = []
    d = START_DATE
    while d <= END_DATE:
        if d.strftime('%Y-%m-%d') not in done_dates:
            all_days.append(d)
        d += timedelta(days=1)

    if test_mode:
        # Pick 6 days spread across seasons and years for a quick smoke test
        all_days = [date(2020, 1, 15), date(2021, 4, 10),
                    date(2022, 7,  4), date(2023, 10, 1),
                    date(2024, 3,  20), date(2024, 8, 5)]
        log.info(f"TEST MODE: processing {len(all_days)} sample days")
    else:
        log.info(f"Days remaining: {len(all_days)} of "
                 f"{(END_DATE - START_DATE).days + 1}")

    session = requests.Session()
    session.headers['User-Agent'] = 'stage4-climo/1.0 (research)'

    n_ok = n_miss = 0
    days_since_ckpt = 0
    t0 = time.time()

    for day_idx, day in enumerate(all_days):
        if _stop:
            break

        # 24 datetime objects for this day
        hours = [datetime(day.year, day.month, day.day, h) for h in range(NH)]

        # ── parallel download all 24 hours for this day ────────────
        day_results = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(fetch_hour, dt, session): dt for dt in hours}
            for fut in as_completed(futs):
                dt, content = fut.result()
                day_results[dt] = content

        # ── sequential decode + accumulate ────────────────────────
        for dt in hours:
            content = day_results[dt]
            if content is None:
                n_miss += 1
                continue
            data, lat_arr, lon_arr = decode_grib(content, dt)
            if data is None:
                n_miss += 1
                continue

            if lats is None:
                lats, lons = lat_arr, lon_arr
                log.info(f"  Grid confirmed ({NY}×{NX}), "
                         f"lat {lats.min():.1f}–{lats.max():.1f}, "
                         f"lon {lons.min():.1f}–{lons.max():.1f}")

            m, h = dt.month - 1, dt.hour
            count_tot[m, h] += 1

            # All thresholds in one vectorised pass
            valid = np.isfinite(data)   # (NY, NX)
            data3 = data[np.newaxis]    # (1, NY, NX)
            thr3  = np.array(THRESHOLDS, dtype=np.float32)[:, None, None]  # (NT,1,1)
            exceed = (valid[np.newaxis] & (data3 >= thr3)).astype(np.float32)  # (NT,NY,NX)
            count_exc[:, m, h] += exceed
            n_ok += 1

        done_dates.add(day.strftime('%Y-%m-%d'))
        days_since_ckpt += 1

        if days_since_ckpt >= SAVE_EVERY or _stop:
            checkpoint_save(count_exc, count_tot, lats, lons, done_dates)
            days_since_ckpt = 0

        if (day_idx + 1) % 30 == 0 or _stop or test_mode:
            elapsed = time.time() - t0
            rate    = (day_idx + 1) / elapsed * 3600   # days/hour
            eta_hr  = (len(all_days) - day_idx - 1) / rate if rate > 0 else 0
            log.info(f"  {day_idx+1}/{len(all_days)} days | "
                     f"ok={n_ok} miss={n_miss} | "
                     f"{rate:.0f} days/hr | ETA {eta_hr:.1f} hr")

    if test_mode:
        log.info("TEST COMPLETE — spot-checking accumulators:")
        log.info(f"  count_tot[0,12] (Jan 12Z): {count_tot[0,12]}")
        log.info(f"  count_exc[0,0,12].mean() (0.25mm Jan 12Z): "
                 f"{count_exc[0,0,12].mean():.4f}")
        log.info(f"  count_exc[4,0,12].mean() (5mm Jan 12Z):   "
                 f"{count_exc[4,0,12].mean():.6f}")
        return

    if not _stop:
        write_netcdf(count_exc, count_tot, lats, lons)
        log.info("STAGE4 CLIMATOLOGY COMPLETE")
    else:
        log.info("Interrupted — checkpoint saved, run again to resume.")


if __name__ == "__main__":
    main()
