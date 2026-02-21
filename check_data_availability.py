"""check_data_availability.py

Usage:
    python check_data_availability.py begin_date end_date lead1 [lead2 ...]

Arguments:
    begin_date  : Start of date range (YYYYMMDDHH, init time)
    end_date    : End of date range   (YYYYMMDDHH, init time)
    lead1 ...   : One or more forecast lead times in hours

Purpose:
    For each 6-hourly init time in [begin_date, end_date] and each lead time,
    checks whether the expected GRAF grib2, MRMS netCDF, and GFS netCDF files
    exist on disk.  Reports individual missing files and contiguous blocks of
    missing init times.

Example:
    python check_data_availability.py 2023110100 2023113018 12 18 24
"""

import os
import sys
from configparser import ConfigParser
from dateutils import daterange, dateshift

# ----------------------------------------------------------------

def detect_config():
    """Select the appropriate config file based on the runtime environment."""
    if os.path.exists('/data2/resnet_data'):
        return 'config_aws.ini'
    elif os.path.exists('/storage2/library/archive/grid'):
        return 'config_hdo.ini'
    else:
        return 'config_laptop.ini'

# ----------------------------------------------------------------

def graf_path(dirs, params, cyyyymmddhh, clead):
    """Returns expected GRAF grib2 file path (mirrors GRAFDataProcessor.get_filenames)."""
    il = int(clead)
    cyyyymmdd = cyyyymmddhh[0:8]
    chh       = cyyyymmddhh[8:10]

    cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
    cyyyymmdd_fcst   = cyyyymmddhh_fcst[0:8]
    chh_fcst         = cyyyymmddhh_fcst[8:10]

    transition = params.get('GRAF_transition_date', '2024040512')
    if int(cyyyymmddhh) > int(transition):
        base_dir = dirs['GRAFdatadir_conus_new']
        prefix   = 'grid.hdo-graf_conus.'
    else:
        base_dir = dirs['GRAFdatadir_conus_old']
        prefix   = 'grid.hdo-graflr_conus.'

    input_dir = os.path.join(base_dir, cyyyymmdd, chh)
    filename  = (f"{prefix}{cyyyymmdd_fcst}T{chh_fcst}0000Z."
                 f"{cyyyymmdd}T{chh}0000Z.PT{clead}H.CONUS@4km.APCP.SFC.grb2")
    return os.path.join(input_dir, filename)


def mrms_path(dirs, cyyyymmddhh_valid):
    """Returns expected MRMS netCDF file path (mirrors GRAFDataProcessor.read_mrms)."""
    cyyyymm  = cyyyymmddhh_valid[0:6] + '/'
    filename = f'MRMS_1h_pamt_and_data_qual_{cyyyymmddhh_valid}.nc'
    return os.path.join(dirs['mrms_data_directory'], cyyyymm, filename)


def gfs_path(dirs, cyyyymmddhh):
    """Returns expected GFS netCDF file path (mirrors GRAFDataProcessor.read_gfs)."""
    gfs_dir  = dirs.get('gfs_data_directory',
                        '/storage1/home/thamill/resnet/resnet_data/gfs')
    return os.path.join(gfs_dir, f'gfs_subset_{cyyyymmddhh}.nc')

# ----------------------------------------------------------------

def report_block_summary(label, missing_dates, total):
    """Print count + contiguous blocks for a list of missing init-time dates."""
    n = len(missing_dates)
    present = total - n
    print(f'  {label}: {present}/{total} present  ({n} missing)')
    if n == 0:
        return

    # Group consecutive 6-hourly dates into blocks
    blocks = []
    start = end = missing_dates[0]
    for d in missing_dates[1:]:
        if d == dateshift(end, 6):
            end = d
        else:
            blocks.append((start, end))
            start = end = d
    blocks.append((start, end))

    for blk_start, blk_end in blocks:
        if blk_start == blk_end:
            print(f'    missing: {blk_start}')
        else:
            print(f'    missing block: {blk_start} -- {blk_end}')

# ----------------------------------------------------------------

def main():
    if len(sys.argv) < 4:
        print('Usage: python check_data_availability.py begin_date end_date lead1 [lead2 ...]')
        print('  e.g.: python check_data_availability.py 2023110100 2023113018 12 18 24')
        sys.exit(1)

    date_begin = sys.argv[1]
    date_end   = sys.argv[2]
    leads      = sys.argv[3:]

    config_file = detect_config()
    print(f'INFO: Using config: {config_file}')
    config = ConfigParser()
    config.read(config_file)
    dirs   = config['DIRECTORIES']
    params = config['PARAMETERS']

    dates = daterange(date_begin, date_end, 6)
    total = len(dates)
    print(f'INFO: Checking {total} init times  ({date_begin} -- {date_end})')
    print(f'INFO: Lead times: {", ".join(leads)}h')
    print()

    # Collect missing dates per variable/lead
    gfs_missing  = []
    graf_missing = {l: [] for l in leads}
    mrms_missing = {l: [] for l in leads}

    for date in dates:
        # GFS: one file per init time, independent of lead
        if not os.path.exists(gfs_path(dirs, date)):
            gfs_missing.append(date)

        for clead in leads:
            if not os.path.exists(graf_path(dirs, params, date, clead)):
                graf_missing[clead].append(date)

            valid = dateshift(date, int(clead))
            if not os.path.exists(mrms_path(dirs, valid)):
                mrms_missing[clead].append(date)

    # ---- Report ----

    print('=' * 50)
    print('GFS  (one file per init time)')
    print('=' * 50)
    report_block_summary('GFS', gfs_missing, total)
    print()

    for clead in leads:
        print('=' * 50)
        print(f'Lead {clead}h')
        print('=' * 50)
        report_block_summary(f'GRAF lead={clead}h', graf_missing[clead], total)
        report_block_summary(f'MRMS lead={clead}h', mrms_missing[clead], total)
        print()

    # ---- Summary table ----
    print('=' * 50)
    print('SUMMARY')
    print('=' * 50)
    ng = len(gfs_missing)
    print(f'  GFS  : {total-ng:5d}/{total} present  ({ng} missing)')
    for clead in leads:
        ng = len(graf_missing[clead])
        nm = len(mrms_missing[clead])
        print(f'  Lead {clead:>3s}h  GRAF: {total-ng:5d}/{total} present  ({ng} missing)'
              f'   MRMS: {total-nm:5d}/{total} present  ({nm} missing)')


if __name__ == '__main__':
    main()
