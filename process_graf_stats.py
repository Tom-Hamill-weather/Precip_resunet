"""
process_graf_stats.py

Usage (run on Cray)

$ python process_graf_stats.py YYYYMMDDHH_begin YYYYMMDDHH_end LEAD
    where YYYYMMDDHH_begin YYYYMMDDHH_end  define the beginnning
    and ending dates, and LEAD is the lead time in hours.

This will examine data every 12 hours, printing statistics of
GRAF precipitation, a useful predecessor to the patch extraction 
routine.   When running this, notice that the domain-average 
precipitation varies significantly from day to day.  As we want
wet samples in general, this led in the patch extraction routine
to coding up using more samples on wet days than dry days.

Tom Hamill, Dec 2025

"""

import sys
import os
import numpy as np
import pygrib
from datetime import datetime, timedelta
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def get_date_range(start_str, end_str):
    """
    Generates a list of datetime objects between start and end (inclusive)
    at 1-hour intervals.
    """
    start_fmt = "%Y%m%d%H"
    try:
        start_dt = datetime.strptime(start_str, start_fmt)
        end_dt = datetime.strptime(end_str, start_fmt)
    except ValueError:
        print("Error: Dates must be in YYYYMMDDHH format.")
        sys.exit(1)

    delta_hours = int((end_dt - start_dt).total_seconds() // 3600)
    return [start_dt + timedelta(hours=i) for i in range(delta_hours + 1)]

def print_header():
    print("-" * 70)
    print(f"{'Date (Init)':<12} | {'Median':<8} | {'Mean':<8} | {'95th %':<8} | {'99th %':<8} | {'Max':<8}")
    print("-" * 70)

def process_graf_stats(start_date, end_date, lead_time):
    # Base directory pattern provided by user
    base_dir_template = "/storage2/library/archive/grid/hdo-graf_conus/{yyyymmdd}/{hh}"
    
    dates = get_date_range(start_date, end_date)
    lead = int(lead_time)

    print_header()
    line_count = 0

    for dt in dates:
        # Filter: Only process 00 and 12 UTC cycles
        if dt.hour not in [0, 12]:
            continue

        # Header Repetition
        if line_count > 0 and line_count % 24 == 0:
             print_header()

        # --- Date String Construction ---
        # Initialization strings
        init_yyyymmddhh = dt.strftime("%Y%m%d%H")
        init_yyyymmdd = dt.strftime("%Y%m%d")
        init_hh = dt.strftime("%H")
        
        # Valid time strings (Init + Lead)
        valid_dt = dt + timedelta(hours=lead)
        valid_yyyymmdd = valid_dt.strftime("%Y%m%d")
        valid_hh = valid_dt.strftime("%H")

        # --- Filename Logic ---
        # 1. Prefix logic based on date (April 5, 2024 switch)
        if int(init_yyyymmddhh) > 2024040512:
            prefix = 'grid.hdo-graf_conus.'
        else:
            prefix = 'grid.hdo-graflr_conus.'

        # 2. Construct specific filename format
        # FIX: Changed {lead:02d} to {lead} to prevent zero-padding for < 10h (e.g. PT3H not PT03H)
        filename = (f"{prefix}{valid_yyyymmdd}T{valid_hh}0000Z."
                    f"{init_yyyymmdd}T{init_hh}0000Z.PT{lead}H.CONUS@4km.APCP.SFC.grb2")

        # Construct full path
        file_dir = base_dir_template.format(yyyymmdd=init_yyyymmdd, hh=init_hh)
        full_path = os.path.join(file_dir, filename)

        # --- Processing ---
        if not os.path.exists(full_path):
            print(f"{init_yyyymmddhh:<12} | {'******':<8} | {'******':<8} | "
                  f"{'******':<8} | {'******':<8} | {'******':<8}")
            line_count += 1
            continue

        try:
            with pygrib.open(full_path) as grb_file:
                # Select message by lead time (endStep)
                try:
                    grb_msg = grb_file.select(endStep=lead)[0]
                except (ValueError, IndexError):
                    print(f"{init_yyyymmddhh:<12} | {'******':<8} | {'******':<8} | "
                          f"{'******':<8} | {'******':<8} | {'******':<8} (Lead not found)")
                    line_count += 1
                    continue

                # Read data
                lats, lons = grb_msg.latlons()
                precip = grb_msg.values

                # --- Bounding Box Filter ---
                # Box: -125 to -65 Longitude, 20 to 53 Latitude
                # Normalize lons if GRAF uses 0-360 (convert >180 to negative)
                lons_norm = np.where(lons > 180, lons - 360, lons)

                mask = (lons_norm >= -125) & (lons_norm <= -65) & \
                       (lats >= 20) & (lats <= 53)
                
                valid_precip = precip[mask]

                if valid_precip.size == 0:
                    print(f"{init_yyyymmddhh:<12} | No data in bbox")
                    line_count += 1
                    continue

                # --- Statistics ---
                median_val = float(np.median(valid_precip))
                mean_val = float(np.mean(valid_precip))
                p95_val = float(np.percentile(valid_precip, 95))
                p99_val = float(np.percentile(valid_precip, 99))
                max_val = float(np.max(valid_precip))

                print(f"{init_yyyymmddhh:<12} | {median_val:<8.2f} | {mean_val:<8.2f} | "
                      f"{p95_val:<8.2f} | {p99_val:<8.2f} | {max_val:<8.2f}")
                
                line_count += 1

        except Exception as e:
            # General error handling
            print(f"{init_yyyymmddhh:<12} | Error: {e}")
            line_count += 1

if __name__ == "__main__":
    if len(sys.argv) == 4:
        # Command line usage
        s_date = sys.argv[1]
        e_date = sys.argv[2]
        l_time = sys.argv[3]
    else:
        # Interactive usage
        print("GRAF Forecast Statistics (Box: -125 to -65 Lon, 20 to 53 Lat)")
        s_date = input("Start Date (YYYYMMDDHH): ")
        e_date = input("End Date   (YYYYMMDDHH): ")
        l_time = input("Lead Time       (hours): ")

    print(f"\nProcessing GRAF Statistics (Lead: {l_time}h)...")
    process_graf_stats(s_date, e_date, l_time)