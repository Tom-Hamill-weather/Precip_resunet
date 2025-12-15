"""
mrms_stats.py

Usage: $ python mrms_stats.py YYYYMMDDHH_begin YYYYMMDDHH_end

(meant to be run on the Cray)

will generate statistics for MRMS hourly precip for every
hour between the dates. This was handy both as a check to 
make sure that all the MRMS data existed before training
began and to check to make sure that the statistics
don't look alarming.

Tom Hamill
Dec 2025

"""


import sys
import os
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta
import warnings

# Suppress warnings about MaskedArray partitioning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) 
    # Suppress mean of empty slice warnings

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
    print(f"{'Date':<12} | {'Median':<8} | {'Mean':<8} | {'95th %':<8} | {'99th %':<8} | {'Max':<8}")
    print("-" * 70)

def process_mrms_files(start_date, end_date):
    # Base directory pattern
    base_dir_template = "/storage1/home/thamill/MRMS/{yyyymm}"
    filename_template = "MRMS_1h_pamt_and_data_qual_{yyyymmddhh}.nc"

    dates = get_date_range(start_date, end_date)

    print_header()

    line_count = 0

    for dt in dates:
        # --- Header Repetition Logic ---
        if line_count > 0 and line_count % 24 == 0:
             print_header()

        yyyymm = dt.strftime("%Y%m")
        yyyymmddhh = dt.strftime("%Y%m%d%H")
        
        # Construct path
        file_dir = base_dir_template.format(yyyymm=yyyymm)
        file_name = filename_template.format(yyyymmddhh=yyyymmddhh)
        full_path = os.path.join(file_dir, file_name)

        if not os.path.exists(full_path):
            print(f"{yyyymmddhh:<12} | File not found")
            line_count += 1
            continue

        try:
            with Dataset(full_path, 'r') as nc:
                if 'precipitation' not in nc.variables or 'data_quality' not in nc.variables:
                    print(f"{yyyymmddhh:<12} | Missing required vars in NetCDF")
                    line_count += 1
                    continue

                # Read data as standard numpy arrays
                precip = np.array(nc.variables['precipitation'][:,:])
                quality = np.array(nc.variables['data_quality'][:,:])

                # --- Create Mask ---
                # 1. Quality must be > 0.1
                # 2. Precipitation must be < 10,000 (Filters out the 1e36 fill values)
                valid_mask = (quality > 0.1) & (precip < 10000.0)
                
                # Extract valid data and copy to memory to avoid read-only errors
                valid_precip = precip[valid_mask].copy()

                if valid_precip.size == 0:
                    print(f"{yyyymmddhh:<12} | No valid data (quality > 0.1)")
                    line_count += 1
                    continue

                # Calculate Statistics
                median_val = float(np.median(valid_precip))
                mean_val = float(np.mean(valid_precip))
                p95_val = float(np.percentile(valid_precip, 95))
                p99_val = float(np.percentile(valid_precip, 99))
                max_val = float(np.max(valid_precip))

                # Print with 2 decimal places
                print(f"{yyyymmddhh:<12} | {median_val:<8.2f} | {mean_val:<8.2f} | "
                      f"{p95_val:<8.2f} | {p99_val:<8.2f} | {max_val:<8.2f}")
                
                line_count += 1

        except Exception as e:
            print(f"{yyyymmddhh:<12} | Error reading file: {e}")
            line_count += 1

if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Command line usage
        s_date = sys.argv[1]
        e_date = sys.argv[2]
    else:
        # Interactive usage
        print("Enter date range in YYYYMMDDHH format.")
        s_date = input("Start Date: ")
        e_date = input("End Date:   ")

    print("\nProcessing MRMS Statistics...")
    process_mrms_files(s_date, e_date)


