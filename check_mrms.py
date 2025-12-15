"""
python check_mrms.py

prompts you for beginning and end dates, and then examines
and prints whether the MRMS files are complete for each day
between those dates.

Tom Hamill, Dec 2025
"""
import os
import sys
from datetime import datetime, timedelta

def check_mrms_files():
    # --- Configuration ---
    base_dir = "/storage1/home/thamill/MRMS"
    
    # --- Step A: Ask user for start and end dates ---
    print("Please enter dates in YYYYMMDDHH format (e.g., 2024101102)")
    start_input = input("Enter Start Date: ").strip()
    end_input = input("Enter End Date:   ").strip()

    try:
        # Parse inputs into datetime objects
        start_dt = datetime.strptime(start_input, "%Y%m%d%H")
        end_dt = datetime.strptime(end_input, "%Y%m%d%H")
    except ValueError:
        print("\nError: Invalid format. Please use YYYYMMDDHH.")
        sys.exit(1)

    if start_dt > end_dt:
        print("\nError: Start date cannot be after end date.")
        sys.exit(1)

    print(f"\nScanning directories in {base_dir}...")
    print("-" * 40)
    print(f"{'Date':<15} | {'Files Found':<15} | {'Status'}")
    print("-" * 40)

    # --- Step B & C: Loop over every day and check 24 hours ---
    
    # We create a pointer starting at the beginning of the start day (00:00)
    # and iterate day by day until we pass the end date.
    
    # Normalize current_day to midnight of the start date to ensure we check the full day
    current_day = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end_day_normalized = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    while current_day <= end_day_normalized:
        valid_files_count = 0
        hours_in_day = 24
        
        # Check every hour (00 through 23) for this specific day
        for hour in range(hours_in_day):
            # Create the timestamp for this specific hour
            check_time = current_day + timedelta(hours=hour)
            
            # Extract year, month, day, hour strings for path building
            yyyy = check_time.strftime("%Y")
            mm = check_time.strftime("%m")
            yyyymmddhh = check_time.strftime("%Y%m%d%H")
            
            # Construct the directory path: /storage1/home/thamill/MRMS/YYYYMM
            dir_path = os.path.join(base_dir, f"{yyyy}{mm}")
            
            # Construct the filename: MRMS_1h_pamt_and_data_qual_YYYYMMDDHH.nc
            filename = f"MRMS_1h_pamt_and_data_qual_{yyyymmddhh}.nc"
            full_path = os.path.join(dir_path, filename)

            # Check if file exists
            if os.path.exists(full_path):
                valid_files_count += 1
        
        # --- Step D: Print the number of files that exist for the day ---
        date_str = current_day.strftime("%Y-%m-%d")
        
        # Visual status indicator
        if valid_files_count == 24:
            status = "COMPLETE"
        elif valid_files_count == 0:
            status = "MISSING ALL"
        else:
            status = "PARTIAL"

        print(f"{date_str:<15} | {valid_files_count:>2}/24        | {status}")

        # Move to the next day
        current_day += timedelta(days=1)

    print("-" * 40)
    print("Scan complete.")

if __name__ == "__main__":
    check_mrms_files()