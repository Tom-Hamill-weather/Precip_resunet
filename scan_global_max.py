"""
scan_global_max.py

Usage:
$ python scan_global_max.py

Why?  In the training process, there's a normalization
    of feature data by max and min.  From experimentation
    if we let the max and min of the sample change from
    lead time to lead time, this can introduce numerical
    oddities and one sees artifacts at inference as well
    as inconsistent probabilities across time.  Better
    to set global normalization max/min consistent across
    all times.  This helps inform that.

Scans all GRAF_Unet_data_train_*.cPick* files.
1. Reports per-file Min/Max for all 5 features.
2. Reports Global Min/Max across the entire dataset.
3. Suggests normalization constants for 
        pytorch_train_resunet.py.
"""

import glob
import os
import sys
import _pickle as cPickle
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = '../resnet_data' 
FILE_PATTERN = "GRAF_Unet_data_train_*.cPick*"

# Feature labels matching the order in the pickle file
FEATURE_NAMES = [
    "0: GRAF Precip",
    "1: Interaction (GRAF*TDiff)",
    "2: Terrain Diff",
    "3: dlon (Slope)",
    "4: dlat (Slope)"
]

def scan_files():
    search_path = os.path.join(DATA_DIR, FILE_PATTERN)
    files = glob.glob(search_path)
    
    if not files:
        print(f"ERROR: No files found matching {search_path}")
        sys.exit(1)
    
    files.sort() # Sort by filename for consistent reading
    print(f"Found {len(files)} training files. Starting scan...\n")

    # Initialize global trackers
    global_maxs = [float('-inf')] * 5
    global_mins = [float('inf')] * 5

    # --- Header for Per-File Report ---
    print(f"{'Filename':<50} | {'Feature':<25} | {'Min':<10} | {'Max':<10}")
    print("-" * 105)

    for fpath in files:
        fname = os.path.basename(fpath)
        
        try:
            with open(fpath, 'rb') as f:
                # Load data in specific order
                graf = cPickle.load(f)          # 0
                _ = cPickle.load(f)             # mrms (skip)
                _ = cPickle.load(f)             # qual (skip)
                terdiff_graf = cPickle.load(f)  # 1
                diff = cPickle.load(f)          # 2
                dlon = cPickle.load(f)          # 3
                dlat = cPickle.load(f)          # 4
                
                arrays = [graf, terdiff_graf, diff, dlon, dlat]
                
                # Print file header (only once per file)
                print(f"{fname:<50} | {'---':<25} | {'---':<10} | {'---':<10}")

                for i, arr in enumerate(arrays):
                    # Local stats
                    l_min = float(np.min(arr))
                    l_max = float(np.max(arr))
                    
                    # Update Globals
                    if l_max > global_maxs[i]: global_maxs[i] = l_max
                    if l_min < global_mins[i]: global_mins[i] = l_min

                    # Print Row
                    # Empty filename column for feature rows to keep it clean
                    print(f"{'':<50} | {FEATURE_NAMES[i]:<25} | {l_min:<10.2f} | {l_max:<10.2f}")
                
                print("-" * 105) # Separator line between files

        except Exception as e:
            print(f"ERROR reading {fname}: {e}")
            print("-" * 105)

    # --- Global Summary ---
    print("\n" + "="*60)
    print("GLOBAL SUMMARY ACROSS ALL FILES")
    print("="*60)
    print(f"{'Feature':<30} | {'Global Min':<15} | {'Global Max':<15}")
    print("-" * 65)
    
    for i, name in enumerate(FEATURE_NAMES):
        print(f"{name:<30} | {global_mins[i]:<15.4f} | {global_maxs[i]:<15.4f}")
    
    print("-" * 65)
    
    # --- Recommendations ---
    # We round up slightly to ensure safety
    rec_inter = np.ceil(global_maxs[1])
    rec_tdiff = np.ceil(global_maxs[2])
    # For slopes (dlon/dlat), we usually take the max of 
    # absolute values or just the max positive
    # Assuming standard symmetric slopes, the max is usually sufficient.
    rec_slope = np.ceil(max(global_maxs[3], global_maxs[4]))

    print("\nRECOMMENDED CODE UPDATE FOR pytorch_train_resunet.py:")
    print("-" * 50)
    print(f"FORCED_MAX_INTERACTION = {rec_inter}")
    print(f"FORCED_MAX_TDIFF       = {rec_tdiff}")
    print(f"FORCED_MAX_SLOPE       = {rec_slope}")
    print("-" * 50)

if __name__ == "__main__":
    scan_files()