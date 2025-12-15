"""
This controls a loop to copy a GRAF precipitation forecast 
to the laptop from cray for the entered date and lead time.

Usage:
$ python control_copy_graf_to_laptop.py cyyyymmddhh clead_begin clead_end

for example

$ python control_copy_graf_to_laptop.py 2025120300 1 48
$ python control_copy_graf_to_laptop.py 2025120812 1 48

"""

import sys
import getpass
# Import the function from your other script
from copy_graf_to_laptop import run_download

# ------------------------------------------------------------------

if len(sys.argv) < 4:
    print("Usage: python control_copy_graf_to_laptop.py cyyyymmddhh clead_begin clead_end")
    sys.exit(1)

cyyyymmddhh = sys.argv[1]
clead_begin  = sys.argv[2]
clead_end = sys.argv[3]

ileadb = int(clead_begin)
ileade = int(clead_end)
ileads = range(ileadb, ileade+1)

# Prompt for the password ONCE
print(f"Preparing to download files for {cyyyymmddhh}...")
user_pass = getpass.getpass(prompt="Enter passphrase (will be used for all files): ")

for ilead in ileads:
    clead = str(ilead)
    
    # Call the python function directly instead of using os.system
    # This keeps the memory state (and the password) available
    run_download(cyyyymmddhh, clead, password=user_pass)

print("\nAll requested files processed.")
