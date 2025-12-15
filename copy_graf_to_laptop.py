"""
This will copy a GRAF precipitation forecast to the laptop from cray
for the entered date and lead time.  This is necessary for running
inference for this case on your laptop.

Usage:
$ python copy_graf_to_laptop.py cyyyymmddhh clead 

The directory where data are stashed is spec'd in config_laptop.ini
"""

import os, sys
import pexpect
import paramiko
import getpass
from scp import SCPClient
from configparser import ConfigParser
from dateutils import dateshift

# ------------------------------------------------------------------

def get_passphrase():
    """
    Prompts the user for a passphrase without echoing the input.
    Returns the entered passphrase as a string.
    """
    try:
        passphrase = getpass.getpass("Enter your passphrase: ")
        return passphrase
    except getpass.GetPassWarning as e:
        print(f"Warning: {e}")
        passphrase = input("Enter your passphrase (input may be echoed): ")
        return passphrase

class GRAFDataProcessor:
    def __init__(self, config_file):
        """Initialize processor by reading configuration."""
        self.params = {}
        self.dirs = {}
        self._load_config(config_file)
        
    def _load_config(self, config_file):
        """Reads the config.ini file."""
        # Only print loading info once per run to keep output clean
        # print(f'INFO: Loading config from {config_file}') 
        config = ConfigParser()
        config.read(config_file)

        if "DIRECTORIES" not in config or "PARAMETERS" not in config:
            raise ValueError(\
                "Config file missing DIRECTORIES or PARAMETERS sections")

        self.dirs = config["DIRECTORIES"]
        self.params = config["PARAMETERS"]
        self.ndays_train = int(self.params.get("ndays_train", 60))

    def get_filenames(self, cyyyymmddhh, clead):
        """ Generates file paths based on date and logic switch."""
        il = int(clead)
        cyyyymmdd = cyyyymmddhh[0:8]
        chh = cyyyymmddhh[8:10]
        
        cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
        cyyyymmdd_fcst = cyyyymmddhh_fcst[0:8]
        chh_fcst = cyyyymmddhh_fcst[8:10]

        if int(cyyyymmddhh) > 2024040512:
            base_dir = self.dirs["GRAFdatadir_conus_new"]
            prefix = 'grid.hdo-graf_conus.'
        else:
            base_dir = self.dirs["GRAFdatadir_conus_old"]
            prefix = 'grid.hdo-graflr_conus.'

        local_dir = self.dirs["GRAFdatadir_conus_laptop"]
        input_dir = os.path.join(base_dir, cyyyymmdd, chh)
        filename = (f"{prefix}{cyyyymmdd_fcst}T{chh_fcst}0000Z."
                    f"{cyyyymmdd}T{chh}0000Z.PT{clead}H.CONUS@4km.APCP.SFC.grb2")
        full_path = os.path.join(input_dir, filename)
        
        return input_dir, filename, full_path, local_dir

# -----------------------------------------------------------------------

def run_download(cyyyymmddhh, clead, password=None):
    """
    Main logic to download the GRAF file.
    Accepts password as an argument to avoid repeated prompts.
    """
    print('============================================================')
    print(f'Running copy_graf_to_laptop. IC {cyyyymmddhh} Lead {clead}')
    
    # --- 1. Setup Processor
    processor = GRAFDataProcessor('config_laptop.ini')
    input_dir, graf_file, full_path, local_dir = \
        processor.get_filenames(cyyyymmddhh, clead)
    
    # Adjust local directory structure
    local_dir = os.path.join(local_dir, cyyyymmddhh[0:6])
    local_file = os.path.join(local_dir, graf_file)
    
    if not os.path.exists(local_dir):
        print ('making directory ', local_dir)
        os.makedirs(local_dir)
    
    # print ('local_dir = ', local_dir)
    # print ('input_dir = ', input_dir)
    print (f'Downloading: {graf_file} ...')
    
    hostname = '10.66.63.22'
    username = 'thamill'
    
    # --- 2. Handle Password Logic
    # If password was not passed in by the control script, ask for it now.
    if password is None:
        password = getpass.getpass(\
            prompt=f'Enter passphrase for {username}@{hostname}: ')

    # --- 3. Connect and Download
    ssh_client = paramiko.SSHClient()
    ssh_client.load_system_host_keys()
    
    try:
        ssh_client.connect(hostname, username=username, password=password)
        
        # Determine if we are downloading or if logic requires specific remote path syntax
        # The original script prepended username/host to filename for pexpect, 
        # but paramiko/scp usually takes a standard path.
        # However, keeping your original path variable logic for safety:
        
        # print (local_file)
        # print (full_path) 
        
        with SCPClient(ssh_client.get_transport()) as scp:
            scp.get(full_path, local_path=local_file)
        print(f"Success: {graf_file} transfer complete.")
        
    except paramiko.AuthenticationException:
        print("Authentication failed. Please check your password.")
        sys.exit(1) # Exit if password is wrong so we don't loop 48 times with bad pw
    except Exception as e:
        print(f"An error occurred during SCP transfer: {e}")
    finally:
        ssh_client.close()

# ================================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python copy_graf_to_laptop.py cyyyymmddhh clead")
        sys.exit(1)
    
    # Standard CLI execution
    c_date = sys.argv[1]
    c_lead = sys.argv[2]
    run_download(c_date, c_lead)