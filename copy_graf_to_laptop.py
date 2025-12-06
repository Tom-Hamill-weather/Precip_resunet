"""

This will copy a GRAF precipitation forecast to the laptop from cray.

python copy_graf_to_laptop.py cyyyymmddhh clead 

for example

python copy_graf_to_laptop.py 2025112400 12

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
        # Fallback if password input cannot be hidden (e.g., on some systems)
        passphrase = input("Enter your passphrase (input may be echoed): ")
        return passphrase

class GRAFDataProcessor:
    def __init__(self, config_file):
        """Initialize processor by reading configuration."""
        self.params = {}
        self.dirs = {}
        self._load_config(config_file)
        
    # -------------------------------------------------------------
        
    def _load_config(self, config_file):
        """Reads the config.ini file."""
        print(f'INFO: Loading config from {config_file}')
        config = ConfigParser()
        config.read(config_file)

        if "DIRECTORIES" not in config or "PARAMETERS" not in config:
            raise ValueError(\
                "Config file missing DIRECTORIES or PARAMETERS sections")

        self.dirs = config["DIRECTORIES"]
        self.params = config["PARAMETERS"]
        
        # Parse specific needed values
        self.ndays_train = int(self.params.get("ndays_train", 60))

    # -------------------------------------------------------------

    def get_filenames(self, cyyyymmddhh, clead):
        
        """ Generates file paths based on date and 
            logic switch (April 2024)."""
        
        il = int(clead)
        # Date math
        cyyyymmdd = cyyyymmddhh[0:8]
        chh = cyyyymmddhh[8:10]
        
        cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
        cyyyymmdd_fcst = cyyyymmddhh_fcst[0:8]
        chh_fcst = cyyyymmddhh_fcst[8:10]

        # Logic for GRAF file naming convention change
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

# ---------------------------------------------------------------------

def scp_with_passphrase(user, host, source_file, destination_path, passphrase):
    # Construct the scp command
    scp_command = f'scp {source_file} {user}@{host}:{destination_path}'
    
    try:
        # Spawn the scp process
        child = pexpect.spawn(scp_command)
        
        # Expect the passphrase prompt or a host key prompt
        index = child.expect(['Enter passphrase for key.*:', '.*(yes/no)\\?'])
        
        # If prompted for host key, send 'yes'
        if index == 1:
            child.sendline('yes')
            child.expect('Enter passphrase for key.*:')
            
        # Send the passphrase
        child.sendline(passphrase)
        
        # Wait for the process to complete or exit (expect EOF)
        child.expect(pexpect.EOF)
        print(child.before.decode()) # Print output for debugging
        print("SCP transfer complete.")

    except pexpect.exceptions.TIMEOUT:
        print("Error: SCP command timed out.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

def main():
    
    if len(sys.argv) < 3:
        print("Usage: python copy_graf_to_laptop.py cyyyymmddhh clead")
        sys.exit(1)
    cyyyymmddhh = sys.argv[1]
    clead = sys.argv[2]
    
    print('============================================================')
    print(f'Running copy_graf_to_laptop. IC {cyyyymmddhh} Lead {clead}')
    print('============================================================')

    # --- 1. Setup Processor
    
    processor = GRAFDataProcessor('config_laptop.ini')
    input_dir, graf_file, full_path, local_dir = \
        processor.get_filenames(cyyyymmddhh, clead)
    cyyyymm = cyyyymmddhh[0:6]
    local_dir = os.path.join(local_dir, cyyyymmddhh[0:6])
    local_file = os.path.join(local_dir, graf_file)
    
    print ('making directory ', local_dir)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    print ('local_dir = ', local_dir)
    print ('input_dir = ', input_dir)
    print ('full_path = ', full_path)
    print ('graf_file = ', graf_file)
    print ('local_file = ', local_file)
    
    hostname = '10.66.63.22'
    username = 'thamill'
    password = getpass.getpass(\
        prompt=f'Enter passphrase for {username}@{hostname}: ')

    ssh_client = paramiko.SSHClient()
    ssh_client.load_system_host_keys()
    ssh_client.connect(hostname, username=username, password=password)

    graf_file = username + hostname + ':' + graf_file
    print (local_file)
    print (graf_file)
    
    try:
        with SCPClient(ssh_client.get_transport()) as scp:
            scp.get(full_path, local_path=local_file)
        print("File transfer complete using Paramiko.")
    except Exception as e:
        print(f"An error occurred during SCP transfer: {e}")
    finally:
        ssh_client.close()

# ================================================================================

if __name__ == "__main__":
    main()