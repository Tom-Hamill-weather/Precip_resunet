"""
Utility functions for loading training data from pickle or NetCDF formats.
Automatically detects format and loads accordingly.
"""

import os
import _pickle as cPickle
import numpy as np
from netCDF4 import Dataset


def load_training_data(filepath):
    """
    Load training data from pickle or NetCDF file.

    Args:
        filepath: Path to .cPick or .nc file (can omit extension)

    Returns:
        Dictionary with keys: GRAF, MRMS, MRMS_qual, terdiff_x_GRAF,
        terrain_diff, dt_dlon, dt_dlat, init_times, valid_times,
        GFS_pwat, GFS_r, GFS_cape
    """

    # Auto-detect format
    if filepath.endswith('.cPick'):
        pickle_file = filepath
        netcdf_file = filepath.replace('.cPick', '.nc')
    elif filepath.endswith('.nc'):
        netcdf_file = filepath
        pickle_file = filepath.replace('.nc', '.cPick')
    else:
        # Try both extensions
        pickle_file = filepath + '.cPick' if not filepath.endswith('.cPick') else filepath
        netcdf_file = filepath + '.nc' if not filepath.endswith('.nc') else filepath

    # Prefer NetCDF if both exist (compressed, faster to load)
    if os.path.exists(netcdf_file):
        print(f"Loading NetCDF: {os.path.basename(netcdf_file)}")
        return load_netcdf(netcdf_file)
    elif os.path.exists(pickle_file):
        print(f"Loading pickle: {os.path.basename(pickle_file)}")
        return load_pickle(pickle_file)
    else:
        raise FileNotFoundError(f"Data file not found: {filepath} (.nc or .cPick)")


def load_pickle(filepath):
    """Load data from pickle file."""
    with open(filepath, 'rb') as f:
        GRAF = cPickle.load(f)
        MRMS = cPickle.load(f)
        MRMS_qual = cPickle.load(f)
        terdiff_x_GRAF = cPickle.load(f)
        terrain_diff = cPickle.load(f)
        dt_dlon = cPickle.load(f)
        dt_dlat = cPickle.load(f)
        init_times = cPickle.load(f)
        valid_times = cPickle.load(f)
        GFS_pwat = cPickle.load(f)
        GFS_r = cPickle.load(f)
        GFS_cape = cPickle.load(f)

    return {
        'GRAF': GRAF,
        'MRMS': MRMS,
        'MRMS_qual': MRMS_qual,
        'terdiff_x_GRAF': terdiff_x_GRAF,
        'terrain_diff': terrain_diff,
        'dt_dlon': dt_dlon,
        'dt_dlat': dt_dlat,
        'init_times': init_times,
        'valid_times': valid_times,
        'GFS_pwat': GFS_pwat,
        'GFS_r': GFS_r,
        'GFS_cape': GFS_cape
    }


def load_netcdf(filepath):
    """Load data from NetCDF file."""
    nc = Dataset(filepath, 'r')

    # Load arrays
    GRAF = nc.variables['GRAF'][:]
    MRMS = nc.variables['MRMS'][:]
    MRMS_qual = nc.variables['MRMS_qual'][:]
    terrain_diff = nc.variables['terrain_diff'][:]
    dt_dlon = nc.variables['dt_dlon'][:]
    dt_dlat = nc.variables['dt_dlat'][:]
    GFS_pwat = nc.variables['GFS_pwat'][:]
    GFS_r = nc.variables['GFS_r'][:]
    GFS_cape = nc.variables['GFS_cape'][:]

    # Reconstruct terdiff_x_GRAF (saves 18% space)
    terdiff_x_GRAF = terrain_diff * GRAF

    # Decode timestamps
    npatches = len(nc.dimensions['patch'])
    init_times = [''.join([c.decode() if isinstance(c, bytes) else c
                           for c in nc.variables['init_times'][i,:]])
                  for i in range(npatches)]
    valid_times = [''.join([c.decode() if isinstance(c, bytes) else c
                            for c in nc.variables['valid_times'][i,:]])
                   for i in range(npatches)]

    nc.close()

    return {
        'GRAF': GRAF,
        'MRMS': MRMS,
        'MRMS_qual': MRMS_qual,
        'terdiff_x_GRAF': terdiff_x_GRAF,
        'terrain_diff': terrain_diff,
        'dt_dlon': dt_dlon,
        'dt_dlat': dt_dlat,
        'init_times': init_times,
        'valid_times': valid_times,
        'GFS_pwat': GFS_pwat,
        'GFS_r': GFS_r,
        'GFS_cape': GFS_cape
    }


def get_data_path(base_path, date, lead_time, dataset_type='train'):
    """
    Get path to training data file, checking both NetCDF and pickle.

    Args:
        base_path: Base directory containing data files
        date: Date string (YYYYMMDDHH)
        lead_time: Lead time string (e.g., '12h')
        dataset_type: 'train', 'test', or 'predict'

    Returns:
        Full path to data file
    """
    basename = f'GRAF_Unet_data_{dataset_type}_{date}_{lead_time}'

    # Check NetCDF first (preferred)
    nc_path = os.path.join(base_path, f'{basename}.nc')
    if os.path.exists(nc_path):
        return nc_path

    # Fall back to pickle
    pickle_path = os.path.join(base_path, f'{basename}.cPick')
    if os.path.exists(pickle_path):
        return pickle_path

    raise FileNotFoundError(f"No data file found for {basename}")


# Example usage:
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_loader_utils.py <file.nc or file.cPick>")
        sys.exit(1)

    data = load_training_data(sys.argv[1])
    print(f"\nLoaded {len(data['GRAF'])} patches")
    print(f"Patch shape: {data['GRAF'].shape}")
    print(f"Data keys: {list(data.keys())}")
