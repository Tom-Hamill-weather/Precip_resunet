#!/usr/bin/env python
"""
Convert pickle files to compressed NetCDF4 format.

Usage:
    python convert_pickle_to_netcdf.py input.cPick output.nc

This converts uncompressed pickle files to NetCDF4 with:
- zlib compression (level 4, good balance of speed/size)
- Chunking optimized for patch access
- Expected 3-5x size reduction

After conversion, verify correctness and delete old pickle files.
"""

import sys
import os
import _pickle as cPickle
import numpy as np
from netCDF4 import Dataset
from datetime import datetime

def convert_pickle_to_netcdf(pickle_file, netcdf_file):
    """Convert pickle patch file to compressed NetCDF4."""

    print(f"Reading {pickle_file}...")

    # Read pickle file
    with open(pickle_file, 'rb') as f:
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

    npatches, ny, nx = GRAF.shape
    print(f"  {npatches} patches of size {ny}×{nx}")

    # Create NetCDF4 file with compression
    print(f"Writing {netcdf_file}...")
    nc = Dataset(netcdf_file, 'w', format='NETCDF4')

    # Dimensions
    nc.createDimension('patch', npatches)
    nc.createDimension('y', ny)
    nc.createDimension('x', nx)
    nc.createDimension('time_str_len', 10)  # YYYYMMDDHH

    # Compression settings: level 4 is good balance
    comp = {'zlib': True, 'complevel': 4, 'shuffle': True}

    # Chunking: optimize for reading individual patches
    chunks = (1, ny, nx)  # One patch at a time

    # Create variables with compression
    print("  Creating compressed variables...")

    # Meteorological fields (compress well)
    nc_graf = nc.createVariable('GRAF', 'f4', ('patch', 'y', 'x'),
                                chunksizes=chunks, **comp)
    nc_mrms = nc.createVariable('MRMS', 'f4', ('patch', 'y', 'x'),
                                chunksizes=chunks, **comp)
    nc_mrms_qual = nc.createVariable('MRMS_qual', 'f4', ('patch', 'y', 'x'),
                                     chunksizes=chunks, **comp)

    # Terrain fields (compress VERY well - mostly smooth/repetitive)
    nc_terdiff = nc.createVariable('terrain_diff', 'f4', ('patch', 'y', 'x'),
                                   chunksizes=chunks, **comp)
    nc_dlon = nc.createVariable('dt_dlon', 'f4', ('patch', 'y', 'x'),
                               chunksizes=chunks, **comp)
    nc_dlat = nc.createVariable('dt_dlat', 'f4', ('patch', 'y', 'x'),
                               chunksizes=chunks, **comp)

    # GFS fields
    nc_pwat = nc.createVariable('GFS_pwat', 'f4', ('patch', 'y', 'x'),
                               chunksizes=chunks, **comp)
    nc_r = nc.createVariable('GFS_r', 'f4', ('patch', 'y', 'x'),
                            chunksizes=chunks, **comp)
    nc_cape = nc.createVariable('GFS_cape', 'f4', ('patch', 'y', 'x'),
                               chunksizes=chunks, **comp)

    # Time stamps as character arrays (compresses well)
    nc_init = nc.createVariable('init_times', 'S1', ('patch', 'time_str_len'))
    nc_valid = nc.createVariable('valid_times', 'S1', ('patch', 'time_str_len'))

    # Add attributes
    nc.description = f'Training patches converted from {os.path.basename(pickle_file)}'
    nc.history = f'Converted on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    nc.patch_size = f'{ny}x{nx}'

    nc_graf.units = 'mm'
    nc_graf.long_name = 'GRAF precipitation forecast'
    nc_mrms.units = 'mm'
    nc_mrms.long_name = 'MRMS precipitation analysis'
    nc_mrms_qual.long_name = 'MRMS data quality'
    nc_terdiff.units = 'm'
    nc_terdiff.long_name = 'Local terrain height difference'
    nc_pwat.units = 'kg m-2'
    nc_pwat.long_name = 'GFS precipitable water'
    nc_r.units = '%'
    nc_r.long_name = 'GFS column-average relative humidity'
    nc_cape.units = 'J kg-1'
    nc_cape.long_name = 'GFS CAPE'

    # Write data
    print("  Writing data arrays...")
    nc_graf[:] = GRAF
    nc_mrms[:] = MRMS
    nc_mrms_qual[:] = MRMS_qual
    nc_terdiff[:] = terrain_diff
    nc_dlon[:] = dt_dlon
    nc_dlat[:] = dt_dlat
    nc_pwat[:] = GFS_pwat
    nc_r[:] = GFS_r
    nc_cape[:] = GFS_cape

    # Convert time strings to character arrays
    print("  Writing timestamps...")
    for i, (init_time, valid_time) in enumerate(zip(init_times, valid_times)):
        nc_init[i] = list(init_time[:10])  # YYYYMMDDHH
        nc_valid[i] = list(valid_time[:10])

    # Note: We can recompute terdiff_x_GRAF = terrain_diff * GRAF, saving 18% space

    nc.close()

    # Report sizes
    old_size = os.path.getsize(pickle_file) / 1024**2
    new_size = os.path.getsize(netcdf_file) / 1024**2
    ratio = old_size / new_size

    print(f"\nConversion complete!")
    print(f"  Old size: {old_size:.1f} MB")
    print(f"  New size: {new_size:.1f} MB")
    print(f"  Compression ratio: {ratio:.2f}x")
    print(f"  Space saved: {old_size - new_size:.1f} MB ({100*(1-1/ratio):.1f}%)")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_pickle_to_netcdf.py input.cPick output.nc")
        sys.exit(1)

    pickle_file = sys.argv[1]
    netcdf_file = sys.argv[2]

    if not os.path.exists(pickle_file):
        print(f"ERROR: File not found: {pickle_file}")
        sys.exit(1)

    convert_pickle_to_netcdf(pickle_file, netcdf_file)
