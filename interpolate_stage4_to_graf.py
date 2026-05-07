#!/usr/bin/env python3
"""
interpolate_stage4_to_graf.py

One-time preprocessing: bilinearly interpolate the Stage IV climatological
exceedance probabilities from the native polar-stereographic grid onto the
GRAF 4-km Lambert-conformal grid, and write the result to a netCDF file.

The GRAF lat/lon grid is read from any existing gamma-mixture probability
file in the probs directory.

Output: /data/resnet_data/stage4_climo_on_graf.nc

Runtime: ~10-20 minutes (one Delaunay triangulation + 84 interpolation calls).
"""

import os, sys, glob, time
import numpy as np
from netCDF4 import Dataset
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

# ── paths ──────────────────────────────────────────────────────────────────

BASE_DIR   = '/data/resnet_data'
CLIMO_IN   = os.path.join(BASE_DIR, 'stage4_climo_2020_2024.nc')
CLIMO_OUT  = os.path.join(BASE_DIR, 'stage4_climo_on_graf.nc')
PROBS_DIR  = os.path.join(BASE_DIR, 'probs')

# ── read GRAF grid from a sample probability file ──────────────────────────

prob_files = sorted(glob.glob(os.path.join(PROBS_DIR, '*_probs_gamma_mixture.nc')))
if not prob_files:
    print('ERROR: no gamma_mixture probability files found in', PROBS_DIR)
    sys.exit(1)

sample_file = prob_files[0]
print(f'Reading GRAF grid from: {sample_file}')
nc = Dataset(sample_file, 'r')
graf_lats = np.asarray(nc.variables['lat'][:, :], dtype=np.float64)
graf_lons = np.asarray(nc.variables['lon'][:, :], dtype=np.float64)
nc.close()
NY_G, NX_G = graf_lats.shape
print(f'  GRAF grid: {NY_G} x {NX_G}  '
      f'lat [{graf_lats.min():.2f}, {graf_lats.max():.2f}]  '
      f'lon [{graf_lons.min():.2f}, {graf_lons.max():.2f}]')

# ── read Stage IV climatology ──────────────────────────────────────────────

print(f'\nReading Stage IV climatology: {CLIMO_IN}')
nc = Dataset(CLIMO_IN, 'r')
climo_prob     = np.asarray(nc.variables['climo_prob'][:],   dtype=np.float32)
thresholds     = np.asarray(nc.variables['threshold'][:],    dtype=np.float32)
months         = np.asarray(nc.variables['month'][:],        dtype=np.int32)
hours          = np.asarray(nc.variables['hour'][:],         dtype=np.int32)
s4_lats        = np.asarray(nc.variables['lat'][:, :],       dtype=np.float64)
s4_lons        = np.asarray(nc.variables['lon'][:, :],       dtype=np.float64)
nc.close()
NT, NM, NH, NY_S4, NX_S4 = climo_prob.shape
print(f'  Stage IV grid: {NY_S4} x {NX_S4},  {NT} thresholds, {NM} months, {NH} hours')

# ── build Delaunay triangulation (once) ────────────────────────────────────

print('\nBuilding Delaunay triangulation over Stage IV grid...')
t0 = time.time()
s4_pts   = np.column_stack([s4_lats.ravel(),   s4_lons.ravel()])
graf_pts = np.column_stack([graf_lats.ravel(),  graf_lons.ravel()])
tri = Delaunay(s4_pts)
print(f'  Done in {time.time()-t0:.1f}s')

# ── interpolate all (threshold, month) slices, stacking hours ─────────────

print(f'\nInterpolating {NT*NM} (threshold x month) slices '
      f'({NT*NM*NH} total) to GRAF grid...')
climo_on_graf = np.full((NT, NM, NH, NY_G, NX_G), np.nan, dtype=np.float32)

t0 = time.time()
for it in range(NT):
    for im in range(NM):
        # Stack all 24 hours into one call: values shape (n_s4_pts, NH)
        data_stack = np.stack([
            climo_prob[it, im, ih].ravel() for ih in range(NH)
        ], axis=1)
        interp_fn = LinearNDInterpolator(tri, data_stack)
        result = interp_fn(graf_pts)           # (n_graf_pts, NH)
        climo_on_graf[it, im, :, :, :] = result.reshape(NH, NY_G, NX_G)

        elapsed = time.time() - t0
        done    = it * NM + im + 1
        rate    = done / elapsed
        eta     = (NT * NM - done) / rate if rate > 0 else 0
        print(f'  thresh={thresholds[it]:.2f}mm  month={months[im]:2d}  '
              f'[{done}/{NT*NM}]  elapsed={elapsed:.0f}s  ETA={eta:.0f}s',
              end='\r', flush=True)

print(f'\n  Interpolation complete in {time.time()-t0:.1f}s')

# ── write output netCDF ────────────────────────────────────────────────────

print(f'\nWriting: {CLIMO_OUT}')
ds = Dataset(CLIMO_OUT, 'w')
ds.createDimension('threshold', NT)
ds.createDimension('month',     NM)
ds.createDimension('hour',      NH)
ds.createDimension('y',         NY_G)
ds.createDimension('x',         NX_G)
kw = dict(zlib=True, complevel=5)

v = ds.createVariable('climo_prob', 'f4',
                       ('threshold', 'month', 'hour', 'y', 'x'), **kw)
v[:] = climo_on_graf
v.long_name = '1-h precipitation exceedance climatological probability on GRAF grid'
v.units     = '1'
v.comment   = ('Bilinearly interpolated from Stage IV polar-stereo grid '
               '(stage4_climo_2020_2024.nc) to GRAF 4-km Lambert-conformal grid')

vt = ds.createVariable('threshold', 'f4', ('threshold',))
vt[:] = thresholds;  vt.units = 'mm';  vt.long_name = 'Exceedance threshold'

vm = ds.createVariable('month', 'i4', ('month',))
vm[:] = months;  vm.long_name = 'Month of year (1=Jan ... 12=Dec)'

vh = ds.createVariable('hour', 'i4', ('hour',))
vh[:] = hours;  vh.long_name = 'UTC hour of 1-h accumulation valid time'

vla = ds.createVariable('lat', 'f4', ('y', 'x'), **kw)
vla[:] = graf_lats.astype(np.float32);  vla.units = 'degrees_north'

vlo = ds.createVariable('lon', 'f4', ('y', 'x'), **kw)
vlo[:] = graf_lons.astype(np.float32);  vlo.units = 'degrees_east'

ds.source        = ('Stage IV climatology interpolated to GRAF grid; '
                    'source: stage4_climo_2020_2024.nc')
ds.graf_sample   = sample_file
ds.close()
print('Done.')
