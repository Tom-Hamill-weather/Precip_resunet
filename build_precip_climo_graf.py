#!/usr/bin/env python
"""build_precip_climo_graf.py — GRAF-grid port of HRRRcal's build_precip_climo.py.

Blend (same design as HRRRcal, Tom-authorized 2026-06-25 there; ported here
2026-09-23 because GRAF's own domain -- 7.2-63.3N, -141.5 to -39.9E, well
beyond CONUS -- needs a full-domain climatology, and PRISM alone is US-only):
  * CORE  — PRISM 800 m monthly ppt normals 1991-2020 (CONUS land; terrain-aware,
            internally seam-free).
  * LAND beyond CONUS — WorldClim 2.1 30s monthly precip, bias-corrected to
            PRISM (median ratio over the CONUS overlap), feathered across the
            PRISM edge. WorldClim's own land/ocean nodata mask doubles as the
            land mask here (GRAF's terrain file, unlike HRRR's static_3km.nc,
            has no explicit land variable).
  * OFFSHORE — ERA5 monthly total precip 1991-2020, hard land/sea boundary,
            no feathering across the coast (same rationale as HRRRcal: land
            carries terrain-induced precip the ocean does not).

Caveat vs. the HRRR build: the ERA5 source file on this box
(/data/hrrr_cal/static/climo_src/era5/era5_tp_monthly_1991_2020.nc) only
covers 20-60N, -140 to -50E -- it was originally fetched for HRRR's smaller
domain. GRAF's full grid extends further south (to 7.2N), further north (to
63.3N) and further east (to -39.9E). Ocean pixels outside the ERA5
rectangle fall back to nearest-valid-neighbor fill (step 4 below), same as
the original script's "rare grid corner" fallback -- just covering a bigger
area here. This only affects open ocean far from CONUS; MRMS-covered
training patches are entirely within the ERA5 rectangle, so this has no
effect on training. Re-run with a wider ERA5 pull later if full-domain
ocean fidelity ever matters.

Output: /data/resnet_data/static/precip_climo_graf.nc, var `precip_climo`
(12, 1308, 1524) on the GRAF grid, mm/month. Diagnostic plots to
/data/resnet_data/plots/climo_blend_graf/.

Run:  python build_precip_climo_graf.py
"""
import os
import numpy as np
import netCDF4
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.interpolate import RegularGridInterpolator

Image.MAX_IMAGE_PIXELS = None

GRAF_TERRAIN = '/home/thamill/resnet/GRAF_CONUS_terrain_info.nc'
CLIMO_SRC = '/data/hrrr_cal/static/climo_src'
OUT_NC = '/data/resnet_data/static/precip_climo_graf.nc'
PLOT_DIR = '/data/resnet_data/plots/climo_blend_graf'
DSCALE = 1.0 / 120.0           # 30 arc-sec, PRISM/WorldClim
DIM = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
EXP_MONTHS = [3, 6, 9, 12]


def log(msg):
    print(msg, flush=True)


def load_graf_grid():
    ds = netCDF4.Dataset(GRAF_TERRAIN)
    lat = np.array(ds['lats'][:], np.float64)
    lon = np.array(ds['lons'][:], np.float64)
    ds.close()
    return lat, lon


def sample_nn(arr, lon_ul, lat_ul, glon, glat):
    """Nearest-neighbour sample a regular lat/lon raster (UL-corner origin,
    pixel size DSCALE, north-up) onto the GRAF lon/lat arrays. Returns float32
    with NaN where the GRAF point falls outside the raster."""
    col = np.round((glon - lon_ul) / DSCALE - 0.5).astype(np.int64)
    row = np.round((lat_ul - glat) / DSCALE - 0.5).astype(np.int64)
    H, W = arr.shape
    ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    out = np.full(glon.shape, np.nan, np.float32)
    out[ok] = arr[row[ok], col[ok]].astype(np.float32)
    return out


def read_prism(month, glon, glat):
    p = f'{CLIMO_SRC}/prism/prism_ppt_us_30s_2020{month:02d}_avg_30y.tif'
    a = np.asarray(Image.open(p)).astype(np.float32)
    a[a < -100] = np.nan                       # nodata -9999
    out = sample_nn(a, -125.0208333333, 49.9375, glon, glat)
    return out


def read_worldclim(month, glon, glat):
    """Crop the global 30-arcsec WorldClim raster to a window covering the
    target grid's bbox (+1deg margin), rather than HRRR's hardcoded window,
    so this generalizes to GRAF's much larger domain."""
    p = f'{CLIMO_SRC}/worldclim/wc2.1_30s_prec_{month:02d}.tif'
    im = Image.open(p)
    W, H = im.size  # (43200, 21600)
    lon_min, lon_max = glon.min() - 1.0, glon.max() + 1.0
    lat_min, lat_max = glat.min() - 1.0, glat.max() + 1.0
    c0 = max(0, int((lon_min - (-180.0)) / DSCALE))
    c1 = min(W, int((lon_max - (-180.0)) / DSCALE) + 1)
    r0 = max(0, int((90.0 - lat_max) / DSCALE))
    r1 = min(H, int((90.0 - lat_min) / DSCALE) + 1)
    a = np.asarray(im.crop((c0, r0, c1, r1))).astype(np.float32)
    a[a < 0] = np.nan                          # nodata -32768 / ocean
    lon_ul = -180.0 + c0 * DSCALE
    lat_ul = 90.0 - r0 * DSCALE
    return sample_nn(a, lon_ul, lat_ul, glon, glat)


def read_era5_climo(month, glon, glat):
    ds = netCDF4.Dataset(f'{CLIMO_SRC}/era5/era5_tp_monthly_1991_2020.nc')
    lat = np.array(ds['latitude'][:], np.float64)
    lon = np.array(ds['longitude'][:], np.float64)
    times = netCDF4.num2date(ds['valid_time'][:], ds['valid_time'].units)
    tp = np.array(ds['tp'][:], np.float64)     # (T, lat, lon), m/day mean
    ds.close()
    sel = np.array([t.month == month for t in times])
    clim = tp[sel].mean(axis=0)                # m/day, climatological mean
    clim_mm = clim * 1000.0 * DIM[month - 1]   # -> mm/month
    if lat[0] > lat[-1]:
        lat = lat[::-1]; clim_mm = clim_mm[::-1, :]
    f = RegularGridInterpolator((lat, lon), clim_mm, bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([glat.ravel(), glon.ravel()])
    return f(pts).reshape(glon.shape).astype(np.float32)


def feather_weight(mask_keep, n):
    dist_out = distance_transform_edt(~mask_keep)
    w = np.clip(1.0 - dist_out / float(n), 0.0, 1.0)
    return w.astype(np.float32)


def blend_month(month, glon, glat):
    prism = read_prism(month, glon, glat)
    wc = read_worldclim(month, glon, glat)
    era5 = read_era5_climo(month, glon, glat)

    pv = np.isfinite(prism)
    wv = np.isfinite(wc)          # WorldClim's own land/ocean nodata mask

    # 1) bias-correct WorldClim to PRISM over the CONUS overlap (no border step)
    ov = pv & wv & (wc > 1.0)
    ratio = float(np.median(prism[ov] / wc[ov])) if ov.sum() > 1000 else 1.0
    wc_corr = wc * ratio

    # 2) land field = PRISM core, WorldClim(corrected) beyond it, feathered at edge
    land_field = np.where(pv, prism, np.nan).astype(np.float32)
    fill = wv & ~pv
    land_field[fill] = wc_corr[fill]
    both = pv & wv
    wpr = feather_weight(binary_erosion(pv, iterations=6), 8)
    band = both & (wpr < 1.0)
    land_field[band] = wpr[band] * prism[band] + (1 - wpr[band]) * wc_corr[band]

    landm = wv                    # no separate land var for GRAF; WorldClim IS the mask
    lf_valid = np.isfinite(land_field)

    # 3) offshore = ERA5 as-is, hard land/sea boundary, no coastal feathering
    field = np.where(landm & lf_valid, land_field, np.nan).astype(np.float32)
    sea = (~landm) & np.isfinite(era5)
    field[sea] = era5[sea]

    # 4) fill any residual NaN by nearest valid (grid corners on HRRR; here also
    #    covers open ocean outside the ERA5 source rectangle -- see module
    #    docstring caveat)
    if not np.all(np.isfinite(field)):
        idx = distance_transform_edt(~np.isfinite(field), return_distances=False,
                                     return_indices=True)
        field = field[tuple(idx)]
    field = np.clip(field, 0.0, None)
    return field, dict(prism_cov=float(pv.mean()), land_cov=float(landm.mean()),
                       wc_ratio=ratio, fmax=float(field.max()), fmean=float(field.mean()))


def write_nc(climo):
    os.makedirs(os.path.dirname(OUT_NC), exist_ok=True)
    ds = netCDF4.Dataset(OUT_NC, 'w')
    ds.createDimension('month', 12)
    ds.createDimension('y', climo.shape[1])
    ds.createDimension('x', climo.shape[2])
    v = ds.createVariable('precip_climo', 'f4', ('month', 'y', 'x'), zlib=True, complevel=4)
    v.units = 'mm/month'
    v.long_name = 'blended monthly precipitation climatology (PRISM+WorldClim+ERA5), GRAF grid'
    v.months = ' '.join(MONTH_NAMES)
    v.blend = ('PRISM 1991-2020 core; WorldClim 2.1 bias-corrected to PRISM for '
               'non-CONUS land (WorldClim nodata also used as the land/sea mask); '
               'ERA5 1991-2020 offshore (source rectangle 20-60N/-140,-50E; open '
               'ocean beyond that filled by nearest-valid-neighbor)')
    v[:] = climo
    ds.close()
    log(f'wrote {OUT_NC}')


def make_plots(climo, glat, glon):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    os.makedirs(PLOT_DIR, exist_ok=True)
    clevs = [0, 5, 10, 20, 40, 60, 80, 120, 160, 220, 300, 400]

    def basemap(ll):
        llon, llat, ulon, ulat = ll
        return Basemap(projection='cyl', resolution='i', llcrnrlon=llon,
                       llcrnrlat=llat, urcrnrlon=ulon, urcrnrlat=ulat)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    full = (glon.min(), glat.min(), glon.max(), glat.max())
    for ax, mo in zip(axes.ravel(), EXP_MONTHS):
        m = basemap(full); m.ax = ax
        x, y = m(glon, glat)
        cs = m.contourf(x, y, climo[mo - 1], clevs, cmap='YlGnBu', extend='max')
        m.drawcoastlines(linewidth=0.5, color='0.3'); m.drawcountries(linewidth=0.5, color='0.3')
        ax.set_title(f'{MONTH_NAMES[mo-1]} precip climatology (mm/month)')
        fig.colorbar(cs, ax=ax, shrink=0.8, ticks=clevs, format='%g')
    fig.suptitle('GRAF-grid blended monthly precip climatology — PRISM core + WorldClim land + ERA5 offshore', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fa = f'{PLOT_DIR}/precip_climo_graf_4months_fulldomain.png'
    fig.savefig(fa, dpi=110); plt.close(fig)
    log(f'wrote {fa}')

    zooms = [('US-Canada border (N. Plains)', (-112, 45, -95, 51)),
             ('US-Mexico border (TX/AZ)',     (-115, 28, -97, 35)),
             ('West coast (CA/OR/WA)',        (-128, 35, -118, 49)),
             ('South of PRISM coverage (Mexico/C. America)', (-105, 8, -85, 25))]
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    dec = climo[11]
    for ax, (ttl, ll) in zip(axes, zooms):
        m = basemap(ll); m.ax = ax
        x, y = m(glon, glat)
        cs = m.pcolormesh(x, y, dec, cmap='YlGnBu', vmin=0, vmax=400, shading='auto')
        m.drawcoastlines(linewidth=0.7, color='red'); m.drawcountries(linewidth=0.7, color='red')
        ax.set_title(f'Dec — {ttl}')
        fig.colorbar(cs, ax=ax, shrink=0.8)
    fig.suptitle('Seam check (red = borders/coast); hard land/sea boundary at coast is intentional.', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fb = f'{PLOT_DIR}/precip_climo_graf_seamcheck_dec.png'
    fig.savefig(fb, dpi=120); plt.close(fig)
    log(f'wrote {fb}')


def main():
    glat, glon = load_graf_grid()
    log(f'GRAF grid {glat.shape}, lat [{glat.min():.2f},{glat.max():.2f}] '
        f'lon [{glon.min():.2f},{glon.max():.2f}]')
    climo = np.empty((12, glat.shape[0], glat.shape[1]), np.float32)
    for mo in range(1, 13):
        field, st = blend_month(mo, glon, glat)
        climo[mo - 1] = field
        log(f'{MONTH_NAMES[mo-1]}: prism_cov={st["prism_cov"]:.3f} land_cov={st["land_cov"]:.3f} '
            f'wc_ratio={st["wc_ratio"]:.3f} mean={st["fmean"]:.1f} max={st["fmax"]:.0f} mm')
    write_nc(climo)
    try:
        make_plots(climo, glat, glon)
    except Exception as e:
        log(f'WARNING: diagnostic plots failed ({e}); climatology NetCDF was still written.')
    log('BUILD PRECIP CLIMO GRAF DONE')


if __name__ == '__main__':
    main()
