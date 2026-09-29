#!/usr/bin/env python
"""build_precip_climo_europe.py — European-grid precip climatology for the
gamma_mixture_season 'precip_climo' input channel (see graf_precip_climo.py
and build_precip_climo_graf.py for the CONUS/GRAF-grid original this is
ported from).

Europe's GRAF domain (33.85-61.63N, -19.77-28.90E) falls entirely outside
both PRISM (US-only) and the ERA5 source file cached on this box
(20-60N/-140,-50E — Western Hemisphere only), so unlike the CONUS build
there is no PRISM core and no ERA5 offshore term here: the whole field is
WorldClim 2.1 30s monthly precip (global, covers all of Europe including
its land), with residual NaN (open sea beyond WorldClim's own land/ocean
nodata mask — North Sea, Atlantic margin, Mediterranean/Baltic interior)
filled by nearest-valid-neighbor from the nearest coastline. This is a
coarser approximation than CONUS's land/ERA5-sea blend (no real offshore
precip source here), acceptable because MRMS-based training/verification
in this project is CONUS-only — the Europe run is transfer-learning
inference, not a training target.

Output: /data/resnet_data/static/precip_climo_europe.nc, var `precip_climo`
(12, 723, 666) on the GRAF Europe grid, mm/month.

Run: python build_precip_climo_europe.py
"""
import os
import numpy as np
import netCDF4
from PIL import Image
from scipy.ndimage import distance_transform_edt

Image.MAX_IMAGE_PIXELS = None

EUROPE_TERRAIN = '/data/resnet_data/terrain/GRAF_Europe_terrain_info.nc'
CLIMO_SRC = '/data/hrrr_cal/static/climo_src'
OUT_NC = '/data/resnet_data/static/precip_climo_europe.nc'
PLOT_DIR = '/data/resnet_data/plots/climo_blend_europe'
DSCALE = 1.0 / 120.0            # 30 arc-sec, WorldClim
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
EXP_MONTHS = [3, 6, 9, 12]


def log(msg):
    print(msg, flush=True)


def load_europe_grid():
    ds = netCDF4.Dataset(EUROPE_TERRAIN)
    lat = np.array(ds['lats'][:], np.float64)
    lon = np.array(ds['lons'][:], np.float64)
    ds.close()
    return lat, lon


def sample_nn(arr, lon_ul, lat_ul, glon, glat):
    """Nearest-neighbour sample a regular lat/lon raster (UL-corner origin,
    pixel size DSCALE, north-up) onto the target lon/lat arrays. Returns
    float32 with NaN where the target point falls outside the raster."""
    col = np.round((glon - lon_ul) / DSCALE - 0.5).astype(np.int64)
    row = np.round((lat_ul - glat) / DSCALE - 0.5).astype(np.int64)
    H, W = arr.shape
    ok = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    out = np.full(glon.shape, np.nan, np.float32)
    out[ok] = arr[row[ok], col[ok]].astype(np.float32)
    return out


def read_worldclim(month, glon, glat):
    """Crop the global 30-arcsec WorldClim raster to a window covering the
    target grid's bbox (+1deg margin) — same helper as
    build_precip_climo_graf.py.read_worldclim, grid-agnostic."""
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


def blend_month(month, glon, glat):
    wc = read_worldclim(month, glon, glat)
    field = np.clip(wc, 0.0, None).astype(np.float32)
    land_cov = float(np.isfinite(wc).mean())

    if not np.all(np.isfinite(field)):
        idx = distance_transform_edt(~np.isfinite(field), return_distances=False,
                                     return_indices=True)
        field = field[tuple(idx)]
    field = np.clip(field, 0.0, None)
    return field, dict(land_cov=land_cov, fmax=float(field.max()),
                       fmean=float(field.mean()))


def write_nc(climo):
    os.makedirs(os.path.dirname(OUT_NC), exist_ok=True)
    ds = netCDF4.Dataset(OUT_NC, 'w')
    ds.createDimension('month', 12)
    ds.createDimension('y', climo.shape[1])
    ds.createDimension('x', climo.shape[2])
    v = ds.createVariable('precip_climo', 'f4', ('month', 'y', 'x'), zlib=True, complevel=4)
    v.units = 'mm/month'
    v.long_name = 'WorldClim monthly precipitation climatology, GRAF Europe grid'
    v.months = ' '.join(MONTH_NAMES)
    v.blend = ('WorldClim 2.1 (global, no PRISM/ERA5 term available for this '
               'domain); residual open-sea NaN filled by nearest-valid-neighbor')
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    full = (glon.min(), glat.min(), glon.max(), glat.max())
    for ax, mo in zip(axes.ravel(), EXP_MONTHS):
        m = basemap(full); m.ax = ax
        x, y = m(glon, glat)
        cs = m.contourf(x, y, climo[mo - 1], clevs, cmap='YlGnBu', extend='max')
        m.drawcoastlines(linewidth=0.5, color='0.3'); m.drawcountries(linewidth=0.5, color='0.3')
        ax.set_title(f'{MONTH_NAMES[mo-1]} precip climatology (mm/month)')
        fig.colorbar(cs, ax=ax, shrink=0.8, ticks=clevs, format='%g')
    fig.suptitle('Europe GRAF-grid precip climatology — WorldClim only', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fa = f'{PLOT_DIR}/precip_climo_europe_4months_fulldomain.png'
    fig.savefig(fa, dpi=110); plt.close(fig)
    log(f'wrote {fa}')


def main():
    glat, glon = load_europe_grid()
    log(f'Europe grid {glat.shape}, lat [{glat.min():.2f},{glat.max():.2f}] '
        f'lon [{glon.min():.2f},{glon.max():.2f}]')
    climo = np.empty((12, glat.shape[0], glat.shape[1]), np.float32)
    for mo in range(1, 13):
        field, st = blend_month(mo, glon, glat)
        climo[mo - 1] = field
        log(f'{MONTH_NAMES[mo-1]}: land_cov={st["land_cov"]:.3f} '
            f'mean={st["fmean"]:.1f} max={st["fmax"]:.0f} mm')
    write_nc(climo)
    try:
        make_plots(climo, glat, glon)
    except Exception as e:
        log(f'WARNING: diagnostic plots failed ({e}); climatology NetCDF was still written.')
    log('BUILD PRECIP CLIMO EUROPE DONE')


if __name__ == '__main__':
    main()
