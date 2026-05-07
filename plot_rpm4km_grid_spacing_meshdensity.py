"""
Compute and plot RPM4km grid spacing using the formula:
    dx(i) = len_disp / meshDensity(i)**0.25
where len_disp = 4000.0 m and meshDensity is read from meshDensity.nc.

Two-panel orthographic projection: North America and Europe.
Triangulates the original unstructured MPAS cell centres directly —
no intermediate binning or regridding needed.
"""

import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import warnings
warnings.filterwarnings('ignore')
from mpl_toolkits.basemap import Basemap

# ── 1.  Load meshDensity ───────────────────────────────────────────────────────
print("Loading meshDensity.nc ...")
ds = nc.Dataset('meshDensity.nc')
mesh_density = np.array(ds.variables['meshDensity'][:])
lat_rad      = np.array(ds.variables['latCell'][:])
lon_rad      = np.array(ds.variables['lonCell'][:])
ds.close()

lat_deg = np.degrees(lat_rad)
lon_deg = np.degrees(lon_rad)
lon_deg = np.where(lon_deg > 180, lon_deg - 360, lon_deg)

# ── 2.  Compute grid spacing (metres → km) ────────────────────────────────────
len_disp = 4000.0
dx_km    = (len_disp / mesh_density**0.25) / 1000.0

print(f"  Grid spacing range: {dx_km.min():.2f} – {dx_km.max():.2f} km")

# ── 3.  Plot ───────────────────────────────────────────────────────────────────
levels = [4.1, 7, 11, 14.9]

panels = [
    dict(name='(a) North America', lat_c=35.0, lon_c=-93.0, half_deg=65, extent_m=4_500_000),
    dict(name='(b) Europe',        lat_c=44.0, lon_c=  3.0, half_deg=40, extent_m=2_700_000),
]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
plt.subplots_adjust(wspace=0.02)

for idx, panel in enumerate(panels):
    lat_c    = panel['lat_c']
    lon_c    = panel['lon_c']
    half_deg = panel['half_deg']
    extent_m = panel['extent_m']
    ax       = axes[idx]

    # ── Basemap orthographic projection with zoom ─────────────────────────────
    m = Basemap(projection='ortho', lat_0=lat_c, lon_0=lon_c,
                resolution='l', ax=ax,
                llcrnrx=-extent_m, llcrnry=-extent_m,
                urcrnrx= extent_m, urcrnry= extent_m)

    m.drawmapboundary(fill_color='lightblue', zorder=0)
    m.fillcontinents(color='lightgrey', lake_color='lightblue', zorder=1)
    m.drawcoastlines(linewidth=0.7,  color='#333333', zorder=4)
    m.drawcountries(linewidth=0.5,   color='#555555', zorder=4)
    m.drawstates(linewidth=0.35,     color='#888888', zorder=4)
    m.drawparallels(range(-90,  91, 15), linewidth=0.4, color='grey',
                    alpha=0.6, linestyle='--', zorder=3)
    m.drawmeridians(range(-180, 181, 15), linewidth=0.4, color='grey',
                    alpha=0.6, linestyle='--', zorder=3)

    # ── Subset points to panel region (padding ensures no edge gaps) ──────────
    pad = 10   # degrees beyond half_deg
    lat_mask = ((lat_deg >= lat_c - half_deg - pad) &
                (lat_deg <= min(75.0, lat_c + half_deg + pad)))
    lon_mask = ((lon_deg >= lon_c - half_deg - pad) &
                (lon_deg <= lon_c + half_deg + pad))
    sel = lat_mask & lon_mask

    # ── Project to map coordinates; discard back-hemisphere points ────────────
    x, y = m(lon_deg[sel], lat_deg[sel])
    in_map = (x < 1e29) & (y < 1e29)
    x_p = x[in_map];  y_p = y[in_map];  z_p = dx_km[sel][in_map]

    # ── Triangulate and mask long-edge triangles (wrap / limb artifacts) ──────
    triang = mtri.Triangulation(x_p, y_p)
    vx = x_p[triang.triangles]
    vy = y_p[triang.triangles]
    edge_len = np.max([
        np.hypot(vx[:, 1] - vx[:, 0], vy[:, 1] - vy[:, 0]),
        np.hypot(vx[:, 2] - vx[:, 1], vy[:, 2] - vy[:, 1]),
        np.hypot(vx[:, 0] - vx[:, 2], vy[:, 0] - vy[:, 2]),
    ], axis=0)
    triang.set_mask(edge_len > 500_000)   # 500 km — far larger than any real edge

    # ── Tricontour ────────────────────────────────────────────────────────────
    cs = ax.tricontour(triang, z_p, levels=levels,
                       colors='black', linewidths=1.0, zorder=5)
    ax.clabel(cs, fmt='%g km', fontsize=13, inline=True, inline_spacing=5)
    for txt in cs.labelTexts:
        txt.set_clip_path(ax.patch)
        txt.set_clip_on(True)

    ax.set_title(panel['name'], fontsize=21, pad=10)

outfile = 'rpm4km_grid_spacing_meshdensity.png'
plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {outfile}")
