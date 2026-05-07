"""
plot_graf_and_pdf.py -- Two-panel figure for LaTeX: zoomed GRAF precipitation map
and fitted zero-inflated gamma mixture PDF at a chosen grid point.

Usage:
    python plot_graf_and_pdf.py <YYYYMMDDHH> <lead_hours> <lat> <lon>

Example:
    python plot_graf_and_pdf.py 2025030100 12 39.5 -105.0
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gamma as gamma_dist
from netCDF4 import Dataset
from dateutils import dateshift
from mpl_toolkits.basemap import Basemap
import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(precision=3, suppress=True)


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def detect_environment():
    aws_paths = ['/data/resnet_data', '/data2/resnet_data']
    for path in aws_paths:
        if os.path.exists(path):
            print(f"Detected AWS environment (found {path})")
            return 'aws', path
    print("Detected local laptop environment")
    return 'laptop', None


ENVIRONMENT, AWS_BASE_PATH = detect_environment()


# ---------------------------------------------------------------------------
# Config reading
# ---------------------------------------------------------------------------

def read_config_file(config_file, directory_object_name):
    from configparser import ConfigParser
    config_object = ConfigParser()
    config_object.read(config_file)
    directory = config_object[directory_object_name]

    if "GRAFdatadir_conus_laptop" in directory:
        GRAFdatadir_conus_new = directory["GRAFdatadir_conus_laptop"]
        GRAFdatadir_conus_old = directory["GRAFdatadir_conus_laptop"]
        GRAFprobsdir_conus = directory["GRAFprobsdir_conus_laptop"]
        GRAF_plot_dir = directory.get("GRAF_plot_dir", directory["GRAFprobsdir_conus_laptop"])
    else:
        GRAFdatadir_conus_new = directory.get("GRAFdatadir_conus_new")
        GRAFdatadir_conus_old = directory.get("GRAFdatadir_conus_old", GRAFdatadir_conus_new)
        base_dir = directory.get("resnet_data_directory", AWS_BASE_PATH or "/data/resnet_data")
        GRAFprobsdir_conus = f"{base_dir}/probs/"
        GRAF_plot_dir = f"{base_dir}/plots/"

    return GRAFdatadir_conus_new, GRAFdatadir_conus_old, GRAFprobsdir_conus, GRAF_plot_dir


# ---------------------------------------------------------------------------
# GRAF data reading
# ---------------------------------------------------------------------------

def read_gribdata(gribfilename, endStep):
    import pygrib
    if not os.path.exists(gribfilename):
        print('grib file does not exist.')
        return -1, np.empty((0,0)), np.empty((0,0)), np.empty((0,0)), 0, 0, 0, 0
    try:
        fcstfile = pygrib.open(gribfilename)
        grb = fcstfile.select(endStep=endStep)[0]
        lats, lons = grb.latlons()
        precipitation = np.where(grb.values > 75., 75., grb.values)
        lon_0 = grb.projparams["lon_0"]
        lat_0 = grb.projparams["lat_0"]
        lat_1 = grb.projparams["lat_1"]
        lat_2 = grb.projparams["lat_2"]
        fcstfile.close()
        return 0, precipitation, lats, lons, lon_0, lat_0, lat_1, lat_2
    except Exception as e:
        print(f'   Error in read_gribdata: {e}')
        return -1, np.empty((0,0)), np.empty((0,0)), np.empty((0,0)), 0, 0, 0, 0


def GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus_new, GRAFdatadir_conus_old):
    il = int(clead)
    cyyyymmdd = cyyyymmddhh[0:8]
    chh = cyyyymmddhh[8:10]
    cyyyymmddhh_fcst = dateshift(cyyyymmddhh, il)
    cyyyymmdd_fcst = cyyyymmddhh_fcst[0:8]
    chh_fcst = cyyyymmddhh_fcst[8:10]

    if int(cyyyymmddhh) >= 2024040100:
        input_directory = GRAFdatadir_conus_new
        prefix = 'grid.hdo-graf_conus.'
    else:
        input_directory = GRAFdatadir_conus_old
        prefix = 'grid.hdo-graflr_conus.'

    input_directory = input_directory + cyyyymmdd + '/' + chh + '/'
    input_file = (prefix + cyyyymmdd_fcst + 'T' + chh_fcst + '0000Z.' +
                  cyyyymmdd + 'T' + chh + '0000Z.PT' + clead + 'H.CONUS@4km.APCP.SFC.grb2')
    infile = input_directory + input_file
    fexist = os.path.exists(infile)
    print(infile, fexist)

    if fexist:
        istat, precipitation, lats, lons, lon_0, lat_0, lat_1, lat_2 = \
            read_gribdata(infile, il)
    else:
        print(f'  could not find {infile}')
        istat = -1
        precipitation = np.empty((0, 0))
        lats = np.empty((0, 0), dtype=float)
        lons = np.empty((0, 0), dtype=float)
        lon_0 = lat_0 = lat_1 = lat_2 = 0.0

    return istat, precipitation, lats, lons, lon_0, lat_0, lat_1, lat_2


# ---------------------------------------------------------------------------
# Gamma mixture utilities
# ---------------------------------------------------------------------------

def find_nearest_point(lats, lons, target_lat, target_lon):
    dist2 = (lats - target_lat) ** 2 + (lons - target_lon) ** 2
    j, i = np.unravel_index(np.argmin(dist2), dist2.shape)
    return int(j), int(i)


def mixture_pdf(x, p0, w, a1, th1, a2, th2):
    f1 = gamma_dist.pdf(x, a=a1, scale=th1)
    f2 = gamma_dist.pdf(x, a=a2, scale=th2)
    return (1.0 - p0) * (w * f1 + (1.0 - w) * f2)


def mixture_exceedance(threshold, p0, w, a1, th1, a2, th2):
    sf1 = gamma_dist.sf(threshold, a=a1, scale=th1)
    sf2 = gamma_dist.sf(threshold, a=a2, scale=th2)
    return (1.0 - p0) * (w * sf1 + (1.0 - w) * sf2)


def reasonable_xmax(p0, w, a1, th1, a2, th2, target_cdf=0.999, minimum=5.0):
    x_test = np.linspace(1e-4, 150.0, 20000)
    cdf1 = gamma_dist.cdf(x_test, a=a1, scale=th1)
    cdf2 = gamma_dist.cdf(x_test, a=a2, scale=th2)
    mix_cdf = p0 + (1.0 - p0) * (w * cdf1 + (1.0 - w) * cdf2)
    idx = min(np.searchsorted(mix_cdf, target_cdf), len(x_test) - 1)
    return max(x_test[idx], minimum)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python plot_graf_and_pdf.py <YYYYMMDDHH> <lead_hours> <lat> <lon>")
        print("Example: python plot_graf_and_pdf.py 2025030100 12 39.5 -105.0")
        sys.exit(1)

    cyyyymmddhh = sys.argv[1]
    clead_arg   = sys.argv[2]
    target_lat  = float(sys.argv[3])
    target_lon  = float(sys.argv[4])
    clead = str(int(clead_arg))

    # --- Config ---
    config_file_name = 'config_aws.ini' if ENVIRONMENT == 'aws' else 'config_laptop.ini'
    GRAFdatadir_conus_new, GRAFdatadir_conus_old, GRAFprobsdir_conus, GRAF_plot_dir = \
        read_config_file(config_file_name, 'DIRECTORIES')
    os.makedirs(GRAF_plot_dir, exist_ok=True)

    # --- Read GRAF precipitation ---
    istat, precipitation_GRAF, lats, lons, lon_0, lat_0, lat_1, lat_2 = \
        GRAF_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus_new, GRAFdatadir_conus_old)
    if istat != 0:
        print('GRAF forecast data not found. Exiting.')
        sys.exit(1)

    # --- Read gamma mixture parameters ---
    nc_file = os.path.join(GRAFprobsdir_conus, f'{cyyyymmddhh}_{clead}_probs_gamma_mixture.nc')
    if not os.path.exists(nc_file):
        nc_file_alt = os.path.join(GRAFprobsdir_conus,
                                   f'{cyyyymmddhh}_{int(clead):02d}_probs_gamma_mixture.nc')
        if os.path.exists(nc_file_alt):
            nc_file = nc_file_alt
        else:
            print(f"ERROR: Cannot find netCDF file.\n  Tried: {nc_file}\n  Tried: {nc_file_alt}")
            sys.exit(1)

    print(f"Reading: {nc_file}")
    nc_prob = Dataset(nc_file, 'r')
    lats_prob      = nc_prob.variables['lat'][:, :]
    lons_prob      = nc_prob.variables['lon'][:, :]
    fraction_zero  = nc_prob.variables['fraction_zero'][:, :]
    mixture_weight = nc_prob.variables['mixture_weight'][:, :]
    shape1         = nc_prob.variables['gamma_shape1'][:, :]
    scale1         = nc_prob.variables['gamma_scale1'][:, :]
    shape2         = nc_prob.variables['gamma_shape2'][:, :]
    scale2         = nc_prob.variables['gamma_scale2'][:, :]
    nc_prob.close()

    # --- Find nearest grid point for PDF ---
    j, i = find_nearest_point(lats_prob, lons_prob, target_lat, target_lon)
    actual_lat = float(lats_prob[j, i])
    actual_lon = float(lons_prob[j, i])
    print(f"Requested : ({target_lat:.3f}N, {target_lon:.3f})")
    print(f"Grid point: ({actual_lat:.3f}N, {actual_lon:.3f})  [row={j}, col={i}]")

    p0  = float(fraction_zero[j, i])
    w   = float(mixture_weight[j, i])
    a1  = float(shape1[j, i])
    th1 = float(scale1[j, i])
    a2  = float(shape2[j, i])
    th2 = float(scale2[j, i])

    mean1 = a1 * th1
    mean2 = a2 * th2
    overall_mean = (1.0 - p0) * (w * mean1 + (1.0 - w) * mean2)
    print(f"P(X=0)={p0:.4f}, overall mean={overall_mean:.3f} mm")

    # --- Compute PDF on fine grid ---
    x_max = reasonable_xmax(p0, w, a1, th1, a2, th2)
    x = np.linspace(1e-4, x_max, 3000)
    pdf_full  = mixture_pdf(x, p0, w, a1, th1, a2, th2)
    pdf_comp1 = (1.0 - p0) * w       * gamma_dist.pdf(x, a=a1, scale=th1)
    pdf_comp2 = (1.0 - p0) * (1 - w) * gamma_dist.pdf(x, a=a2, scale=th2)

    keep_thresholds  = [5.0, 10.0]
    keep_labels      = ['5.0 mm', '10.0 mm']
    keep_colors_thr  = ['#ff7f00', '#e41a1c']
    keep_exceedances = [mixture_exceedance(t, p0, w, a1, th1, a2, th2)
                        for t in keep_thresholds]

    # --- Map zoom domain centred on target point ---
    llcrnrlon = target_lon - 6.0
    urcrnrlon = target_lon + 6.0
    llcrnrlat = target_lat - 6.0
    urcrnrlat = target_lat + 6.0

    # ---------------------------------------------------------------------------
    # Build figure
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 6.5))

    plt.suptitle(
        f'Hourly GRAF precipitation forecast and Attention ResUNet Probabilities, '
        f'{clead}-hour forecast for {cyyyymmddhh}',
        fontsize=19, y=0.995
    )

    # -----------------------------------------------------------------------
    # Panel (a): GRAF precipitation map
    # -----------------------------------------------------------------------
    ax1 = fig.add_axes([0.03, 0.14, 0.484, 0.737])
    ax1.set_title('(a) GRAF forecast', fontsize=18)

    m = Basemap(rsphere=(6378137.00, 6356752.3142),
                resolution='l', projection='lcc', area_thresh=1000.,
                lat_1=lat_1, lat_2=lat_2, lat_0=lat_0, lon_0=lon_0,
                llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
                urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
                ax=ax1)

    x_map, y_map = m(lons, lats)

    colorst_precip = ['White', '#E4FFFF', '#C4E8FF', '#8FB3FF', '#D8F9D8',
                      '#A6ECA6', '#42F742', 'Yellow', 'Gold', 'Orange', '#FCD5D9',
                      '#F6A3AE', '#FA5257', 'Orchid', '#AD8ADB', '#A449FF', 'LightGray']
    clevs_precip = [0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 35]

    CS1 = m.contourf(x_map, y_map, precipitation_GRAF, clevs_precip,
                     cmap=None, colors=colorst_precip, extend='both')
    m.drawcoastlines(linewidth=0.6, color='Gray')
    m.drawcountries(linewidth=0.4, color='Gray')
    m.drawstates(linewidth=0.2, color='Gray')

    # + marker at target lat/lon
    xpt, ypt = m(target_lon, target_lat)
    ax1.plot(xpt, ypt, 'k+', markersize=12, markeredgewidth=2.0, zorder=10)

    # Colorbar below map
    cax1 = fig.add_axes([0.04, 0.085, 0.464, 0.025])
    cb1 = plt.colorbar(CS1, orientation='horizontal', cax=cax1,
                       drawedges=True, ticks=clevs_precip, format='%g')
    cb1.ax.tick_params(labelsize=11)
    cb1.set_label('Precipitation (mm)', fontsize=14)

    # -----------------------------------------------------------------------
    # Panel (b): Gamma mixture PDF
    # -----------------------------------------------------------------------
    ax2 = fig.add_axes([0.56, 0.30, 0.41, 0.58])
    ax2.set_title('(b) Fitted probabilities', fontsize=18)

    ax2.fill_between(x, pdf_full, alpha=0.18, color='steelblue', zorder=1)
    ax2.plot(x, pdf_full, color='steelblue', linewidth=2.0, zorder=2,
             label='Mixture PDF')
    ax2.plot(x, pdf_comp1, color='#4daf4a', linewidth=1.5, linestyle='--',
             alpha=0.85, zorder=3,
             label=f'Comp 1: α={a1:.2f}, θ={th1:.2f}, μ={mean1:.2f} mm')
    ax2.plot(x, pdf_comp2, color='#e41a1c', linewidth=1.5, linestyle='--',
             alpha=0.85, zorder=3,
             label=f'Comp 2: α={a2:.2f}, θ={th2:.2f}, μ={mean2:.2f} mm')

    y_tick_top = pdf_full.max() * 1.12

    ax2.set_xlim(left=0.0, right=x_max)
    ax2.set_ylim(bottom=0.0, top=y_tick_top * 1.05)
    ax2.set_xlabel('Precipitation amount (mm)', fontsize=13)
    ax2.set_ylabel('Probability density (mm⁻¹)', fontsize=13)
    ax2.tick_params(axis='both', labelsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')

    lon_dir = 'W' if actual_lon < 0 else 'E'
    ax2.text(0.02, 0.98,
             f'({actual_lat:.2f}°N, {abs(actual_lon):.2f}°{lon_dir})',
             transform=ax2.transAxes, fontsize=8, va='top')

    patch = mpatches.Patch(facecolor='steelblue', edgecolor='navy',
                           label=f'P(X=0) = {p0:.3f}')
    leg_h, leg_l = ax2.get_legend_handles_labels()
    ax2.legend(handles=leg_h + [patch], labels=leg_l + [f'P(X=0) = {p0:.3f}'],
               loc='upper right', fontsize=8, framealpha=0.85)

    # Probability bar below PDF axes
    ax_pbar = fig.add_axes([0.56, 0.055, 0.41, 0.14])

    ax_pbar.barh(0, 1.0 - p0, left=p0, height=0.55,
                 color='#aec7e8', edgecolor='navy', linewidth=1.5, align='center')
    ax_pbar.barh(0, p0, left=0.0, height=0.55,
                 color='steelblue', edgecolor='navy', linewidth=1.5, align='center')
    ax_pbar.axvline(x=p0, color='navy', linewidth=2.0, zorder=5)

    if p0 > 0.06:
        ax_pbar.text(p0 / 2, 0, f'P(X=0) = {p0:.3f}',
                     ha='center', va='center', fontsize=10, color='white')
    else:
        ax_pbar.text(p0 + 0.01, 0, f'P(X=0)={p0:.3f}',
                     ha='left', va='center', fontsize=10, color='navy')
    if (1.0 - p0) > 0.12:
        ax_pbar.text(p0 + (1.0 - p0) / 2, 0, f'P(wet) = {1.0 - p0:.3f}',
                     ha='center', va='center', fontsize=10, color='navy')

    ax_pbar.set_xlim(0.0, 1.0)
    ax_pbar.set_ylim(-0.5, 0.5)
    ax_pbar.set_yticks([])
    ax_pbar.set_xticks([])
    ax_pbar.set_frame_on(False)

    # --- Save ---
    out_name = os.path.join(
        GRAF_plot_dir,
        f'GRAF_PDF_{cyyyymmddhh}_{clead}h_{target_lat:.2f}_{target_lon:.2f}.png'
    )
    fig.savefig(out_name, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved: {out_name}")
    plt.close()
