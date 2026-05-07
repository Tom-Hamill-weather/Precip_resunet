"""
plot_gamma_mixture_pdf.py  --  visualize the zero-inflated 2-component Gamma mixture PDF
at a user-chosen point from resunet_inference_gamma_mixture_optimized.py output.

Usage:
    python plot_gamma_mixture_pdf.py <YYYYMMDDHH> <lead_hours> <lat> <lon>

Example:
    python plot_gamma_mixture_pdf.py 2025030100 12 39.5 -105.0

The zero-precipitation probability mass (P(X=0) = p0) is displayed as a bar
whose area equals p0, following the approach in Scheuerer & Hamill (2015, MWR).
The continuous mixture PDF (area = 1 - p0) is plotted for x > 0.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist
from netCDF4 import Dataset

# ---------------------------------------------------------------------------
# Environment detection (mirrors inference script logic)
# ---------------------------------------------------------------------------

def detect_environment():
    aws_paths = ['/data/resnet_data', '/data2/resnet_data']
    for path in aws_paths:
        if os.path.exists(path):
            return 'aws', path
    aws_training_paths = ['/data/trainings', '/data2/trainings']
    for path in aws_training_paths:
        if os.path.exists(path):
            return 'aws', os.path.dirname(path)
    return 'laptop', None

ENVIRONMENT, AWS_BASE_PATH = detect_environment()


def get_probs_dir():
    if ENVIRONMENT == 'aws':
        return f'{AWS_BASE_PATH}/probs/'
    from configparser import ConfigParser
    config = ConfigParser()
    config.read('config_laptop.ini')
    return config['DIRECTORIES']['GRAFprobsdir_conus_laptop']


# ---------------------------------------------------------------------------
# Grid and distribution utilities
# ---------------------------------------------------------------------------

def find_nearest_point(lats, lons, target_lat, target_lon):
    dist2 = (lats - target_lat) ** 2 + (lons - target_lon) ** 2
    j, i = np.unravel_index(np.argmin(dist2), dist2.shape)
    return int(j), int(i)


def mixture_pdf(x, p0, w, a1, th1, a2, th2):
    """Full PDF of the zero-inflated mixture for x > 0."""
    f1 = gamma_dist.pdf(x, a=a1, scale=th1)
    f2 = gamma_dist.pdf(x, a=a2, scale=th2)
    return (1.0 - p0) * (w * f1 + (1.0 - w) * f2)


def mixture_exceedance(threshold, p0, w, a1, th1, a2, th2):
    """P(X > threshold) for the full mixture."""
    sf1 = gamma_dist.sf(threshold, a=a1, scale=th1)
    sf2 = gamma_dist.sf(threshold, a=a2, scale=th2)
    return (1.0 - p0) * (w * sf1 + (1.0 - w) * sf2)


def reasonable_xmax(p0, w, a1, th1, a2, th2, target_cdf=0.999, minimum=5.0):
    """Return x such that the mixture CDF reaches target_cdf."""
    x_test = np.linspace(1e-4, 150.0, 20000)
    cdf1 = gamma_dist.cdf(x_test, a=a1, scale=th1)
    cdf2 = gamma_dist.cdf(x_test, a=a2, scale=th2)
    mix_cdf = p0 + (1.0 - p0) * (w * cdf1 + (1.0 - w) * cdf2)
    idx = np.searchsorted(mix_cdf, target_cdf)
    idx = min(idx, len(x_test) - 1)
    return max(x_test[idx], minimum)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: python plot_gamma_mixture_pdf.py <YYYYMMDDHH> <lead_hours> <lat> <lon>")
        print("Example: python plot_gamma_mixture_pdf.py 2025030100 12 39.5 -105.0")
        sys.exit(1)

    cyyyymmddhh = sys.argv[1]
    clead_arg   = sys.argv[2]
    target_lat  = float(sys.argv[3])
    target_lon  = float(sys.argv[4])

    # Normalise lead to an integer string (e.g. "06" → "6")
    clead = str(int(clead_arg))

    # --- Locate netCDF file ---
    probs_dir = get_probs_dir()
    nc_file = os.path.join(probs_dir, f'{cyyyymmddhh}_{clead}_probs_gamma_mixture.nc')
    if not os.path.exists(nc_file):
        # Try zero-padded lead (rare but possible)
        nc_file_alt = os.path.join(probs_dir,
                                   f'{cyyyymmddhh}_{int(clead):02d}_probs_gamma_mixture.nc')
        if os.path.exists(nc_file_alt):
            nc_file = nc_file_alt
        else:
            print(f"ERROR: Cannot find output file.\n  Tried: {nc_file}\n  Tried: {nc_file_alt}")
            sys.exit(1)

    print(f"Reading: {nc_file}")
    nc   = Dataset(nc_file, 'r')
    lats = nc.variables['lat'][:, :]
    lons = nc.variables['lon'][:, :]
    fraction_zero  = nc.variables['fraction_zero'][:, :]   # auto-scaled to [0, 1]
    mixture_weight = nc.variables['mixture_weight'][:, :]  # auto-scaled to [0, 1]
    shape1 = nc.variables['gamma_shape1'][:, :]
    scale1 = nc.variables['gamma_scale1'][:, :]
    shape2 = nc.variables['gamma_shape2'][:, :]
    scale2 = nc.variables['gamma_scale2'][:, :]
    nc.close()

    # --- Find nearest grid point ---
    j, i = find_nearest_point(lats, lons, target_lat, target_lon)
    actual_lat = float(lats[j, i])
    actual_lon = float(lons[j, i])
    print(f"Requested : ({target_lat:.3f}°, {target_lon:.3f}°)")
    print(f"Grid point: ({actual_lat:.3f}°, {actual_lon:.3f}°)  [row={j}, col={i}]")

    p0  = float(fraction_zero[j, i])
    w   = float(mixture_weight[j, i])
    a1  = float(shape1[j, i])
    th1 = float(scale1[j, i])
    a2  = float(shape2[j, i])
    th2 = float(scale2[j, i])

    mean1 = a1 * th1
    mean2 = a2 * th2
    overall_mean = (1.0 - p0) * (w * mean1 + (1.0 - w) * mean2)

    print(f"\nMixture parameters at this point:")
    print(f"  P(X = 0)              = {p0:.4f}")
    print(f"  Mixing weight (w)     = {w:.4f}  (fraction for component 1)")
    print(f"  Component 1 (light):  α={a1:.4f}, θ={th1:.4f}, mean={mean1:.3f} mm")
    print(f"  Component 2 (heavy):  α={a2:.4f}, θ={th2:.4f}, mean={mean2:.3f} mm")
    print(f"  Overall mean precip   = {overall_mean:.3f} mm")

    # --- Compute PDF on a fine grid ---
    x_max = reasonable_xmax(p0, w, a1, th1, a2, th2)
    x = np.linspace(1e-4, x_max, 3000)

    pdf_full  = mixture_pdf(x, p0, w, a1, th1, a2, th2)
    pdf_comp1 = (1.0 - p0) * w       * gamma_dist.pdf(x, a=a1, scale=th1)
    pdf_comp2 = (1.0 - p0) * (1 - w) * gamma_dist.pdf(x, a=a2, scale=th2)

    # --- Standard thresholds ---
    thresholds   = [0.25, 1.0, 2.5, 5.0, 10.0]
    thresh_labels = ['0.25 mm', '1.0 mm', '2.5 mm', '5.0 mm', '10.0 mm']
    thresh_colors = ['#4daf4a', '#377eb8', '#984ea3', '#ff7f00', '#e41a1c']
    exceedances   = [mixture_exceedance(t, p0, w, a1, th1, a2, th2) for t in thresholds]

    for lbl, exc in zip(thresh_labels, exceedances):
        print(f"  P(X > {lbl:7s}) = {exc:.4f}")

    # --- Build plot ---
    # Axes placed manually to guarantee the probability bar never overlaps
    # the main plot's xlabel/ticks regardless of content.
    fig = plt.figure(figsize=(10, 8))
    ax      = fig.add_axes([0.10, 0.25, 0.86, 0.67])   # main PDF plot
    ax_pbar = fig.add_axes([0.10, 0.04, 0.86, 0.12])   # probability bar

    # Shaded area under the full continuous mixture PDF
    ax.fill_between(x, pdf_full, alpha=0.18, color='steelblue', zorder=1)

    # Full mixture PDF
    ax.plot(x, pdf_full, color='steelblue', linewidth=2.2, zorder=2,
            label='Mixture PDF (area = 1-p₀)')
    ax.plot(x, pdf_comp1, color='#4daf4a', linewidth=1.5, linestyle='--',
            alpha=0.85, zorder=3,
            label=f'Comp 1 (light):  α={a1:.2f}, θ={th1:.2f}, μ={mean1:.2f} mm')
    ax.plot(x, pdf_comp2, color='#e41a1c', linewidth=1.5, linestyle='--',
            alpha=0.85, zorder=3,
            label=f'Comp 2 (heavy):  α={a2:.2f}, θ={th2:.2f}, μ={mean2:.2f} mm')

    # Threshold vertical lines — keep only orange (5 mm) and red (10 mm)
    keep_thresholds   = [5.0, 10.0]
    keep_labels       = ['5.0 mm', '10.0 mm']
    keep_colors       = ['#ff7f00', '#e41a1c']
    keep_exceedances  = [mixture_exceedance(t, p0, w, a1, th1, a2, th2)
                         for t in keep_thresholds]

    y_tick_top = pdf_full.max() * 1.12
    for thr, lbl, col, exc in zip(keep_thresholds, keep_labels,
                                   keep_colors, keep_exceedances):
        if thr < x_max * 0.90:
            ax.axvline(x=thr, color=col, linestyle=':', linewidth=1.5,
                       alpha=0.85, zorder=4)
            ax.text(thr, y_tick_top * 0.97,
                    f'{lbl}\nP>{exc:.3f}',
                    ha='center', va='top', fontsize=14,
                    color=col, fontweight='bold')

    # Axes limits and labels
    ax.set_xlim(left=0.0, right=x_max)
    ax.set_ylim(bottom=0.0, top=y_tick_top * 1.05)
    ax.set_xlabel('Precipitation amount (mm)', fontsize=22)
    ax.set_ylabel('Probability density (mm⁻¹)', fontsize=22)
    ax.tick_params(axis='both', labelsize=18)

    # Title
    date_str = (f'{cyyyymmddhh[0:4]}-{cyyyymmddhh[4:6]}-{cyyyymmddhh[6:8]} '
                f'{cyyyymmddhh[8:10]}Z')
    lon_dir  = 'W' if actual_lon < 0 else 'E'
    ax.set_title(
        f'Zero-inflated 2-component Gamma mixture PDF\n'
        f'Init: {date_str}  |  Lead: {clead}h  |  '
        f'Grid point: ({actual_lat:.2f}°N, {abs(actual_lon):.2f}°{lon_dir})',
        fontsize=19
    )

    ax.grid(True, alpha=0.3, linestyle='--')

    from matplotlib.patches import Patch
    leg_handles, leg_labels = ax.get_legend_handles_labels()
    leg_handles.append(Patch(facecolor='steelblue', edgecolor='navy'))
    leg_labels.append(f'P(X = 0) = {p0:.3f}  [area = p₀]')
    ax.legend(handles=leg_handles, labels=leg_labels,
              loc='upper right', fontsize=12, framealpha=0.85)

    # --- Horizontal probability bar (bottom subplot) ---
    # Full bar background represents P(wet); dark-filled portion = P(X=0)
    ax_pbar.barh(0, 1.0 - p0, left=p0, height=0.55,
                 color='#aec7e8', edgecolor='navy', linewidth=1.5, align='center')
    ax_pbar.barh(0, p0, left=0.0, height=0.55,
                 color='steelblue', edgecolor='navy', linewidth=1.5, align='center')

    # Boundary line between the two segments
    ax_pbar.axvline(x=p0, color='navy', linewidth=2.0, zorder=5)

    # Labels inside / near the bar
    if p0 > 0.06:
        ax_pbar.text(p0 / 2, 0, f'P(X=0) = {p0:.3f}',
                     ha='center', va='center', fontsize=15,
                     color='white', fontweight='bold')
    else:
        ax_pbar.text(p0 + 0.01, 0, f'P(X=0)={p0:.3f}',
                     ha='left', va='center', fontsize=15,
                     color='navy', fontweight='bold')
    if (1.0 - p0) > 0.12:
        ax_pbar.text(p0 + (1.0 - p0) / 2, 0,
                     f'P(wet) = {1.0 - p0:.3f}',
                     ha='center', va='center', fontsize=15,
                     color='navy', fontweight='bold')

    ax_pbar.set_xlim(0.0, 1.0)
    ax_pbar.set_ylim(-0.5, 0.5)
    ax_pbar.set_yticks([])
    ax_pbar.set_xlabel('Probability', fontsize=18)
    ax_pbar.tick_params(axis='x', labelsize=15)

    # --- Save ---
    out_name = (f'pdf_{cyyyymmddhh}_{clead}h_'
                f'{target_lat:.2f}_{target_lon:.2f}.png')
    plt.savefig(out_name, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {out_name}")
    plt.close()
