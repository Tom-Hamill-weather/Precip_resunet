"""
python make_plots_6hourly_mlp_4panel.py cyyyymmddhh clead

Four-panel 2x2 plot for 6-hourly MLP probabilistic forecasts:
  (a) GRAF deterministic 6-h precipitation (sum of 6 1-h APCP files)
  (b) 6-h P(>= 0.25 mm)
  (c) 6-h P(>= 2.5 mm)
  (d) 6-h P(>= 10 mm)

The MLP checkpoint must exist at:
    mlp_trainings/6h_mlp_lead{clead}h.pth

Tom Hamill, May 2026
"""

from configparser import ConfigParser
import numpy as np
import os, sys
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import warnings
from dateutils import dateshift
from scipy.special import gammainc
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)

# =========================================================================
# Environment detection
# =========================================================================

def detect_environment():
    for path in ['/data/resnet_data', '/data2/resnet_data']:
        if os.path.exists(path):
            print(f"Detected AWS environment (found {path})")
            return 'aws', path
    print("Detected local laptop environment")
    return 'laptop', None

ENVIRONMENT, AWS_BASE_PATH = detect_environment()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================================
# Config
# =========================================================================

def read_config_file(config_file):
    print(f'INFO: {config_file}')
    config_object = ConfigParser()
    config_object.read(config_file)
    directory = config_object['DIRECTORIES']

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

    print(f"  GRAF new path: {GRAFdatadir_conus_new}")
    print(f"  GRAF old path: {GRAFdatadir_conus_old}")
    print(f"  Probs path: {GRAFprobsdir_conus}")
    print(f"  Plot directory: {GRAF_plot_dir}")
    return GRAFdatadir_conus_new, GRAFdatadir_conus_old, GRAFprobsdir_conus, GRAF_plot_dir

# =========================================================================
# MLP model (must match train_6hourly_mlp.py exactly)
# =========================================================================

SHAPE_MIN    = 0.1
SCALE_MIN    = 0.01
HIDDEN_SIZES = [72, 144, 72, 36, 12]


class GammaMixtureMLP(nn.Module):
    def __init__(self, hidden_sizes=HIDDEN_SIZES,
                 shape_min=SHAPE_MIN, scale_min=SCALE_MIN):
        super().__init__()
        self.shape_min = shape_min
        self.scale_min = scale_min
        layer_sizes = [36] + hidden_sizes
        layers = []
        for in_sz, out_sz in zip(layer_sizes, layer_sizes[1:]):
            layers += [nn.Linear(in_sz, out_sz),
                       nn.BatchNorm1d(out_sz),
                       nn.ReLU()]
        layers.append(nn.Linear(hidden_sizes[-1], 6))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        raw        = self.net(x)
        frac_zero  = torch.sigmoid(raw[:, 0])
        mix_weight = torch.sigmoid(raw[:, 1])
        shape1     = self.shape_min + F.softplus(raw[:, 2])
        scale1     = self.scale_min + F.softplus(raw[:, 3])
        shape2     = self.shape_min + F.softplus(raw[:, 4])
        scale2     = self.scale_min + F.softplus(raw[:, 5])
        swap           = (shape1 * scale1 > shape2 * scale2).float()
        shape1_out     = (1 - swap) * shape1  + swap * shape2
        scale1_out     = (1 - swap) * scale1  + swap * scale2
        shape2_out     = (1 - swap) * shape2  + swap * shape1
        scale2_out     = (1 - swap) * scale2  + swap * scale1
        mix_weight_out = (1 - swap) * mix_weight + swap * (1 - mix_weight)
        return frac_zero, mix_weight_out, shape1_out, scale1_out, shape2_out, scale2_out


def load_mlp(clead, device):
    ckpt_path = os.path.join(SCRIPT_DIR, 'mlp_trainings',
                             f'6h_mlp_lead{clead}h.pth')
    if not os.path.exists(ckpt_path):
        print(f'ERROR: MLP checkpoint not found: {ckpt_path}')
        print(f'  Run:  python train_6hourly_mlp.py {clead}')
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden_sizes = ckpt.get('hidden_sizes', HIDDEN_SIZES)
    shape_min    = ckpt.get('shape_min',    SHAPE_MIN)
    scale_min    = ckpt.get('scale_min',    SCALE_MIN)

    model = GammaMixtureMLP(hidden_sizes=hidden_sizes,
                            shape_min=shape_min, scale_min=scale_min)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    feat_mean = ckpt['feature_mean']
    feat_std  = ckpt['feature_std']
    print(f'Loaded MLP from {ckpt_path}  (epoch {ckpt["epoch"]+1})')
    return model, feat_mean, feat_std

# =========================================================================
# GRAF 6-h precipitation (sum of 6 consecutive 1-h APCP files)
# =========================================================================

def read_gribdata(gribfilename, endStep):
    import pygrib
    if not os.path.exists(gribfilename):
        print(f'  grib file not found: {gribfilename}')
        return -1, None, None, None, None, None, None, None

    try:
        fcstfile = pygrib.open(gribfilename)
        grb = fcstfile.select(endStep=endStep)[0]
        lats, lons = grb.latlons()
        precipitation = np.where(grb.values < 0., 0., grb.values)
        lon_0 = grb.projparams["lon_0"]
        lat_0 = grb.projparams["lat_0"]
        lat_1 = grb.projparams["lat_1"]
        lat_2 = grb.projparams["lat_2"]
        fcstfile.close()
        return 0, precipitation, lats, lons, lon_0, lat_0, lat_1, lat_2
    except Exception as e:
        print(f'  Error reading {gribfilename}: {e}')
        return -1, None, None, None, None, None, None, None


def graf_grib_path(lead, cyyyymmddhh, GRAFdatadir_conus_new, GRAFdatadir_conus_old):
    """Return the file path for the 1-h APCP GRAF grib at the given lead."""
    cyyyymmdd = cyyyymmddhh[0:8]
    chh       = cyyyymmddhh[8:10]
    cyyyymmddhh_fcst = dateshift(cyyyymmddhh, lead)
    cyyyymmdd_fcst   = cyyyymmddhh_fcst[0:8]
    chh_fcst         = cyyyymmddhh_fcst[8:10]

    if int(cyyyymmddhh) >= 2024040100:
        input_directory = GRAFdatadir_conus_new
        prefix = 'grid.hdo-graf_conus.'
    else:
        input_directory = GRAFdatadir_conus_old
        prefix = 'grid.hdo-graflr_conus.'

    clead_str = str(lead)
    input_directory = input_directory + cyyyymmdd + '/' + chh + '/'
    input_file = (prefix + cyyyymmdd_fcst + 'T' + chh_fcst + '0000Z.'
                  + cyyyymmdd + 'T' + chh + '0000Z.PT' + clead_str
                  + 'H.CONUS@4km.APCP.SFC.grb2')
    return input_directory + input_file, lead


def GRAF_6h_precip_read(clead, cyyyymmddhh, GRAFdatadir_conus_new, GRAFdatadir_conus_old):
    """
    Sum 6 consecutive 1-h GRAF APCP files (leads clead-5 through clead)
    to get 6-h accumulated precipitation.
    """
    lead_times = list(range(clead - 5, clead + 1))
    precip_6h = None
    lats = lons = None
    lon_0 = lat_0 = lat_1 = lat_2 = None

    for lt in lead_times:
        fpath, endstep = graf_grib_path(lt, cyyyymmddhh,
                                        GRAFdatadir_conus_new, GRAFdatadir_conus_old)
        print(fpath, os.path.exists(fpath))
        istat, precip, _lats, _lons, _lon0, _lat0, _lat1, _lat2 = \
            read_gribdata(fpath, endstep)
        if istat != 0:
            print(f'  Missing GRAF file at lead {lt}h — cannot compute 6-h total.')
            return -1, None, None, None, None, None, None, None, None, None, None, None, None

        if precip_6h is None:
            precip_6h = precip.copy()
            lats  = _lats;  lons  = _lons
            lon_0 = _lon0;  lat_0 = _lat0
            lat_1 = _lat1;  lat_2 = _lat2
        else:
            precip_6h += precip

    precip_6h = np.where(precip_6h > 200., 200., precip_6h)
    ny, nx = lats.shape
    latmax = np.max(lats); latmin = np.min(lats)
    lonmax = np.max(lons); lonmin = np.min(lons)
    return (0, precip_6h, lats, lons, ny, nx,
            latmin, latmax, lonmin, lonmax, lon_0, lat_0, lat_1, lat_2)

# =========================================================================
# Read 6 hourly gamma-mixture parameter files and apply MLP
# =========================================================================

PARAM_VARS = [
    'fraction_zero', 'mixture_weight',
    'gamma_shape1',  'gamma_scale1',
    'gamma_shape2',  'gamma_scale2',
]
MLP_BATCH = 131072


def read_prob_params_6h(probs_dir, cyyyymmddhh, clead):
    lead_times = list(range(clead - 5, clead + 1))
    stacks = {k: [] for k in PARAM_VARS}
    lat = lon = None

    for lt in lead_times:
        fname = os.path.join(probs_dir,
                             f'{cyyyymmddhh}_{lt}_probs_gamma_mixture.nc')
        if not os.path.exists(fname):
            print(f'  Missing prob file: {fname}')
            return None, None, None
        try:
            with Dataset(fname, 'r') as ds:
                for k in PARAM_VARS:
                    arr = ds.variables[k][:].data.astype(np.float32)
                    stacks[k].append(arr)
                if lat is None:
                    lat = ds.variables['lat'][:].data.astype(np.float32)
                    lon = ds.variables['lon'][:].data.astype(np.float32)
        except Exception as exc:
            print(f'  WARNING: cannot read {fname}: {exc}')
            return None, None, None

    for k in PARAM_VARS:
        stacks[k] = np.stack(stacks[k], axis=0)   # (6, ny, nx)

    return stacks, lat, lon


def apply_mlp_fulldomain(model, feat_mean, feat_std, params_6h, ny, nx, device):
    npix   = ny * nx
    blocks = [params_6h[k].reshape(6, npix).T for k in PARAM_VARS]
    feats  = np.concatenate(blocks, axis=1).astype(np.float32)   # (npix, 36)

    std_safe   = np.where(feat_std < 1e-8, 1.0, feat_std)
    feats_norm = (feats - feat_mean) / std_safe

    out = {i: [] for i in range(6)}
    with torch.no_grad():
        for start in range(0, npix, MLP_BATCH):
            end = min(start + MLP_BATCH, npix)
            xb  = torch.tensor(feats_norm[start:end], dtype=torch.float32,
                               device=device)
            fz, mw, s1, sc1, s2, sc2 = model(xb)
            for i, t in enumerate([fz, mw, s1, sc1, s2, sc2]):
                out[i].append(t.cpu().numpy())

    result = [np.concatenate(out[i]).reshape(ny, nx) for i in range(6)]
    return tuple(result)   # frac_zero, mix_weight, shape1, scale1, shape2, scale2


def exceedance_prob(frac_zero, mix_weight, shape1, scale1, shape2, scale2, threshold):
    if threshold <= 0.0:
        return np.clip(1.0 - frac_zero, 0.0, 1.0)
    eps = 1e-7
    s1  = np.maximum(shape1, eps);  sc1 = np.maximum(scale1, eps)
    s2  = np.maximum(shape2, eps);  sc2 = np.maximum(scale2, eps)
    sf1 = 1.0 - gammainc(s1, threshold / sc1)
    sf2 = 1.0 - gammainc(s2, threshold / sc2)
    mw  = np.clip(mix_weight, 0.0, 1.0)
    p_nz = np.clip(1.0 - frac_zero, 0.0, 1.0)
    return np.clip(p_nz * (mw * sf1 + (1.0 - mw) * sf2), 0.0, 1.0)

# =========================================================================
# Four-panel 2x2 plot
# =========================================================================

def plot_4panel(lat_1, lat_2, lat_0, lon_0, lons, lats,
                llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat,
                cyyyymmddhh, clead,
                precip_6h_GRAF, prob_0p25mm, prob_2p5mm, prob_10mm,
                GRAF_plot_dir):

    m = Basemap(rsphere=(6378137.00, 6356752.3142),
                resolution='i', projection='lcc', area_thresh=1000.,
                lat_1=lat_1, lat_2=lat_2, lat_0=lat_0, lon_0=lon_0,
                llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
                urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat)

    x, y = m(lons, lats)

    cyyyymmddhh_valid = dateshift(cyyyymmddhh, int(clead))
    cmonths = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    cmonth = cmonths[int(cyyyymmddhh_valid[4:6]) - 1]
    datestring = (cyyyymmddhh_valid[8:10] + ' UTC ' +
                  cyyyymmddhh_valid[6:8] + ' ' + cmonth + ' ' +
                  cyyyymmddhh_valid[0:4])

    clead_minus = str(int(clead) - 5)

    # Precipitation colour scale (6-h totals)
    colorst_precip = ['White', '#E4FFFF', '#C4E8FF', '#8FB3FF', '#D8F9D8',
                      '#A6ECA6', '#42F742', 'Yellow', 'Gold', 'Orange',
                      '#FCD5D9', '#F6A3AE', '#FA5257', 'Orchid',
                      '#AD8ADB', '#A449FF', 'LightGray']
    clevs_precip = [0, 0.5, 1, 2, 5, 10, 15, 20, 25, 35, 50, 75]

    # Probability colour scale (shared by all three probability panels)
    clevs_prob = [0, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 1.0]

    # 2x2 panel layout (matches make_plots_gamma_mixture2_fulldomain.py)
    axlocs = [
        [0.01, 0.560, 0.48, 0.36],   # (a) top-left
        [0.51, 0.560, 0.48, 0.36],   # (b) top-right
        [0.01, 0.110, 0.48, 0.36],   # (c) bottom-left
        [0.51, 0.110, 0.48, 0.36],   # (d) bottom-right
    ]
    cbar_y   = [0.537, 0.537, 0.087, 0.087]
    cbar_x   = [0.03,  0.53,  0.03,  0.53]
    cbar_w   = 0.44
    cbar_h   = 0.013

    fig = plt.figure(figsize=(9, 11.))
    plt.suptitle(clead_minus + ' to ' + clead +
                 '-h GRAF-based 6-h probabilistic forecasts, valid ' + datestring,
                 fontsize=14, y=0.975)

    def draw_panel(axloc, data, clevs, colors, cx, cy, title, cbar_label):
        ax = fig.add_axes(axloc)
        ax.set_title(title, fontsize=14, color='Black')
        CS = m.contourf(x, y, data, clevs, cmap=None, colors=colors, extend='both')
        m.drawcoastlines(linewidth=0.6, color='Gray')
        m.drawcountries(linewidth=0.4, color='Gray')
        m.drawstates(linewidth=0.2, color='Gray')
        cax = fig.add_axes([cx, cy, cbar_w, cbar_h])
        cb = plt.colorbar(CS, orientation='horizontal', cax=cax,
                          drawedges=True, ticks=clevs, format='%g')
        cb.ax.tick_params(labelsize=6)
        cb.set_label(cbar_label, fontsize=10)
        return ax

    x_dot, y_dot = m(-122.25, 44.)

    for axloc, data, clevs, colors, cx, cy, title, cbar_label in [
        (axlocs[0], precip_6h_GRAF, clevs_precip, colorst_precip,
         cbar_x[0], cbar_y[0],
         f'(a) GRAF 6-h precipitation ({clead_minus}–{clead} h)', 'Precipitation (mm)'),
        (axlocs[1], prob_0p25mm, clevs_prob, colorst_precip,
         cbar_x[1], cbar_y[1], r'(b) 6-h P($\geq$0.25 mm)', 'Probability'),
        (axlocs[2], prob_2p5mm, clevs_prob, colorst_precip,
         cbar_x[2], cbar_y[2], r'(c) 6-h P($\geq$2.5 mm)', 'Probability'),
        (axlocs[3], prob_10mm, clevs_prob, colorst_precip,
         cbar_x[3], cbar_y[3], r'(d) 6-h P($\geq$10 mm)', 'Probability'),
    ]:
        ax = draw_panel(axloc, data, clevs, colors, cx, cy, title, cbar_label)
        ax.plot(x_dot, y_dot, 'k.', markersize=4)

    plot_title = (GRAF_plot_dir + '6h_MLP_4panel_IC' +
                  cyyyymmddhh + '_lead' + clead + 'h.png')
    fig.savefig(plot_title, dpi=400, bbox_inches='tight')
    print('Saved plot to:', plot_title)
    plt.close(fig)
    return 0

# =========================================================================
# Main
# =========================================================================

cyyyymmddhh = sys.argv[1]
clead_int   = int(sys.argv[2])
clead       = sys.argv[2]

if clead_int < 6:
    print(f'ERROR: clead must be >= 6 (need 6 consecutive lead times)')
    sys.exit(1)

if ENVIRONMENT == 'aws':
    config_file_name = 'config_aws.ini'
else:
    config_file_name = 'config_laptop.ini'

print(f"Using config file: {config_file_name}")
GRAFdatadir_conus_new, GRAFdatadir_conus_old, GRAFprobsdir_conus, \
    GRAF_plot_dir = read_config_file(config_file_name)

os.makedirs(GRAF_plot_dir, exist_ok=True)

# --- GRAF 6-h precipitation ---
istat_GRAF, precip_6h_GRAF, lats, lons, ny, nx, \
    latmin, latmax, lonmin, lonmax, lon_0, lat_0, lat_1, lat_2 = \
    GRAF_6h_precip_read(clead_int, cyyyymmddhh,
                        GRAFdatadir_conus_new, GRAFdatadir_conus_old)

if istat_GRAF != 0:
    print('GRAF 6-h precipitation data not found. Exiting.')
    sys.exit(1)

# --- MLP probabilities ---
params_6h, lat_prob, lon_prob = read_prob_params_6h(
    GRAFprobsdir_conus, cyyyymmddhh, clead_int)

if params_6h is None:
    print('Gamma-mixture parameter files not found. Exiting.')
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Torch device: {device}')

model, feat_mean, feat_std = load_mlp(clead_int, device)

fz, mw, s1, sc1, s2, sc2 = apply_mlp_fulldomain(
    model, feat_mean, feat_std, params_6h, ny, nx, device)

prob_0p25mm = exceedance_prob(fz, mw, s1, sc1, s2, sc2, 0.25)
prob_2p5mm  = exceedance_prob(fz, mw, s1, sc1, s2, sc2, 2.5)
prob_10mm   = exceedance_prob(fz, mw, s1, sc1, s2, sc2, 10.0)

print(f'P(>=0.25 mm) max={prob_0p25mm.max():.3f} mean={prob_0p25mm.mean():.4f}')
print(f'P(>=2.5 mm)  max={prob_2p5mm.max():.3f}  mean={prob_2p5mm.mean():.4f}')
print(f'P(>=10 mm)   max={prob_10mm.max():.3f}   mean={prob_10mm.mean():.4f}')

# --- Domain corners ---
if cyyyymmddhh == '2025120412':
    llcrnrlon = -125; llcrnrlat = 38.5
    urcrnrlon = -106.5; urcrnrlat = 53.
elif cyyyymmddhh == '2025120812':
    llcrnrlon = -125; llcrnrlat = 38.5
    urcrnrlon = -106.5; urcrnrlat = 53.
elif cyyyymmddhh == '2025122500':
    llcrnrlon = -125; llcrnrlat = 25
    urcrnrlon = -108.; urcrnrlat = 42
elif cyyyymmddhh == '2025120300':
    llcrnrlon = -112; llcrnrlat = 33.
    urcrnrlon = -90; urcrnrlat = 48.
else:
    llcrnrlon = -125; llcrnrlat = 24.
    urcrnrlon = -66.;  urcrnrlat = 50.

plot_4panel(lat_1, lat_2, lat_0, lon_0, lons, lats,
            llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat,
            cyyyymmddhh, clead,
            precip_6h_GRAF, prob_0p25mm, prob_2p5mm, prob_10mm,
            GRAF_plot_dir)
