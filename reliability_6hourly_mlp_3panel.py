"""
reliability_6hourly_mlp_3panel.py — 3-panel reliability diagram for 6-hourly MLP

Usage:
    python reliability_6hourly_mlp_3panel.py <clead>

    clead : integer lead time (hours) for the END of the 6-h window.
            A trained checkpoint must exist at:
                mlp_trainings/6h_mlp_lead{clead}h.pth

Identical to reliability_6hourly_mlp.py except that the three thresholds
(0.25, 2.5, 10.0 mm) are plotted as side-by-side panels in a single figure.

Tom Hamill, May 2026
"""

import os
import sys
import math
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import _pickle as cPickle
from scipy.special import gammainc
from scipy.stats import ttest_rel, wilcoxon
from dateutils import dateshift, daterange, splitdate, dayofyear
from netCDF4 import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F


def julian_features(cyyyymmddhh):
    """Cyclic day-of-year encoding: cos/sin(2*pi*julian_day/365). Must match
    sample_6hourly_prob_mrms.py exactly."""
    yyyy, mm, dd, hh = splitdate(cyyyymmddhh)
    doy = dayofyear(yyyy, mm, dd)
    angle = 2.0 * math.pi * doy / 365.0
    return math.cos(angle), math.sin(angle)

np.set_printoptions(precision=3, suppress=True)

# =========================================================================
# Environment detection
# =========================================================================

def detect_environment():
    for path in ['/data/resnet_data', '/data2/resnet_data']:
        if os.path.exists(path):
            print(f'Detected AWS environment ({path})')
            return 'aws', path
    print('Detected laptop environment')
    return 'laptop', None

ENVIRONMENT, AWS_BASE_PATH = detect_environment()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================================
# MLP model (must match train_6hourly_mlp.py exactly)
# =========================================================================

SHAPE_MIN    = 0.1
SCALE_MIN    = 0.01
HIDDEN_SIZES = [72, 144, 72, 36, 12]


class GammaMixtureMLP(nn.Module):
    def __init__(self, hidden_sizes=HIDDEN_SIZES,
                 shape_min=SHAPE_MIN, scale_min=SCALE_MIN, n_input=38,
                 min_separation=0.5):
        super().__init__()
        self.shape_min = shape_min
        self.scale_min = scale_min
        self.min_separation = min_separation
        layer_sizes = [n_input] + hidden_sizes
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
        # Hard ordering constraint (must match train_6hourly_mlp.py exactly):
        # shape2 = shape1 + softplus(offset) + min_separation.
        shape2_offset = F.softplus(raw[:, 4])
        shape2        = shape1 + shape2_offset + self.min_separation
        scale2        = self.scale_min + F.softplus(raw[:, 5])
        return frac_zero, mix_weight, shape1, scale1, shape2, scale2

# =========================================================================
# Checkpoint loader
# =========================================================================

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
    n_input      = ckpt.get('n_input',      38)

    model = GammaMixtureMLP(hidden_sizes=hidden_sizes,
                            shape_min=shape_min, scale_min=scale_min,
                            n_input=n_input)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    feat_mean = ckpt['feature_mean']
    feat_std  = ckpt['feature_std']
    print(f'Loaded MLP from {ckpt_path}  (epoch {ckpt["epoch"]+1})')
    return model, feat_mean, feat_std

# =========================================================================
# Path helpers
# =========================================================================

def get_paths():
    if ENVIRONMENT == 'aws':
        base = AWS_BASE_PATH
        return (
            os.path.join(base, 'probs'),
            os.path.join(base, 'MRMS'),
            os.path.join(base, 'relia'),
            os.path.join(base, 'probs_control'),
        )
    base = os.path.expanduser('~/python/resnet_data')
    return (
        os.path.join(base, 'probs'),
        os.path.join(base, 'MRMS'),
        os.path.join(base, 'relia'),
        os.path.join(base, 'probs_control'),
    )

# =========================================================================
# Read 6 consecutive hourly gamma-mixture parameter files
# =========================================================================

PARAM_VARS = [
    'fraction_zero', 'mixture_weight',
    'gamma_shape1',  'gamma_scale1',
    'gamma_shape2',  'gamma_scale2',
]


def read_prob_params_6h(probs_dir, cyyyymmddhh, clead):
    lead_times = list(range(clead - 5, clead + 1))
    stacks = {k: [] for k in PARAM_VARS}
    lat = lon = None

    for lt in lead_times:
        fname = os.path.join(probs_dir,
                             f'{cyyyymmddhh}_{lt}_probs_gamma_mixture.nc')
        if not os.path.exists(fname):
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
        stacks[k] = np.stack(stacks[k], axis=0)

    return stacks, lat, lon

# =========================================================================
# Read precomputed independence-assumption ensemble control
# (written by generate_6h_independence_control.py)
# =========================================================================

CONTROL_VARNAME_BY_THRESH = {0.25: 'prob_0p25mm', 2.5: 'prob_2p5mm', 10.0: 'prob_10mm'}


def read_control_probs_6h(control_dir, cyyyymmddhh, clead, pthresholds):
    fname = os.path.join(control_dir,
                         f'{cyyyymmddhh}_{clead}_indep_ensemble_probs.nc')
    if not os.path.exists(fname):
        return None
    try:
        with Dataset(fname, 'r') as ds:
            probs = {}
            for t in pthresholds:
                probs[t] = ds.variables[CONTROL_VARNAME_BY_THRESH[t]][:].data.astype(np.float32)
    except Exception as exc:
        print(f'  WARNING: cannot read {fname}: {exc}')
        return None
    return probs


def read_copula_control_probs_6h(control_dir, cyyyymmddhh, clead, pthresholds):
    """Same format as read_control_probs_6h, but for the conditional-copula
    control (generate_6h_conditional_copula_control.py)."""
    fname = os.path.join(control_dir,
                         f'{cyyyymmddhh}_{clead}_copula_ensemble_probs.nc')
    if not os.path.exists(fname):
        return None
    try:
        with Dataset(fname, 'r') as ds:
            probs = {}
            for t in pthresholds:
                probs[t] = ds.variables[CONTROL_VARNAME_BY_THRESH[t]][:].data.astype(np.float32)
    except Exception as exc:
        print(f'  WARNING: cannot read {fname}: {exc}')
        return None
    return probs

# =========================================================================
# Apply MLP to full domain
# =========================================================================

MLP_BATCH = 131072


def apply_mlp_fulldomain(model, feat_mean, feat_std, params_6h, ny, nx, device,
                          cos_doy, sin_doy):
    npix = ny * nx
    blocks = [params_6h[k].reshape(6, npix).T for k in PARAM_VARS]
    hourly_feats   = np.concatenate(blocks, axis=1).astype(np.float32)
    seasonal_feats = np.tile(np.array([cos_doy, sin_doy], dtype=np.float32), (npix, 1))
    feats = np.concatenate([hourly_feats, seasonal_feats], axis=1)

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
    return tuple(result)

# =========================================================================
# P(6h >= threshold) from zero-inflated 2-component Gamma mixture
# =========================================================================

def exceedance_prob(frac_zero, mix_weight, shape1, scale1, shape2, scale2, threshold):
    if threshold <= 0.0:
        return np.clip(1.0 - frac_zero, 0.0, 1.0)

    eps = 1e-7
    s1  = np.maximum(shape1, eps)
    sc1 = np.maximum(scale1, eps)
    s2  = np.maximum(shape2, eps)
    sc2 = np.maximum(scale2, eps)

    sf1 = 1.0 - gammainc(s1, threshold / sc1)
    sf2 = 1.0 - gammainc(s2, threshold / sc2)

    mw = np.clip(mix_weight, 0.0, 1.0)
    p_nonzero = np.clip(1.0 - frac_zero, 0.0, 1.0)
    return np.clip(p_nonzero * (mw * sf1 + (1.0 - mw) * sf2), 0.0, 1.0)

# =========================================================================
# Read 6 hourly MRMS files → 6-h accumulation and mean quality
# =========================================================================

def read_mrms_6h(mrms_dir, cyyyymmddhh, clead):
    lead_times  = list(range(clead - 5, clead + 1))
    precip_list = []
    quality_list = []

    for lt in lead_times:
        verif_time = dateshift(cyyyymmddhh, lt)
        cyyyymm    = verif_time[:6]
        fname = os.path.join(mrms_dir, cyyyymm,
                             f'MRMS_1h_pamt_and_data_qual_{verif_time}.nc')
        if not os.path.exists(fname):
            return None, None, -1
        try:
            with Dataset(fname, 'r') as ds:
                precip  = ds.variables['precipitation'][:].data.astype(np.float32)
                quality = ds.variables['data_quality'][:].data.astype(np.float32)
            precip_list.append(precip)
            quality_list.append(quality)
        except Exception as exc:
            print(f'  WARNING: cannot read {fname}: {exc}')
            return None, None, -1

    precip_6h    = np.stack(precip_list,  axis=0).sum(axis=0)
    mean_quality = np.stack(quality_list, axis=0).mean(axis=0)
    return precip_6h, mean_quality, 0

# =========================================================================
# Contingency table / Brier Score accumulation
# =========================================================================

def compute_contab_BS(ny, nx, prob, obs, quality, ncats, threshold):
    contab = np.zeros((ncats, 2), dtype=np.int64)
    base   = quality > 0.6

    binary_obs = -1 * np.ones((ny, nx), dtype=np.int8)
    a = np.where(np.logical_and(base,
            np.logical_and(obs >= threshold, obs <= 200.0)))
    binary_obs[a] = 1
    a = np.where(np.logical_and(base,
            np.logical_and(obs >= 0.0,
            np.logical_and(obs < threshold, obs <= 200.0))))
    binary_obs[a] = 0

    for icat in range(ncats):
        pmin = max(0.0, float(icat) / (ncats - 1) - 0.5 / (ncats - 1))
        pmax = min(1.0, float(icat) / (ncats - 1) + 0.5 / (ncats - 1))
        in_bin = (prob >= pmin) & (prob < pmax if icat < ncats - 1 else prob <= pmax)
        contab[icat, 1] += int(np.sum(in_bin & (binary_obs == 1)))
        contab[icat, 0] += int(np.sum(in_bin & (binary_obs == 0)))

    good_0 = np.where(binary_obs == 0)
    good_1 = np.where(binary_obs == 1)
    BS = float(np.sum(prob[good_0] ** 2) + np.sum((1.0 - prob[good_1]) ** 2))
    nsamps     = len(good_0[0]) + len(good_1[0])
    nobs_exceed = len(good_1[0])
    nobs_total  = nsamps
    return contab, BS, nsamps, nobs_exceed, nobs_total


def compute_relia(contab, ncats):
    frequse = np.zeros(ncats, dtype=float)
    relia   = np.full(ncats, -99.99)
    total   = float(np.sum(contab))
    for icat in range(ncats):
        n = np.sum(contab[icat, :])
        frequse[icat] = n / total if total > 0 else 0.0
        if n > 5:
            relia[icat] = float(contab[icat, 1]) / n
    return frequse, relia

# =========================================================================
# 3-panel reliability plot
# =========================================================================

PANEL_LABELS = [
    r'(a) $\geq$ 0.25 mm/6h',
    r'(b) $\geq$ 2.5 mm/6h',
    r'(c) $\geq$ 10 mm/6h',
]


def plot_3panel(probability, relia_arr, frequse_arr, BSS_arr,
                relia_control_arr, frequse_control_arr, BSS_control_arr,
                pthresholds, clead, date_start, date_end, out_path):
    """
    Produce a 3-panel side-by-side reliability figure and save to out_path.
    wspace=0.12 gives panels that are approximately square (both axes span
    0–100 in data units; panel height ≈ 4.18 in, panel width ≈ 4.18 in).

    Each panel shows the MLP reliability curve alongside the naive
    independence-assumption ensemble control (sum of independent hourly
    draws).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.12, top=0.88, wspace=0.12)

    # Shrink each axes box by 0.02 in both x and y (figure-fraction units),
    # anchoring at the bottom-left so the freed space falls on the right and
    # top — the right-side gap gives room for adjacent y-axis labels.
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width - 0.02, pos.height - 0.02])

    for idx, (ax, thresh, panel_label) in enumerate(
            zip(axes, pthresholds, PANEL_LABELS)):

        relia           = relia_arr[idx]
        frequse         = frequse_arr[idx]
        BSS             = BSS_arr[idx]
        relia_control   = relia_control_arr[idx]
        frequse_control = frequse_control_arr[idx]
        BSS_control     = BSS_control_arr[idx]

        # --- perfect-reliability diagonal ---
        ax.plot([0, 100], [0, 100], '--', color='k', lw=1.0)

        # --- independence-assumption control curve ---
        relia_control_ma = ma.masked_where(relia_control < -99., relia_control)
        cbss_control_label = (f'Indep. ensemble (BSS = {BSS_control:.3f})'
                              if not np.isnan(BSS_control) else 'Indep. ensemble (BSS = N/A)')
        ax.plot(probability, 100. * relia_control_ma, 's-',
                color='red', linewidth=2, label=cbss_control_label)

        # --- reliability curve ---
        relia_ma   = ma.masked_where(relia < -99., relia)
        cbss_label = f'MLP (BSS = {BSS:.3f})' if not np.isnan(BSS) else 'MLP (BSS = N/A)'
        ax.plot(probability, 100. * relia_ma, 'o-',
                color='RoyalBlue', linewidth=2, label=cbss_label)

        ax.set_xlim(-1, 101)
        ax.set_ylim(-1, 101)
        ax.set_title(panel_label, fontsize=19)
        ax.set_xlabel('Forecast probability (%)', fontsize=14)
        ax.set_ylabel('Observed relative frequency (%)', fontsize=14)
        ax.legend(loc='lower right', fontsize=11)

        # --- frequency-of-usage inset (upper-left of each panel) ---
        # Bars are offset left/mid/right of the bin center so the three
        # series sit side by side instead of overlapping.
        bar_offset = 1.3
        bar_width  = 2.4
        ax_in = ax.inset_axes([0.13, 0.65, 0.42, 0.25])
        ax_in.bar(probability - bar_offset, frequse_control, width=bar_width, bottom=1e-5,
                  log=True, color='red', edgecolor='None', align='center',
                  alpha=0.6, label='Indep.')
        ax_in.bar(probability + bar_offset, frequse, width=bar_width, bottom=1e-5,
                  log=True, color='RoyalBlue', edgecolor='None', align='center',
                  alpha=0.6, label='MLP')
        ax_in.set_xlim(-5, 105)
        ax_in.set_ylim(1e-4, 1.)
        ax_in.set_title('Frequency of usage', fontsize=10)
        ax_in.set_xlabel('Fcst prob.', fontsize=8)
        ax_in.set_ylabel('Frequency',  fontsize=8)
        ax_in.hlines([1e-3, 0.001, 0.01, 0.1], 0, 100,
                     linestyles='dashed', colors='gray', lw=0.5)
        ax_in.tick_params(labelsize=7)
        ax_in.legend(loc='upper right', fontsize=6)

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved 3-panel figure: {out_path}')

# =========================================================================
# Out-of-sample test date list: day 10 through month-end, all 12 months
# of 2025, 00Z/12Z only.  Days 1-7 of each month are used for MLP training
# (see sample_6hourly_prob_mrms.py) and days 8-9 are a gap left out
# entirely, so persistent synoptic systems can't leak across the
# train/test boundary.
# =========================================================================

MONTHS_2025 = [
    (1, 31), (2, 28), (3, 31), (4, 30), (5, 31), (6, 30),
    (7, 31), (8, 31), (9, 30), (10, 31), (11, 30), (12, 31),
]
TEST_DAY_START = 10   # days 1-7 train, 8-9 gap, 10-end test


def build_test_datelist():
    date_list = []
    for mm, ndays in MONTHS_2025:
        for dd in range(TEST_DAY_START, ndays + 1):
            date_list.append(f'2025{mm:02d}{dd:02d}00')
            date_list.append(f'2025{mm:02d}{dd:02d}12')
    return date_list


# =========================================================================
# Main
# =========================================================================

def main():
    if len(sys.argv) != 2:
        print('Usage: python reliability_6hourly_mlp_3panel.py <clead>')
        sys.exit(1)

    clead = int(sys.argv[1])
    if clead < 6:
        print('ERROR: clead must be >= 6')
        sys.exit(1)

    print(f'reliability_6hourly_mlp_3panel.py  clead={clead}h  '
          f'(6-h window: {clead-5}–{clead} h)')

    probs_dir, mrms_dir, relia_dir, control_dir = get_paths()
    os.makedirs(relia_dir, exist_ok=True)

    # Date list — out-of-sample test months (see build_test_datelist)
    cyyyymmddhh_list = build_test_datelist()
    date_start = cyyyymmddhh_list[0]
    date_end   = cyyyymmddhh_list[-1]

    pthresholds = [0.25, 2.5, 10.0]
    nthresholds = len(pthresholds)
    ncats       = 11

    pick_fname = os.path.join(
        relia_dir,
        f'relia_6h_MLP_3panel_q0.6_{date_start}_to_{date_end}_lead{clead}h.cPick')

    # ------------------------------------------------------------------
    # Load pre-computed statistics if available; otherwise recompute.
    # ------------------------------------------------------------------
    if os.path.exists(pick_fname):
        with open(pick_fname, 'rb') as fh:
            d = cPickle.load(fh)
    else:
        d = {}

    if 'ttest_p' in d:
        print(f'Loading saved statistics from:\n  {pick_fname}')
        probability         = d['probability']
        relia_arr           = d['relia']
        frequse_arr         = d['frequse']
        BSS_arr             = d['BSS']
        relia_control_arr   = d['relia_control']
        frequse_control_arr = d['frequse_control']
        BSS_control_arr     = d['BSS_control']
        relia_copula_arr    = d['relia_copula']
        frequse_copula_arr  = d['frequse_copula']
        BSS_copula_arr      = d['BSS_copula']
        n_days_arr          = d['n_days_sig']
        ttest_p_arr         = d['ttest_p']
        wilcoxon_p_arr      = d['wilcoxon_p']
        pthresholds         = d['pthresholds']
        print(f'  Loaded.  ngood={d["ngood"]}')
    else:
        if os.path.exists(pick_fname):
            print(f'Cached statistics at {pick_fname} lack the paired-significance-test '
                  f'fields; recomputing.')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Torch device: {device}')
        model, feat_mean, feat_std = load_mlp(clead, device)

        ndates = len(cyyyymmddhh_list)
        print(f'Total init times to process: {ndates}')

        contab          = np.zeros((nthresholds, ncats, 2), dtype=np.int64)
        BS_sum          = np.zeros(nthresholds, dtype=float)
        nsamps_sum      = np.zeros(nthresholds, dtype=float)
        nobs_exceed_sum = np.zeros(nthresholds, dtype=float)
        nobs_total_sum  = np.zeros(nthresholds, dtype=float)

        contab_control          = np.zeros((nthresholds, ncats, 2), dtype=np.int64)
        BS_sum_control          = np.zeros(nthresholds, dtype=float)
        nsamps_sum_control      = np.zeros(nthresholds, dtype=float)
        nobs_exceed_sum_control = np.zeros(nthresholds, dtype=float)
        nobs_total_sum_control  = np.zeros(nthresholds, dtype=float)

        contab_copula          = np.zeros((nthresholds, ncats, 2), dtype=np.int64)
        BS_sum_copula          = np.zeros(nthresholds, dtype=float)
        nsamps_sum_copula      = np.zeros(nthresholds, dtype=float)
        nobs_exceed_sum_copula = np.zeros(nthresholds, dtype=float)
        nobs_total_sum_copula  = np.zeros(nthresholds, dtype=float)
        ngood = 0

        # Per-case-day mean Brier Score, one value per verification date per
        # threshold, for the MLP and the control.  Following Hamill (1999,
        # Wea. Forecasting), all grid points within a case day are pooled into
        # a single per-day score before testing, rather than treating every
        # grid point as an independent sample, since nearby grid points are
        # spatially correlated within a synoptic event.  These per-day series
        # are the basis for the paired significance test reported in Sec. 4.
        daily_BS_mlp     = [[] for _ in range(nthresholds)]
        daily_BS_control = [[] for _ in range(nthresholds)]

        for idate, cdate in enumerate(cyyyymmddhh_list):
            params_6h, lat, lon = read_prob_params_6h(probs_dir, cdate, clead)
            prob_ok = params_6h is not None

            precip_6h, mean_qual, mrms_istat = read_mrms_6h(mrms_dir, cdate, clead)
            mrms_ok = mrms_istat == 0

            control_probs = read_control_probs_6h(control_dir, cdate, clead, pthresholds)
            control_ok = control_probs is not None

            copula_probs = read_copula_control_probs_6h(control_dir, cdate, clead, pthresholds)
            copula_ok = copula_probs is not None

            ps = 'ok' if prob_ok else 'missing'
            ms = 'ok' if mrms_ok else 'missing'
            cs = 'ok' if control_ok else 'missing'
            cps = 'ok' if copula_ok else 'missing'
            print(f'{idate+1:4d}/{ndates}  init={cdate}  params={ps}  mrms={ms}  '
                  f'control={cs}  copula={cps}')

            if not prob_ok or not mrms_ok or not control_ok or not copula_ok:
                continue

            ny, nx = precip_6h.shape
            ngood += 1

            cos_doy, sin_doy = julian_features(cdate)
            fz, mw, s1, sc1, s2, sc2 = apply_mlp_fulldomain(
                model, feat_mean, feat_std, params_6h, ny, nx, device,
                cos_doy, sin_doy)

            for ithresh, thresh in enumerate(pthresholds):
                prob = exceedance_prob(fz, mw, s1, sc1, s2, sc2, thresh)
                ctab, bs, ns, nex, ntot = compute_contab_BS(
                    ny, nx, prob, precip_6h, mean_qual, ncats, thresh)
                contab[ithresh]          += ctab
                BS_sum[ithresh]          += bs
                nsamps_sum[ithresh]      += ns
                nobs_exceed_sum[ithresh] += nex
                nobs_total_sum[ithresh]  += ntot

                ctab_c, bs_c, ns_c, nex_c, ntot_c = compute_contab_BS(
                    ny, nx, control_probs[thresh], precip_6h, mean_qual, ncats, thresh)
                contab_control[ithresh]          += ctab_c
                BS_sum_control[ithresh]          += bs_c
                nsamps_sum_control[ithresh]      += ns_c
                nobs_exceed_sum_control[ithresh] += nex_c
                nobs_total_sum_control[ithresh]  += ntot_c

                if ns > 0 and ns_c > 0:
                    daily_BS_mlp[ithresh].append(bs / ns)
                    daily_BS_control[ithresh].append(bs_c / ns_c)

                ctab_p, bs_p, ns_p, nex_p, ntot_p = compute_contab_BS(
                    ny, nx, copula_probs[thresh], precip_6h, mean_qual, ncats, thresh)
                contab_copula[ithresh]          += ctab_p
                BS_sum_copula[ithresh]          += bs_p
                nsamps_sum_copula[ithresh]      += ns_p
                nobs_exceed_sum_copula[ithresh] += nex_p
                nobs_total_sum_copula[ithresh]  += ntot_p

        if ngood == 0:
            print('\nERROR: No dates with complete data found.')
            sys.exit(1)

        print(f'\n{ngood}/{ndates} init times had complete data.')

        probability          = np.arange(ncats) * 100.0 / float(ncats - 1)
        relia_arr            = np.full((nthresholds, ncats), -99.99)
        frequse_arr          = np.zeros((nthresholds, ncats))
        BSS_arr              = np.full(nthresholds, np.nan)
        BS_arr               = np.full(nthresholds, np.nan)
        BS_climo_arr         = np.full(nthresholds, np.nan)
        climo_arr            = np.full(nthresholds, np.nan)

        relia_control_arr    = np.full((nthresholds, ncats), -99.99)
        frequse_control_arr  = np.zeros((nthresholds, ncats))
        BSS_control_arr      = np.full(nthresholds, np.nan)
        BS_control_arr       = np.full(nthresholds, np.nan)

        relia_copula_arr     = np.full((nthresholds, ncats), -99.99)
        frequse_copula_arr   = np.zeros((nthresholds, ncats))
        BSS_copula_arr       = np.full(nthresholds, np.nan)
        BS_copula_arr        = np.full(nthresholds, np.nan)

        n_days_arr      = np.zeros(nthresholds, dtype=int)
        ttest_p_arr     = np.full(nthresholds, np.nan)
        wilcoxon_p_arr  = np.full(nthresholds, np.nan)

        for ithresh, thresh in enumerate(pthresholds):
            if nsamps_sum[ithresh] == 0:
                print(f'  thresh={thresh} mm: no valid samples')
                continue

            BS_mean    = BS_sum[ithresh] / nsamps_sum[ithresh]
            climo_freq = (nobs_exceed_sum[ithresh] / nobs_total_sum[ithresh]
                          if nobs_total_sum[ithresh] > 0 else np.nan)
            BS_climo   = climo_freq * (1.0 - climo_freq) if not np.isnan(climo_freq) else np.nan
            BSS        = (1.0 - BS_mean / BS_climo
                          if (not np.isnan(BS_climo) and BS_climo > 0) else np.nan)

            frequse, relia = compute_relia(contab[ithresh], ncats)

            relia_arr[ithresh]    = relia
            frequse_arr[ithresh]  = frequse
            BS_arr[ithresh]       = BS_mean
            BS_climo_arr[ithresh] = BS_climo
            BSS_arr[ithresh]      = BSS
            climo_arr[ithresh]    = climo_freq

            cbss = f'{BSS:.3f}' if not np.isnan(BSS) else 'N/A'
            print(f'  thresh={thresh:5.2f} mm | climo={climo_freq:.4f}  '
                  f'BS={BS_mean:.5f}  BS_climo={BS_climo:.5f}  BSS={cbss}')

            # --- independence-assumption ensemble control (same climo/obs) ---
            BS_mean_c = (BS_sum_control[ithresh] / nsamps_sum_control[ithresh]
                        if nsamps_sum_control[ithresh] > 0 else np.nan)
            BSS_c     = (1.0 - BS_mean_c / BS_climo
                        if (not np.isnan(BS_climo) and BS_climo > 0
                            and not np.isnan(BS_mean_c)) else np.nan)

            frequse_c, relia_c = compute_relia(contab_control[ithresh], ncats)

            relia_control_arr[ithresh]   = relia_c
            frequse_control_arr[ithresh] = frequse_c
            BS_control_arr[ithresh]      = BS_mean_c
            BSS_control_arr[ithresh]     = BSS_c

            cbss_c = f'{BSS_c:.3f}' if not np.isnan(BSS_c) else 'N/A'
            print(f'    [control] BS={BS_mean_c:.5f}  BSS={cbss_c}')

            # --- MLP-vs-control significance test, following Hamill (1999,
            # Wea. Forecasting, 14, 155-167): per-case-day mean Brier Score
            # (pooling all grid points into one score per day, to avoid
            # treating spatially correlated grid points as independent
            # samples), then a paired t test and Wilcoxon signed-rank test on
            # the day-by-day MLP-minus-control differences. ---
            d_mlp = np.asarray(daily_BS_mlp[ithresh])
            d_ctl = np.asarray(daily_BS_control[ithresh])
            n_days = len(d_mlp)
            if n_days > 1 and not np.allclose(d_mlp, d_ctl):
                ttest_p    = float(ttest_rel(d_mlp, d_ctl).pvalue)
                wilcoxon_p = float(wilcoxon(d_mlp, d_ctl).pvalue)
            else:
                ttest_p    = np.nan
                wilcoxon_p = np.nan
            n_days_arr[ithresh]     = n_days
            ttest_p_arr[ithresh]    = ttest_p
            wilcoxon_p_arr[ithresh] = wilcoxon_p
            print(f'    [sig. test] n_days={n_days}  paired-t p={ttest_p:.4f}  '
                  f'Wilcoxon p={wilcoxon_p:.4f}')

            # --- conditional-copula ensemble control (same climo/obs) ---
            BS_mean_p = (BS_sum_copula[ithresh] / nsamps_sum_copula[ithresh]
                        if nsamps_sum_copula[ithresh] > 0 else np.nan)
            BSS_p     = (1.0 - BS_mean_p / BS_climo
                        if (not np.isnan(BS_climo) and BS_climo > 0
                            and not np.isnan(BS_mean_p)) else np.nan)

            frequse_p, relia_p = compute_relia(contab_copula[ithresh], ncats)

            relia_copula_arr[ithresh]   = relia_p
            frequse_copula_arr[ithresh] = frequse_p
            BS_copula_arr[ithresh]      = BS_mean_p
            BSS_copula_arr[ithresh]     = BSS_p

            cbss_p = f'{BSS_p:.3f}' if not np.isnan(BSS_p) else 'N/A'
            print(f'    [copula]  BS={BS_mean_p:.5f}  BSS={cbss_p}')

        out_dict = {
            'pthresholds':      pthresholds,
            'probability':      probability,
            'ngood':            ngood,
            'relia':            relia_arr,
            'frequse':          frequse_arr,
            'BS':               BS_arr,
            'BS_climo':         BS_climo_arr,
            'BSS':              BSS_arr,
            'climo_freq':       climo_arr,
            'contab':           contab,
            'nsamps':           nsamps_sum,
            'nobs_exceed':      nobs_exceed_sum,
            'nobs_total':       nobs_total_sum,
            'relia_control':    relia_control_arr,
            'frequse_control':  frequse_control_arr,
            'BS_control':       BS_control_arr,
            'BSS_control':      BSS_control_arr,
            'contab_control':   contab_control,
            'nsamps_control':   nsamps_sum_control,
            'relia_copula':     relia_copula_arr,
            'frequse_copula':   frequse_copula_arr,
            'BS_copula':        BS_copula_arr,
            'BSS_copula':       BSS_copula_arr,
            'contab_copula':    contab_copula,
            'nsamps_copula':    nsamps_sum_copula,
            'n_days_sig':       n_days_arr,
            'ttest_p':          ttest_p_arr,
            'wilcoxon_p':       wilcoxon_p_arr,
        }
        with open(pick_fname, 'wb') as fh:
            cPickle.dump(out_dict, fh)
        print(f'Saved statistics to {pick_fname}')

    # 3-panel figure
    plot_fname = (f'Relia_6h_MLP_MRMS_3panel_{date_start}_to_{date_end}'
                  f'_{clead}h.png')
    plot_3panel(probability, relia_arr, frequse_arr, BSS_arr,
                relia_control_arr, frequse_control_arr, BSS_control_arr,
                pthresholds, clead, date_start, date_end, plot_fname)


if __name__ == '__main__':
    main()
