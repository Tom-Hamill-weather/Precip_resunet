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
from dateutils import dateshift, daterange
from netCDF4 import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# Path helpers
# =========================================================================

def get_paths():
    if ENVIRONMENT == 'aws':
        base = AWS_BASE_PATH
        return (
            os.path.join(base, 'probs'),
            os.path.join(base, 'MRMS'),
            os.path.join(base, 'relia'),
        )
    base = os.path.expanduser('~/python/resnet_data')
    return (
        os.path.join(base, 'probs'),
        os.path.join(base, 'MRMS'),
        os.path.join(base, 'relia'),
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
# Apply MLP to full domain
# =========================================================================

MLP_BATCH = 131072


def apply_mlp_fulldomain(model, feat_mean, feat_std, params_6h, ny, nx, device):
    npix = ny * nx
    blocks = [params_6h[k].reshape(6, npix).T for k in PARAM_VARS]
    feats  = np.concatenate(blocks, axis=1).astype(np.float32)

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
    base   = quality > 0.5

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
                pthresholds, clead, date_start, date_end, out_path):
    """
    Produce a 3-panel side-by-side reliability figure and save to out_path.
    wspace=0.12 gives panels that are approximately square (both axes span
    0–100 in data units; panel height ≈ 4.18 in, panel width ≈ 4.18 in).
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

        relia   = relia_arr[idx]
        frequse = frequse_arr[idx]
        BSS     = BSS_arr[idx]

        # --- perfect-reliability diagonal ---
        ax.plot([0, 100], [0, 100], '--', color='k', lw=1.0)

        # --- reliability curve ---
        relia_ma   = ma.masked_where(relia < -99., relia)
        cbss_label = f'BSS = {BSS:.3f}' if not np.isnan(BSS) else 'BSS = N/A'
        ax.plot(probability, 100. * relia_ma, 'o-',
                color='RoyalBlue', linewidth=2, label=cbss_label)

        ax.set_xlim(-1, 101)
        ax.set_ylim(-1, 101)
        ax.set_title(panel_label, fontsize=19)
        ax.set_xlabel('Forecast probability (%)', fontsize=14)
        ax.set_ylabel('Observed relative frequency (%)', fontsize=14)
        ax.legend(loc='lower right', fontsize=12)

        # --- frequency-of-usage inset (upper-left of each panel) ---
        ax_in = ax.inset_axes([0.13, 0.65, 0.42, 0.25])
        ax_in.bar(probability, frequse, width=1.5, bottom=1e-5,
                  log=True, color='RoyalBlue', edgecolor='None', align='center')
        ax_in.set_xlim(-5, 105)
        ax_in.set_ylim(1e-4, 1.)
        ax_in.set_title('Frequency of usage', fontsize=10)
        ax_in.set_xlabel('Fcst prob.', fontsize=8)
        ax_in.set_ylabel('Frequency',  fontsize=8)
        ax_in.hlines([1e-3, 0.001, 0.01, 0.1], 0, 100,
                     linestyles='dashed', colors='gray', lw=0.5)
        ax_in.tick_params(labelsize=7)

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved 3-panel figure: {out_path}')

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

    probs_dir, mrms_dir, relia_dir = get_paths()
    os.makedirs(relia_dir, exist_ok=True)

    # Date lists — 6-h stride, Mar / Jun / Sep / Dec 2025
    mar = daterange('2025030100', '2025033118', 6)
    jun = daterange('2025060100', '2025063018', 6)
    sep = daterange('2025090100', '2025093018', 6)
    dec = daterange('2025120100', '2025123118', 6)
    cyyyymmddhh_list = mar + jun + sep + dec
    date_start = cyyyymmddhh_list[0]
    date_end   = cyyyymmddhh_list[-1]

    pthresholds = [0.25, 2.5, 10.0]
    nthresholds = len(pthresholds)
    ncats       = 11

    pick_fname = os.path.join(
        relia_dir,
        f'relia_6h_MLP_3panel_q0.5_{date_start}_to_{date_end}_lead{clead}h.cPick')

    # ------------------------------------------------------------------
    # Load pre-computed statistics if available; otherwise recompute.
    # ------------------------------------------------------------------
    if os.path.exists(pick_fname):
        print(f'Loading saved statistics from:\n  {pick_fname}')
        with open(pick_fname, 'rb') as fh:
            d = cPickle.load(fh)
        probability  = d['probability']
        relia_arr    = d['relia']
        frequse_arr  = d['frequse']
        BSS_arr      = d['BSS']
        pthresholds  = d['pthresholds']
        print(f'  Loaded.  ngood={d["ngood"]}')
    else:
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
        ngood = 0

        for idate, cdate in enumerate(cyyyymmddhh_list):
            params_6h, lat, lon = read_prob_params_6h(probs_dir, cdate, clead)
            prob_ok = params_6h is not None

            precip_6h, mean_qual, mrms_istat = read_mrms_6h(mrms_dir, cdate, clead)
            mrms_ok = mrms_istat == 0

            ps = 'ok' if prob_ok else 'missing'
            ms = 'ok' if mrms_ok else 'missing'
            print(f'{idate+1:4d}/{ndates}  init={cdate}  params={ps}  mrms={ms}')

            if not prob_ok or not mrms_ok:
                continue

            ny, nx = precip_6h.shape
            ngood += 1

            fz, mw, s1, sc1, s2, sc2 = apply_mlp_fulldomain(
                model, feat_mean, feat_std, params_6h, ny, nx, device)

            for ithresh, thresh in enumerate(pthresholds):
                prob = exceedance_prob(fz, mw, s1, sc1, s2, sc2, thresh)
                ctab, bs, ns, nex, ntot = compute_contab_BS(
                    ny, nx, prob, precip_6h, mean_qual, ncats, thresh)
                contab[ithresh]          += ctab
                BS_sum[ithresh]          += bs
                nsamps_sum[ithresh]      += ns
                nobs_exceed_sum[ithresh] += nex
                nobs_total_sum[ithresh]  += ntot

        if ngood == 0:
            print('\nERROR: No dates with complete data found.')
            sys.exit(1)

        print(f'\n{ngood}/{ndates} init times had complete data.')

        probability  = np.arange(ncats) * 100.0 / float(ncats - 1)
        relia_arr    = np.full((nthresholds, ncats), -99.99)
        frequse_arr  = np.zeros((nthresholds, ncats))
        BSS_arr      = np.full(nthresholds, np.nan)
        BS_arr       = np.full(nthresholds, np.nan)
        BS_climo_arr = np.full(nthresholds, np.nan)
        climo_arr    = np.full(nthresholds, np.nan)

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

        out_dict = {
            'pthresholds':   pthresholds,
            'probability':   probability,
            'ngood':         ngood,
            'relia':         relia_arr,
            'frequse':       frequse_arr,
            'BS':            BS_arr,
            'BS_climo':      BS_climo_arr,
            'BSS':           BSS_arr,
            'climo_freq':    climo_arr,
            'contab':        contab,
            'nsamps':        nsamps_sum,
            'nobs_exceed':   nobs_exceed_sum,
            'nobs_total':    nobs_total_sum,
        }
        with open(pick_fname, 'wb') as fh:
            cPickle.dump(out_dict, fh)
        print(f'Saved statistics to {pick_fname}')

    # 3-panel figure
    plot_fname = (f'Relia_6h_MLP_MRMS_3panel_{date_start}_to_{date_end}'
                  f'_{clead}h.png')
    plot_3panel(probability, relia_arr, frequse_arr, BSS_arr,
                pthresholds, clead, date_start, date_end, plot_fname)


if __name__ == '__main__':
    main()
