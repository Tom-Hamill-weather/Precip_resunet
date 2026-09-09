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


def hour_of_day_features(cyyyymmddhh):
    """Cyclic GRAF init-cycle encoding: cos/sin(2*pi*hour/24). Must match
    train_6hourly_mlp.py's load_data() exactly."""
    hour = int(cyyyymmddhh[-2:])
    angle = 2.0 * math.pi * hour / 24.0
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


FZ_HEAD_HIDDEN = 16


class GammaMixtureMLP(nn.Module):
    def __init__(self, hidden_sizes=HIDDEN_SIZES,
                 shape_min=SHAPE_MIN, scale_min=SCALE_MIN, n_input=38,
                 min_separation=0.5, dedicated_fz_head=False, fz_head_hidden=FZ_HEAD_HIDDEN):
        super().__init__()
        self.shape_min = shape_min
        self.scale_min = scale_min
        self.min_separation = min_separation
        self.dedicated_fz_head = dedicated_fz_head
        layer_sizes = [n_input] + hidden_sizes
        trunk_layers = []
        for in_sz, out_sz in zip(layer_sizes, layer_sizes[1:]):
            trunk_layers += [nn.Linear(in_sz, out_sz),
                             nn.BatchNorm1d(out_sz),
                             nn.ReLU(),
                             nn.Dropout(0.15)]  # no-op in eval() mode; kept only so
                                                # Sequential indices (and therefore
                                                # state_dict keys) match train_6hourly_mlp.py
        if dedicated_fz_head:
            self.trunk = nn.Sequential(*trunk_layers)
            self.main_head = nn.Linear(hidden_sizes[-1], 5)
            self.fz_head = nn.Sequential(
                nn.Linear(hidden_sizes[-1], fz_head_hidden), nn.ReLU(),
                nn.Linear(fz_head_hidden, 1))
        else:
            trunk_layers.append(nn.Linear(hidden_sizes[-1], 6))
            self.net = nn.Sequential(*trunk_layers)

    def forward(self, x):
        if self.dedicated_fz_head:
            h = self.trunk(x)
            frac_zero  = torch.sigmoid(self.fz_head(h).squeeze(-1))
            main_raw   = self.main_head(h)
            mix_weight = torch.sigmoid(main_raw[:, 0])
            shape1     = self.shape_min + F.softplus(main_raw[:, 1])
            scale1     = self.scale_min + F.softplus(main_raw[:, 2])
            shape2_offset = F.softplus(main_raw[:, 3])
            shape2        = shape1 + shape2_offset + self.min_separation
            scale2        = self.scale_min + F.softplus(main_raw[:, 4])
            return frac_zero, mix_weight, shape1, scale1, shape2, scale2

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


FILM_HIDDEN = 16


class GammaMixtureMLPFiLM(nn.Module):
    """Must match train_6hourly_mlp.py's GammaMixtureMLPFiLM exactly."""

    def __init__(self, hidden_sizes=HIDDEN_SIZES,
                 shape_min=SHAPE_MIN, scale_min=SCALE_MIN, n_input=39,
                 min_separation=0.5, film_hidden=FILM_HIDDEN):
        super().__init__()
        self.shape_min = shape_min
        self.scale_min = scale_min
        self.min_separation = min_separation
        self.trunk_input = n_input - 1

        layer_sizes = [self.trunk_input] + hidden_sizes
        self.linears = nn.ModuleList()
        self.bns     = nn.ModuleList()
        for in_sz, out_sz in zip(layer_sizes, layer_sizes[1:]):
            self.linears.append(nn.Linear(in_sz, out_sz))
            self.bns.append(nn.BatchNorm1d(out_sz))
        self.output_layer = nn.Linear(hidden_sizes[-1], 6)

        self.film_body = nn.Sequential(nn.Linear(1, film_hidden), nn.ReLU())
        self.film_heads = nn.ModuleList()
        for out_sz in hidden_sizes:
            head = nn.Linear(film_hidden, 2 * out_sz)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.film_heads.append(head)

    def forward(self, x):
        terrain   = x[:, self.trunk_input:self.trunk_input + 1]
        h         = x[:, :self.trunk_input]
        film_feat = self.film_body(terrain)

        for linear, bn, head in zip(self.linears, self.bns, self.film_heads):
            h = linear(h)
            h = bn(h)
            gamma, beta = head(film_feat).chunk(2, dim=1)
            h = (1.0 + gamma) * h + beta
            h = F.relu(h)

        raw = self.output_layer(h)
        frac_zero  = torch.sigmoid(raw[:, 0])
        mix_weight = torch.sigmoid(raw[:, 1])
        shape1     = self.shape_min + F.softplus(raw[:, 2])
        scale1     = self.scale_min + F.softplus(raw[:, 3])
        shape2_offset = F.softplus(raw[:, 4])
        shape2        = shape1 + shape2_offset + self.min_separation
        scale2        = self.scale_min + F.softplus(raw[:, 5])
        return frac_zero, mix_weight, shape1, scale1, shape2, scale2


GRU_HIDDEN = 48
N_HOURLY_FEATS_PER_HOUR = 9

# Literal column indices, verified identical to train_6hourly_mlp.py's
# _gru_column_indices(73) output (that file derives these from FEATURE_VARS/
# TEXTURE_SPATIAL_VARS, which aren't duplicated here -- see this class's
# docstring). Must match load_data()'s/apply_mlp_fulldomain()'s fixed
# column-block order if that ever changes.
_GRU_HOURLY_IDX = [
    [0, 6, 12, 18, 24, 30, 41, 47, 53],
    [1, 7, 13, 19, 25, 31, 42, 48, 54],
    [2, 8, 14, 20, 26, 32, 43, 49, 55],
    [3, 9, 15, 21, 27, 33, 44, 50, 56],
    [4, 10, 16, 22, 28, 34, 45, 51, 57],
    [5, 11, 17, 23, 29, 35, 46, 52, 58],
]
_GRU_STATIC_IDX = [36, 37, 38, 39, 40, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]


class GammaMixtureGRU(nn.Module):
    """Must match train_6hourly_mlp.py's GammaMixtureGRU exactly. Requires n_input==73."""

    def __init__(self, hidden_sizes=HIDDEN_SIZES,
                 shape_min=SHAPE_MIN, scale_min=SCALE_MIN, n_input=73,
                 min_separation=0.5, gru_hidden=GRU_HIDDEN,
                 dedicated_fz_head=False, fz_head_hidden=FZ_HEAD_HIDDEN):
        super().__init__()
        if n_input != 73:
            raise ValueError(f'GammaMixtureGRU requires n_input==73; got {n_input}')
        self.shape_min = shape_min
        self.scale_min = scale_min
        self.min_separation = min_separation
        self.dedicated_fz_head = dedicated_fz_head

        self.register_buffer('hourly_idx', torch.tensor(_GRU_HOURLY_IDX, dtype=torch.long))
        self.register_buffer('static_idx', torch.tensor(_GRU_STATIC_IDX, dtype=torch.long))

        self.gru = nn.GRU(input_size=N_HOURLY_FEATS_PER_HOUR, hidden_size=gru_hidden,
                          num_layers=1, batch_first=True)

        n_static = len(_GRU_STATIC_IDX)
        layer_sizes = [gru_hidden + n_static] + hidden_sizes
        trunk_layers = []
        for in_sz, out_sz in zip(layer_sizes, layer_sizes[1:]):
            trunk_layers += [nn.Linear(in_sz, out_sz),
                             nn.BatchNorm1d(out_sz),
                             nn.ReLU(),
                             nn.Dropout(0.15)]  # no-op in eval() mode; kept for state_dict key parity
        if dedicated_fz_head:
            self.trunk = nn.Sequential(*trunk_layers)
            self.main_head = nn.Linear(hidden_sizes[-1], 5)
            self.fz_head = nn.Sequential(
                nn.Linear(hidden_sizes[-1], fz_head_hidden), nn.ReLU(),
                nn.Linear(fz_head_hidden, 1))
        else:
            trunk_layers.append(nn.Linear(hidden_sizes[-1], 6))
            self.net = nn.Sequential(*trunk_layers)

    def forward(self, x):
        seq    = x[:, self.hourly_idx]
        static = x[:, self.static_idx]
        _, h_n = self.gru(seq)
        gru_summary = h_n[-1]
        combined = torch.cat([gru_summary, static], dim=1)

        if self.dedicated_fz_head:
            h = self.trunk(combined)
            frac_zero  = torch.sigmoid(self.fz_head(h).squeeze(-1))
            main_raw   = self.main_head(h)
            mix_weight = torch.sigmoid(main_raw[:, 0])
            shape1     = self.shape_min + F.softplus(main_raw[:, 1])
            scale1     = self.scale_min + F.softplus(main_raw[:, 2])
            shape2_offset = F.softplus(main_raw[:, 3])
            shape2        = shape1 + shape2_offset + self.min_separation
            scale2        = self.scale_min + F.softplus(main_raw[:, 4])
            return frac_zero, mix_weight, shape1, scale1, shape2, scale2

        raw = self.net(combined)
        frac_zero  = torch.sigmoid(raw[:, 0])
        mix_weight = torch.sigmoid(raw[:, 1])
        shape1     = self.shape_min + F.softplus(raw[:, 2])
        scale1     = self.scale_min + F.softplus(raw[:, 3])
        shape2_offset = F.softplus(raw[:, 4])
        shape2        = shape1 + shape2_offset + self.min_separation
        scale2        = self.scale_min + F.softplus(raw[:, 5])
        return frac_zero, mix_weight, shape1, scale1, shape2, scale2


BERNSTEIN_DEGREE = 10   # must match train_6hourly_mlp.py's BERNSTEIN_DEGREE


class BernsteinGRU(nn.Module):
    """Must match train_6hourly_mlp.py's BernsteinGRU exactly. Requires n_input==73.
    Replaces the 2-component Gamma mixture's conditional-positive params with
    a degree-BERNSTEIN_DEGREE Bernstein-quantile function; frac_zero unchanged."""

    def __init__(self, hidden_sizes=HIDDEN_SIZES, n_input=73, degree=BERNSTEIN_DEGREE,
                 gru_hidden=GRU_HIDDEN):
        super().__init__()
        if n_input != 73:
            raise ValueError(f'BernsteinGRU requires n_input==73; got {n_input}')
        self.degree = degree
        self.shape_min = None
        self.scale_min = None

        self.register_buffer('hourly_idx', torch.tensor(_GRU_HOURLY_IDX, dtype=torch.long))
        self.register_buffer('static_idx', torch.tensor(_GRU_STATIC_IDX, dtype=torch.long))

        self.gru = nn.GRU(input_size=N_HOURLY_FEATS_PER_HOUR, hidden_size=gru_hidden,
                          num_layers=1, batch_first=True)

        n_static = len(_GRU_STATIC_IDX)
        layer_sizes = [gru_hidden + n_static] + hidden_sizes
        trunk_layers = []
        for in_sz, out_sz in zip(layer_sizes, layer_sizes[1:]):
            trunk_layers += [nn.Linear(in_sz, out_sz),
                             nn.BatchNorm1d(out_sz),
                             nn.ReLU(),
                             nn.Dropout(0.15)]  # no-op in eval() mode; kept for state_dict key parity
        trunk_layers.append(nn.Linear(hidden_sizes[-1], 1 + (degree + 1)))
        self.net = nn.Sequential(*trunk_layers)

    def forward(self, x):
        seq    = x[:, self.hourly_idx]
        static = x[:, self.static_idx]
        _, h_n = self.gru(seq)
        gru_summary = h_n[-1]
        combined = torch.cat([gru_summary, static], dim=1)

        raw = self.net(combined)
        frac_zero  = torch.sigmoid(raw[:, 0])
        increments = F.softplus(raw[:, 1:])
        coeffs     = torch.cumsum(increments, dim=1)
        return frac_zero, coeffs


def _bernstein_comb_vec(degree):
    from scipy.special import comb
    return np.array([comb(degree, k) for k in range(degree + 1)], dtype=np.float32)


_BERNSTEIN_COMB_VEC = _bernstein_comb_vec(BERNSTEIN_DEGREE)


def _eval_bernstein_poly(coeffs, tau, degree, comb_vec):
    """coeffs: (...,degree+1), tau: (...) matching coeffs' leading shape ->
    (...) Bernstein-polynomial value at each element's own tau. Computed via
    an accumulator loop over the degree+1 terms using only incremental
    multiplication (never division, so no risk near tau->0/1) and never
    materializing a (...,degree+1) intermediate array -- an earlier version
    built that full array via np.stack + scipy.special.comb calls at every
    one of 40 bisection iterations x 3 thresholds x 932 dates, which
    profiled at ~42 sec/call at full CONUS domain (~29 hours projected total
    eval time); this form profiles at a small fraction of that (see
    bernstein_exceedance_prob's docstring for the actual measured speedup)."""
    K = degree
    omt = 1.0 - tau
    tau_powers = [np.ones_like(tau)]
    omt_powers = [np.ones_like(tau)]
    for _ in range(K):
        tau_powers.append(tau_powers[-1] * tau)
        omt_powers.append(omt_powers[-1] * omt)
    result = coeffs[..., 0] * comb_vec[0] * omt_powers[K]
    for k in range(1, K + 1):
        result += coeffs[..., k] * comb_vec[k] * tau_powers[k] * omt_powers[K - k]
    return result


def bernstein_exceedance_prob(frac_zero, coeffs, threshold, degree=BERNSTEIN_DEGREE, n_iter=30):
    """
    P(Y >= threshold) for the BernsteinGRU family = (1-frac_zero) * (1 - tau_star),
    where tau_star solves Q_pos(tau_star) = threshold via bisection (Q_pos is
    guaranteed non-decreasing by construction -- see BernsteinGRU -- so
    bisection is always well-posed, no derivative/Newton-step fragility).
    n_iter=30 gives tau precision of 2^-30 (~1e-9), far more than needed for
    a probability reported to a few decimal places -- reduced from an
    initial 40 as part of the same performance fix as _eval_bernstein_poly
    (measured: full-CONUS single call dropped from ~42 sec to well under 1
    sec after both fixes -- see git history/memory for the exact benchmark).

    frac_zero : (ny,nx)
    coeffs    : (ny,nx,degree+1)
    threshold : scalar mm
    """
    if threshold <= 0.0:
        return np.clip(1.0 - frac_zero, 0.0, 1.0)

    comb_vec = _BERNSTEIN_COMB_VEC if degree == BERNSTEIN_DEGREE else _bernstein_comb_vec(degree)
    coeffs32 = coeffs.astype(np.float32)
    lo = np.zeros_like(frac_zero, dtype=np.float32)
    hi = np.ones_like(frac_zero, dtype=np.float32)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        q_mid = _eval_bernstein_poly(coeffs32, mid, degree, comb_vec)
        go_right = q_mid < threshold
        lo = np.where(go_right, mid, lo)
        hi = np.where(go_right, hi, mid)
    tau_star = 0.5 * (lo + hi)

    p_nonzero = np.clip(1.0 - frac_zero, 0.0, 1.0)
    return np.clip(p_nonzero * (1.0 - tau_star), 0.0, 1.0)

# =========================================================================
# Checkpoint loader
# =========================================================================

TERRAIN_MASK_NC = os.path.join(SCRIPT_DIR, 'terrain_roughness_mask_graf.nc')


def load_local_std_grid():
    """Static (ny, nx) local terrain-roughness field, same grid as GRAF/MRMS.
    Must match sample_6hourly_prob_mrms.py's load_local_std() exactly.
    log1p-transformed to match the skew-correction applied to sample_local_std
    in train_6hourly_mlp.py's load_data()."""
    with Dataset(TERRAIN_MASK_NC, 'r') as ds:
        return np.log1p(np.asarray(ds.variables['local_std'][:], dtype=np.float32))


TERRAIN_INFO_NC = os.path.join(SCRIPT_DIR, 'GRAF_CONUS_terrain_info.nc')


def load_terrain_grad_grids():
    """Static (ny, nx) terrain elevation-deviation and smoothed-gradient
    fields, same grid as GRAF/MRMS. Must match sample_6hourly_prob_mrms.py's
    load_terrain_fields() exactly (raw, untransformed here -- the sign-log/
    log1p transforms are applied inside apply_mlp_fulldomain(), matching
    train_6hourly_mlp.py's load_data())."""
    with Dataset(TERRAIN_INFO_NC, 'r') as ds:
        return {
            'terrain_height_local_difference':
                np.asarray(ds.variables['terrain_height_local_difference'][:], dtype=np.float32),
            'dterrain_dlon_smoothed':
                np.asarray(ds.variables['dterrain_dlon_smoothed'][:], dtype=np.float32),
            'dterrain_dlat_smoothed':
                np.asarray(ds.variables['dterrain_dlat_smoothed'][:], dtype=np.float32),
        }


def load_mlp(clead, device, variant=None):
    suffix = f'_{variant}' if variant else ''
    ckpt_path = os.path.join(SCRIPT_DIR, 'mlp_trainings',
                             f'6h_mlp_lead{clead}h{suffix}.pth')
    if not os.path.exists(ckpt_path):
        print(f'ERROR: MLP checkpoint not found: {ckpt_path}')
        print(f'  Run:  python train_6hourly_mlp.py {clead}'
              f'{" " + variant if variant else ""}')
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden_sizes = ckpt.get('hidden_sizes', HIDDEN_SIZES)
    shape_min    = ckpt.get('shape_min',    SHAPE_MIN)
    scale_min    = ckpt.get('scale_min',    SCALE_MIN)
    n_input      = ckpt.get('n_input',      38)
    architecture = ckpt.get('architecture', 'concat')
    dedicated_fz_head = ckpt.get('dedicated_fz_head', False)
    bernstein_degree  = ckpt.get('bernstein_degree', None)
    precip_transform  = ckpt.get('precip_transform', 'log1p')

    if architecture == 'bernstein_gru':
        model = BernsteinGRU(hidden_sizes=hidden_sizes, n_input=n_input,
                             degree=bernstein_degree or BERNSTEIN_DEGREE)
    else:
        ModelClass = (GammaMixtureGRU if architecture == 'gru'
                     else GammaMixtureMLPFiLM if architecture == 'film'
                     else GammaMixtureMLP)
        extra_kwargs = {} if architecture == 'film' else {'dedicated_fz_head': dedicated_fz_head}
        model = ModelClass(hidden_sizes=hidden_sizes,
                           shape_min=shape_min, scale_min=scale_min,
                           n_input=n_input, **extra_kwargs)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    feat_mean = ckpt['feature_mean']
    feat_std  = ckpt['feature_std']
    print(f'Loaded MLP from {ckpt_path}  (epoch {ckpt["epoch"]+1})')
    return model, feat_mean, feat_std, precip_transform

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


def get_texture_dir():
    """Separate from get_paths() (kept 4-tuple for backward compat with its
    other callers): same base-path detection, new 'graf_texture' subdir --
    see save_graf_texture_features.py."""
    base = AWS_BASE_PATH if ENVIRONMENT == 'aws' else os.path.expanduser('~/python/resnet_data')
    return os.path.join(base, 'graf_texture')

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


def read_texture_params_6h(texture_dir, cyyyymmddhh, clead):
    """Read one save_graf_texture_features.py output file (already covers
    the whole 6-hour window -- unlike read_prob_params_6h, no per-hour
    stacking needed). Returns (texture_spatial, texture_temporal,
    raw_precip_6h), each None (all three) if the file is missing/unreadable."""
    fname = os.path.join(texture_dir, f'{cyyyymmddhh}_{clead}_graf_texture_features.nc')
    if not os.path.exists(fname):
        return None, None, None
    try:
        with Dataset(fname, 'r') as ds:
            texture_spatial = {
                'wet_area_fraction':  ds.variables['wet_area_fraction'][:].data.astype(np.float32),
                'peak_to_mean_ratio': ds.variables['peak_to_mean_ratio'][:].data.astype(np.float32),
                'coeff_variation':    ds.variables['coeff_variation'][:].data.astype(np.float32),
            }
            texture_temporal = {
                'wetdry_jaccard':      ds.variables['wetdry_jaccard'][:].data.astype(np.float32),
                'zscore_pattern_corr': ds.variables['zscore_pattern_corr'][:].data.astype(np.float32),
            }
            raw_precip_6h = ds.variables['precip_6h_total'][:].data.astype(np.float32)
    except Exception as exc:
        print(f'  WARNING: cannot read {fname}: {exc}')
        return None, None, None
    return texture_spatial, texture_temporal, raw_precip_6h

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
                          cos_doy, sin_doy, local_std=None, cos_hod=None, sin_hod=None,
                          texture_spatial=None, texture_temporal=None,
                          raw_precip_6h=None, terrain_grad=None, precip_transform='log1p'):
    """
    texture_spatial : dict with 'wet_area_fraction'/'peak_to_mean_ratio'/
        'coeff_variation', each (6,ny,nx), from save_graf_texture_features.py
    texture_temporal : dict with 'wetdry_jaccard'/'zscore_pattern_corr',
        each (5,ny,nx)
    raw_precip_6h : (ny,nx) raw GRAF 6-h precip total
    terrain_grad : dict with 'terrain_height_local_difference'/
        'dterrain_dlon_smoothed'/'dterrain_dlat_smoothed', each (ny,nx)
    precip_transform : 'log1p' (default) or 'sqrt' -- must match whatever
        the loaded checkpoint was trained with (see train_6hourly_mlp.py's
        load_data() docstring); load_mlp() reads this from the checkpoint
        and the caller passes it through, so a mismatch shouldn't happen
        in practice, but this silently produces garbage predictions if it
        ever does -- same caveat as the column-order note below.

    Normalization/column order must exactly match train_6hourly_mlp.py's
    load_data(): hourly, seasonal, local_std, hour-of-day, [texture spatial,
    texture temporal, raw precip, terrain gradient] -- see TEXTURE_SPATIAL_VARS
    etc. in that file. A mismatch here silently produces garbage predictions.
    """
    npix = ny * nx
    blocks = [params_6h[k].reshape(6, npix).T for k in PARAM_VARS]
    hourly_feats   = np.concatenate(blocks, axis=1).astype(np.float32)
    seasonal_feats = np.tile(np.array([cos_doy, sin_doy], dtype=np.float32), (npix, 1))
    feat_blocks = [hourly_feats, seasonal_feats]
    if local_std is not None:
        feat_blocks.append(local_std.reshape(npix, 1).astype(np.float32))
    if cos_hod is not None and sin_hod is not None:
        feat_blocks.append(np.tile(np.array([cos_hod, sin_hod], dtype=np.float32), (npix, 1)))
    if texture_spatial is not None:
        waf  = texture_spatial['wet_area_fraction'].reshape(6, npix).T.astype(np.float32)
        p2m  = np.log1p(texture_spatial['peak_to_mean_ratio']).reshape(6, npix).T.astype(np.float32)
        cv   = np.log1p(texture_spatial['coeff_variation']).reshape(6, npix).T.astype(np.float32)
        feat_blocks.append(np.concatenate([waf, p2m, cv], axis=1))   # (npix, 18)
    if texture_temporal is not None:
        jac   = texture_temporal['wetdry_jaccard'].reshape(5, npix).T.astype(np.float32)
        pcorr = texture_temporal['zscore_pattern_corr'].reshape(5, npix).T.astype(np.float32)
        feat_blocks.append(np.concatenate([jac, pcorr], axis=1))     # (npix, 10)
    if raw_precip_6h is not None:
        transformed = (np.sqrt(raw_precip_6h) if precip_transform == 'sqrt'
                      else np.log1p(raw_precip_6h))
        feat_blocks.append(transformed.reshape(npix, 1).astype(np.float32))
    if terrain_grad is not None:
        diff = terrain_grad['terrain_height_local_difference']
        diff = np.sign(diff) * np.log1p(np.abs(diff))
        dlon = terrain_grad['dterrain_dlon_smoothed']
        dlat = terrain_grad['dterrain_dlat_smoothed']
        feat_blocks.append(np.stack(
            [diff.reshape(npix), dlon.reshape(npix), dlat.reshape(npix)], axis=1).astype(np.float32))  # (npix, 3)
    feats = np.concatenate(feat_blocks, axis=1)

    std_safe   = np.where(feat_std < 1e-8, 1.0, feat_std)
    feats_norm = (feats - feat_mean) / std_safe

    if isinstance(model, BernsteinGRU):
        fz_out, coeffs_out = [], []
        with torch.no_grad():
            for start in range(0, npix, MLP_BATCH):
                end = min(start + MLP_BATCH, npix)
                xb  = torch.tensor(feats_norm[start:end], dtype=torch.float32,
                                   device=device)
                fz, coeffs = model(xb)
                fz_out.append(fz.cpu().numpy())
                coeffs_out.append(coeffs.cpu().numpy())
        frac_zero_full = np.concatenate(fz_out).reshape(ny, nx)
        coeffs_full    = np.concatenate(coeffs_out).reshape(ny, nx, -1)
        return frac_zero_full, coeffs_full

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
# Out-of-sample test date list: day 12 through month-end, all 12 months
# of 2025, all 4 GRAF cycles (00/06/12/18Z).  Days 1-9 of each month are
# used for MLP training (see sample_6hourly_prob_mrms.py) and days 10-11
# are a gap left out entirely, so persistent synoptic systems can't leak
# across the train/test boundary.
# (2026-08-05: widened train window 1-7 -> 1-9 to add more independent
# synoptic snapshots, part of the sampling-redundancy fix; gap shifted
# 8-9 -> 10-11, test start shifted 10 -> 12.)
# (2026-08-06: extended test cycles 00/12Z -> all 4 cycles, to match the
# 4-cycles/day training data and get 06/18Z verification coverage.)
# =========================================================================

MONTHS_2025 = [
    (1, 31), (2, 28), (3, 31), (4, 30), (5, 31), (6, 30),
    (7, 31), (8, 31), (9, 30), (10, 31), (11, 30), (12, 31),
]
TEST_DAY_START = 12   # days 1-9 train, 10-11 gap, 12-end test
TEST_CYCLES = ['00', '06', '12', '18']


def build_test_datelist():
    date_list = []
    for mm, ndays in MONTHS_2025:
        for dd in range(TEST_DAY_START, ndays + 1):
            for cyc in TEST_CYCLES:
                date_list.append(f'2025{mm:02d}{dd:02d}{cyc}')
    return date_list


# =========================================================================
# Main
# =========================================================================

class VariantState:
    """Per-variant bookkeeping for a shared-I/O multi-variant evaluation run.

    Evaluating several checkpoints that differ only in the model/precip
    transform (e.g. the sqrt- vs. log1p-precip transform comparison) over the
    same ~900-date test window used to mean re-reading the same params/MRMS/
    control/copula/texture netCDFs once per variant -- the dominant per-date
    cost (see timing note in the sqrt-precip-all-leads driver). This class
    holds one variant's model/accumulators/percache so main()'s date loop can
    read each date's raw data ONCE and then loop over variants for the (much
    cheaper) forward pass + contingency-table accumulation. Semantics for a
    single variant are unchanged from the pre-refactor single-variant code
    path -- same filenames, same percache format, same staleness checks.
    """

    def __init__(self, clead, variant, relia_dir, date_start, date_end, nthresholds, ncats):
        self.variant = variant
        self.suffix = f'_{variant}' if variant else ''
        self.pick_fname = os.path.join(
            relia_dir,
            f'relia_6h_MLP_3panel_q0.6_{date_start}_to_{date_end}_lead{clead}h{self.suffix}.cPick')
        self.percache_fname = os.path.join(
            relia_dir, f'relia_6h_MLP_3panel_percache_lead{clead}h{self.suffix}.cPick')

        # Computed here (not just when recomputing) so the aggregate
        # pick_fname cache can be validated against it too -- a retrain under
        # the same variant name must not silently serve stale results just
        # because the .cPick file already exists.
        ckpt_path = os.path.join(SCRIPT_DIR, 'mlp_trainings', f'6h_mlp_lead{clead}h{self.suffix}.pth')
        self.ckpt_mtime = os.path.getmtime(ckpt_path) if os.path.exists(ckpt_path) else None

        d = {}
        if os.path.exists(self.pick_fname):
            with open(self.pick_fname, 'rb') as fh:
                d = cPickle.load(fh)

        self.done = 'ttest_p' in d and d.get('ckpt_mtime') == self.ckpt_mtime
        if self.done:
            print(f'[{variant}] Loading saved statistics from:\n  {self.pick_fname}')
            self.probability         = d['probability']
            self.relia_arr           = d['relia']
            self.frequse_arr         = d['frequse']
            self.BSS_arr             = d['BSS']
            self.relia_control_arr   = d['relia_control']
            self.frequse_control_arr = d['frequse_control']
            self.BSS_control_arr     = d['BSS_control']
            self.pthresholds         = d['pthresholds']
            print(f'  Loaded.  ngood={d["ngood"]}')
            return

        if os.path.exists(self.pick_fname):
            if 'ttest_p' not in d:
                print(f'[{variant}] Cached statistics at {self.pick_fname} lack the paired-'
                      f'significance-test fields; recomputing.')
            else:
                print(f'[{variant}] Cached statistics at {self.pick_fname} predate the current '
                      f'checkpoint (stale after a retrain); recomputing.')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'[{variant}] Torch device: {device}')
        model, feat_mean, feat_std, precip_transform = load_mlp(clead, device, variant=variant)
        self.device = device
        self.model, self.feat_mean, self.feat_std = model, feat_mean, feat_std
        self.precip_transform = precip_transform
        self.is_bernstein = isinstance(model, BernsteinGRU)

        # Only checkpoints trained with the local_std terrain feature
        # (2026-08-04 onward) have n_input>=39; older checkpoints stay at 38
        # and must not get a 39th feature appended. Checkpoints trained with
        # the cos/sin hour-of-day feature (2026-08 CRPS experiment onward)
        # have n_input=41 and need the 40th/41st features appended too.
        # Checkpoints trained with --texture have n_input=73.
        self.local_std_grid    = load_local_std_grid() if len(feat_mean) >= 39 else None
        self.use_hod_feat      = len(feat_mean) >= 41
        self.use_texture_feats = len(feat_mean) >= 73
        self.terrain_grad_grids = load_terrain_grad_grids() if self.use_texture_feats else None

        # Per-date result cache (expensive part -- model forward pass +
        # MRMS/control/copula reads) so widening the test date list only
        # computes the new dates. Invalidated wholesale if the checkpoint has
        # been retrained since the cache was written.
        self.percache = {'ckpt_mtime': self.ckpt_mtime, 'dates': {}}
        if os.path.exists(self.percache_fname):
            with open(self.percache_fname, 'rb') as fh:
                loaded = cPickle.load(fh)
            if loaded.get('ckpt_mtime') == self.ckpt_mtime:
                self.percache = loaded
                print(f'[{variant}] Loaded per-date cache: {len(self.percache["dates"])} dates '
                      f'already computed ({self.percache_fname})')
            else:
                print(f'[{variant}] Per-date cache checkpoint mtime mismatch (stale after a '
                      f'retrain) -- discarding {self.percache_fname}')

        self.contab          = np.zeros((nthresholds, ncats, 2), dtype=np.int64)
        self.BS_sum          = np.zeros(nthresholds, dtype=float)
        self.nsamps_sum      = np.zeros(nthresholds, dtype=float)
        self.nobs_exceed_sum = np.zeros(nthresholds, dtype=float)
        self.nobs_total_sum  = np.zeros(nthresholds, dtype=float)

        self.contab_control          = np.zeros((nthresholds, ncats, 2), dtype=np.int64)
        self.BS_sum_control          = np.zeros(nthresholds, dtype=float)
        self.nsamps_sum_control      = np.zeros(nthresholds, dtype=float)
        self.nobs_exceed_sum_control = np.zeros(nthresholds, dtype=float)
        self.nobs_total_sum_control  = np.zeros(nthresholds, dtype=float)

        self.contab_copula          = np.zeros((nthresholds, ncats, 2), dtype=np.int64)
        self.BS_sum_copula          = np.zeros(nthresholds, dtype=float)
        self.nsamps_sum_copula      = np.zeros(nthresholds, dtype=float)
        self.nobs_exceed_sum_copula = np.zeros(nthresholds, dtype=float)
        self.nobs_total_sum_copula  = np.zeros(nthresholds, dtype=float)
        self.ngood     = 0
        self.ncached   = 0
        self.ncomputed = 0

        # Per-case-day mean Brier Score, one value per verification date per
        # threshold, for the MLP and the control. Following Hamill (1999,
        # Wea. Forecasting), all grid points within a case day are pooled
        # into a single per-day score before testing, rather than treating
        # every grid point as an independent sample, since nearby grid points
        # are spatially correlated within a synoptic event.
        self.daily_BS_mlp     = [[] for _ in range(nthresholds)]
        self.daily_BS_control = [[] for _ in range(nthresholds)]

    def accumulate_cached(self, date_result, nthresholds):
        self.ngood   += 1
        self.ncached += 1
        for ithresh in range(nthresholds):
            ctab, bs, ns, nex, ntot = date_result['mlp'][ithresh]
            self.contab[ithresh]          += ctab
            self.BS_sum[ithresh]          += bs
            self.nsamps_sum[ithresh]      += ns
            self.nobs_exceed_sum[ithresh] += nex
            self.nobs_total_sum[ithresh]  += ntot

            ctab_c, bs_c, ns_c, nex_c, ntot_c = date_result['control'][ithresh]
            self.contab_control[ithresh]          += ctab_c
            self.BS_sum_control[ithresh]          += bs_c
            self.nsamps_sum_control[ithresh]      += ns_c
            self.nobs_exceed_sum_control[ithresh] += nex_c
            self.nobs_total_sum_control[ithresh]  += ntot_c

            if ns > 0 and ns_c > 0:
                self.daily_BS_mlp[ithresh].append(bs / ns)
                self.daily_BS_control[ithresh].append(bs_c / ns_c)

            ctab_p, bs_p, ns_p, nex_p, ntot_p = date_result['copula'][ithresh]
            self.contab_copula[ithresh]          += ctab_p
            self.BS_sum_copula[ithresh]          += bs_p
            self.nsamps_sum_copula[ithresh]      += ns_p
            self.nobs_exceed_sum_copula[ithresh] += nex_p
            self.nobs_total_sum_copula[ithresh]  += ntot_p

    def compute_and_accumulate(self, cdate, params_6h, ny, nx, precip_6h, mean_qual,
                               control_probs, copula_probs, copula_ok,
                               texture_spatial, texture_temporal, raw_precip_6h,
                               cos_doy, sin_doy, pthresholds, ncats):
        cos_hod, sin_hod = (hour_of_day_features(cdate) if self.use_hod_feat
                            else (None, None))
        mlp_out = apply_mlp_fulldomain(
            self.model, self.feat_mean, self.feat_std, params_6h, ny, nx, self.device,
            cos_doy, sin_doy, local_std=self.local_std_grid,
            cos_hod=cos_hod, sin_hod=sin_hod,
            texture_spatial=texture_spatial, texture_temporal=texture_temporal,
            raw_precip_6h=raw_precip_6h, terrain_grad=self.terrain_grad_grids,
            precip_transform=self.precip_transform)
        if self.is_bernstein:
            frac_zero_full, coeffs_full = mlp_out
        else:
            fz, mw, s1, sc1, s2, sc2 = mlp_out

        nthresholds = len(pthresholds)
        date_result = {'mlp': [None] * nthresholds, 'control': [None] * nthresholds,
                       'copula': [None] * nthresholds}

        for ithresh, thresh in enumerate(pthresholds):
            if self.is_bernstein:
                prob = bernstein_exceedance_prob(frac_zero_full, coeffs_full, thresh)
            else:
                prob = exceedance_prob(fz, mw, s1, sc1, s2, sc2, thresh)
            ctab, bs, ns, nex, ntot = compute_contab_BS(
                ny, nx, prob, precip_6h, mean_qual, ncats, thresh)
            self.contab[ithresh]          += ctab
            self.BS_sum[ithresh]          += bs
            self.nsamps_sum[ithresh]      += ns
            self.nobs_exceed_sum[ithresh] += nex
            self.nobs_total_sum[ithresh]  += ntot
            date_result['mlp'][ithresh] = (ctab, bs, ns, nex, ntot)

            ctab_c, bs_c, ns_c, nex_c, ntot_c = compute_contab_BS(
                ny, nx, control_probs[thresh], precip_6h, mean_qual, ncats, thresh)
            self.contab_control[ithresh]          += ctab_c
            self.BS_sum_control[ithresh]          += bs_c
            self.nsamps_sum_control[ithresh]      += ns_c
            self.nobs_exceed_sum_control[ithresh] += nex_c
            self.nobs_total_sum_control[ithresh]  += ntot_c
            date_result['control'][ithresh] = (ctab_c, bs_c, ns_c, nex_c, ntot_c)

            if ns > 0 and ns_c > 0:
                self.daily_BS_mlp[ithresh].append(bs / ns)
                self.daily_BS_control[ithresh].append(bs_c / ns_c)

            if copula_ok:
                ctab_p, bs_p, ns_p, nex_p, ntot_p = compute_contab_BS(
                    ny, nx, copula_probs[thresh], precip_6h, mean_qual, ncats, thresh)
                self.contab_copula[ithresh]          += ctab_p
                self.BS_sum_copula[ithresh]          += bs_p
                self.nsamps_sum_copula[ithresh]      += ns_p
                self.nobs_exceed_sum_copula[ithresh] += nex_p
                self.nobs_total_sum_copula[ithresh]  += ntot_p
                date_result['copula'][ithresh] = (ctab_p, bs_p, ns_p, nex_p, ntot_p)
            else:
                date_result['copula'][ithresh] = (
                    np.zeros((ncats, 2), dtype=np.int64), 0.0, 0.0, 0.0, 0.0)

        self.percache['dates'][cdate] = date_result
        self.ngood     += 1
        self.ncomputed += 1

    def save_percache(self):
        with open(self.percache_fname, 'wb') as fh:
            cPickle.dump(self.percache, fh)

    def finalize(self, ndates, pthresholds, ncats):
        if self.ncomputed > 0:
            self.save_percache()
        print(f'\n[{self.variant}] Per-date cache: {self.ncached} reused, {self.ncomputed} '
              f'newly computed, {len(self.percache["dates"])} total cached -> '
              f'{self.percache_fname}')

        if self.ngood == 0:
            print(f'\n[{self.variant}] ERROR: No dates with complete data found -- skipping.')
            return

        print(f'\n[{self.variant}] {self.ngood}/{ndates} init times had complete data.')

        nthresholds = len(pthresholds)
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
            if self.nsamps_sum[ithresh] == 0:
                print(f'  [{self.variant}] thresh={thresh} mm: no valid samples')
                continue

            BS_mean    = self.BS_sum[ithresh] / self.nsamps_sum[ithresh]
            climo_freq = (self.nobs_exceed_sum[ithresh] / self.nobs_total_sum[ithresh]
                          if self.nobs_total_sum[ithresh] > 0 else np.nan)
            BS_climo   = climo_freq * (1.0 - climo_freq) if not np.isnan(climo_freq) else np.nan
            BSS        = (1.0 - BS_mean / BS_climo
                          if (not np.isnan(BS_climo) and BS_climo > 0) else np.nan)

            frequse, relia = compute_relia(self.contab[ithresh], ncats)

            relia_arr[ithresh]    = relia
            frequse_arr[ithresh]  = frequse
            BS_arr[ithresh]       = BS_mean
            BS_climo_arr[ithresh] = BS_climo
            BSS_arr[ithresh]      = BSS
            climo_arr[ithresh]    = climo_freq

            cbss = f'{BSS:.3f}' if not np.isnan(BSS) else 'N/A'
            print(f'  [{self.variant}] thresh={thresh:5.2f} mm | climo={climo_freq:.4f}  '
                  f'BS={BS_mean:.5f}  BS_climo={BS_climo:.5f}  BSS={cbss}')

            # --- independence-assumption ensemble control (same climo/obs) ---
            BS_mean_c = (self.BS_sum_control[ithresh] / self.nsamps_sum_control[ithresh]
                        if self.nsamps_sum_control[ithresh] > 0 else np.nan)
            BSS_c     = (1.0 - BS_mean_c / BS_climo
                        if (not np.isnan(BS_climo) and BS_climo > 0
                            and not np.isnan(BS_mean_c)) else np.nan)

            frequse_c, relia_c = compute_relia(self.contab_control[ithresh], ncats)

            relia_control_arr[ithresh]   = relia_c
            frequse_control_arr[ithresh] = frequse_c
            BS_control_arr[ithresh]      = BS_mean_c
            BSS_control_arr[ithresh]     = BSS_c

            cbss_c = f'{BSS_c:.3f}' if not np.isnan(BSS_c) else 'N/A'
            print(f'    [control] BS={BS_mean_c:.5f}  BSS={cbss_c}')

            # --- MLP-vs-control significance test, following Hamill (1999,
            # Wea. Forecasting, 14, 155-167). ---
            d_mlp = np.asarray(self.daily_BS_mlp[ithresh])
            d_ctl = np.asarray(self.daily_BS_control[ithresh])
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
            BS_mean_p = (self.BS_sum_copula[ithresh] / self.nsamps_sum_copula[ithresh]
                        if self.nsamps_sum_copula[ithresh] > 0 else np.nan)
            BSS_p     = (1.0 - BS_mean_p / BS_climo
                        if (not np.isnan(BS_climo) and BS_climo > 0
                            and not np.isnan(BS_mean_p)) else np.nan)

            frequse_p, relia_p = compute_relia(self.contab_copula[ithresh], ncats)

            relia_copula_arr[ithresh]   = relia_p
            frequse_copula_arr[ithresh] = frequse_p
            BS_copula_arr[ithresh]      = BS_mean_p
            BSS_copula_arr[ithresh]     = BSS_p

            cbss_p = f'{BSS_p:.3f}' if not np.isnan(BSS_p) else 'N/A'
            print(f'    [copula]  BS={BS_mean_p:.5f}  BSS={cbss_p}')

        out_dict = {
            'ckpt_mtime':       self.ckpt_mtime,
            'pthresholds':      pthresholds,
            'probability':      probability,
            'ngood':            self.ngood,
            'relia':            relia_arr,
            'frequse':          frequse_arr,
            'BS':               BS_arr,
            'BS_climo':         BS_climo_arr,
            'BSS':              BSS_arr,
            'climo_freq':       climo_arr,
            'contab':           self.contab,
            'nsamps':           self.nsamps_sum,
            'nobs_exceed':      self.nobs_exceed_sum,
            'nobs_total':       self.nobs_total_sum,
            'relia_control':    relia_control_arr,
            'frequse_control':  frequse_control_arr,
            'BS_control':       BS_control_arr,
            'BSS_control':      BSS_control_arr,
            'contab_control':   self.contab_control,
            'nsamps_control':   self.nsamps_sum_control,
            'relia_copula':     relia_copula_arr,
            'frequse_copula':   frequse_copula_arr,
            'BS_copula':        BS_copula_arr,
            'BSS_copula':       BSS_copula_arr,
            'contab_copula':    self.contab_copula,
            'nsamps_copula':    self.nsamps_sum_copula,
            'n_days_sig':       n_days_arr,
            'ttest_p':          ttest_p_arr,
            'wilcoxon_p':       wilcoxon_p_arr,
        }
        with open(self.pick_fname, 'wb') as fh:
            cPickle.dump(out_dict, fh)
        print(f'[{self.variant}] Saved statistics to {self.pick_fname}')

        self.probability         = probability
        self.relia_arr           = relia_arr
        self.frequse_arr         = frequse_arr
        self.BSS_arr             = BSS_arr
        self.relia_control_arr   = relia_control_arr
        self.frequse_control_arr = frequse_control_arr
        self.BSS_control_arr     = BSS_control_arr
        self.pthresholds         = pthresholds


def main():
    if len(sys.argv) not in (2, 3):
        print('Usage: python reliability_6hourly_mlp_3panel.py <clead> [variant1[,variant2,...]]')
        sys.exit(1)

    clead = int(sys.argv[1])
    variant_arg = sys.argv[2] if len(sys.argv) == 3 else None
    # A comma-separated variant list runs all of them through one pass over
    # the test dates, reading each date's params/MRMS/control/copula/texture
    # netCDFs only once and sharing that across variants (see VariantState's
    # docstring) -- multiple variants of the same architecture (e.g. sqrt-
    # vs. log1p-precip) is the case this was built for.
    variants = variant_arg.split(',') if variant_arg else [None]
    if clead < 6:
        print('ERROR: clead must be >= 6')
        sys.exit(1)

    print(f'reliability_6hourly_mlp_3panel.py  clead={clead}h  '
          f'(6-h window: {clead-5}–{clead} h)  variants={variants}')

    probs_dir, mrms_dir, relia_dir, control_dir = get_paths()
    os.makedirs(relia_dir, exist_ok=True)

    # Date list — out-of-sample test months (see build_test_datelist)
    cyyyymmddhh_list = build_test_datelist()
    date_start = cyyyymmddhh_list[0]
    date_end   = cyyyymmddhh_list[-1]
    ndates     = len(cyyyymmddhh_list)

    pthresholds = [0.25, 2.5, 10.0]
    nthresholds = len(pthresholds)
    ncats       = 11

    vstates = [VariantState(clead, v, relia_dir, date_start, date_end, nthresholds, ncats)
              for v in variants]
    needing = [vs for vs in vstates if not vs.done]

    if needing:
        need_texture = any(vs.use_texture_feats for vs in needing)
        texture_dir = get_texture_dir() if need_texture else None

        print(f'Total init times to process: {ndates}  '
              f'(variants needing (re)compute: {[vs.variant for vs in needing]})')

        SAVE_EVERY = 50
        for idate, cdate in enumerate(cyyyymmddhh_list):
            still_needed = [vs for vs in needing if cdate not in vs.percache['dates']]
            for vs in needing:
                if vs not in still_needed:
                    vs.accumulate_cached(vs.percache['dates'][cdate], nthresholds)
            if not still_needed:
                if (idate + 1) % 100 == 0 or idate + 1 == ndates:
                    print(f'{idate+1:4d}/{ndates}  init={cdate}  (from per-date cache, all variants)')
                continue

            # Raw per-date reads happen at most once here, regardless of how
            # many variants still need this date -- this is the whole point
            # of the shared-I/O refactor.
            params_6h, lat, lon = read_prob_params_6h(probs_dir, cdate, clead)
            prob_ok = params_6h is not None

            precip_6h, mean_qual, mrms_istat = read_mrms_6h(mrms_dir, cdate, clead)
            mrms_ok = mrms_istat == 0

            control_probs = read_control_probs_6h(control_dir, cdate, clead, pthresholds)
            control_ok = control_probs is not None

            copula_probs = read_copula_control_probs_6h(control_dir, cdate, clead, pthresholds)
            copula_ok = copula_probs is not None

            if need_texture:
                texture_spatial, texture_temporal, raw_precip_6h = \
                    read_texture_params_6h(texture_dir, cdate, clead)
                texture_ok = texture_spatial is not None
            else:
                texture_spatial = texture_temporal = raw_precip_6h = None
                texture_ok = True

            ps  = 'ok' if prob_ok else 'missing'
            ms  = 'ok' if mrms_ok else 'missing'
            cs  = 'ok' if control_ok else 'missing'
            cps = 'ok' if copula_ok else 'missing'
            txs = 'ok' if texture_ok else 'missing'
            print(f'{idate+1:4d}/{ndates}  init={cdate}  params={ps}  mrms={ms}  '
                  f'control={cs}  copula={cps}  texture={txs}')

            # copula_ok is intentionally NOT required here (2026-08-09): the
            # conditional-copula control has only ever been fit/generated for
            # a subset of leads (see build_test_datelist()'s callers). Copula
            # columns simply stay at all-zero/NaN when unavailable.
            if not prob_ok or not mrms_ok or not control_ok:
                continue

            ny, nx = precip_6h.shape
            cos_doy, sin_doy = julian_features(cdate)

            for vs in still_needed:
                # A variant that needs texture features can't use this date
                # if the texture file is missing, even though non-texture
                # variants in the same run still can.
                if vs.use_texture_feats and not texture_ok:
                    continue
                vs.compute_and_accumulate(
                    cdate, params_6h, ny, nx, precip_6h, mean_qual,
                    control_probs, copula_probs, copula_ok,
                    texture_spatial, texture_temporal, raw_precip_6h,
                    cos_doy, sin_doy, pthresholds, ncats)
                if vs.ncomputed % SAVE_EVERY == 0:
                    vs.save_percache()

    for vs in vstates:
        if not vs.done:
            vs.finalize(ndates, pthresholds, ncats)
            if vs.ngood == 0:
                continue
        plot_fname = (f'Relia_6h_MLP_MRMS_3panel_{date_start}_to_{date_end}'
                      f'_{clead}h{vs.suffix}.png')
        plot_3panel(vs.probability, vs.relia_arr, vs.frequse_arr, vs.BSS_arr,
                    vs.relia_control_arr, vs.frequse_control_arr, vs.BSS_control_arr,
                    vs.pthresholds, clead, date_start, date_end, plot_fname)


if __name__ == '__main__':
    main()
