"""
train_6hourly_mlp.py — MLP for 6-Hourly Gamma Mixture Parameters

Usage:
    python train_6hourly_mlp.py <clead>

    <clead>  Lead time in hours (integer).  Only lead times for which
             prob_MRMS_samples_*_lead{clead}h.nc files exist will train.

Description
-----------
Reads the importance-sampled dataset produced by sample_6hourly_prob_mrms.py
and trains a small MLP to predict the six parameters of a zero-inflated
two-component Gamma mixture distribution for 6-hourly accumulated precipitation.

Input features (41 total):
    Six consecutive hourly gamma-mixture parameters (clead-5 … clead),
    grouped by variable:
        fraction_zero  × 6 hours   (cols  0-5)
        mixture_weight × 6 hours   (cols  6-11)
        gamma_shape1   × 6 hours   (cols 12-17)
        gamma_scale1   × 6 hours   (cols 18-23)
        gamma_shape2   × 6 hours   (cols 24-29)
        gamma_scale2   × 6 hours   (cols 30-35)
    plus cos/sin(day-of-year) (cols 36-37), log1p(local terrain std)
    (col 38), and cos/sin(hour-of-day) (cols 39-40, derived from the
    init cycle -- 00/06/12/18Z).

Output parameters (6 per sample):
    fraction_zero, mixture_weight, shape1, scale1, shape2, scale2

Loss function:
    NLL for zero-inflated two-component Gamma mixture (default); or
    discretized CRPS via reparameterized Monte Carlo samples ("crps"
    variant -- see crps_loss() for why CRPS can't be computed from the
    closed-form CDF the way NLL is computed from the closed-form PDF);
    or a hurdle-model loss ("hurdle" variant, see hurdle_loss()): exact
    NLL for the zero/positive split, CRPS for the conditional-on-positive
    mixture -- targets CRPS's weak gradient near frac_zero's extremes
    without giving up CRPS's calibration advantage for the amount
    distribution.

Checkpoint saved to:
    mlp_trainings/6h_mlp_lead{clead}h.pth                  (default / NLL)
    mlp_trainings/6h_mlp_lead{clead}h_film.pth             (FiLM architecture)
    mlp_trainings/6h_mlp_lead{clead}h_crps.pth             (CRPS loss)
    mlp_trainings/6h_mlp_lead{clead}h_hurdle.pth           (hurdle loss)
    mlp_trainings/6h_mlp_lead{clead}h_hurdle_texture.pth   (+ --texture flag,
        composes with any loss variant -- see TEXTURE_SPATIAL_VARS etc. and
        N_INPUT_TEXTURE)

Tom Hamill, Apr 2026
"""

import os
import sys
import glob
import math
import numpy as np
from netCDF4 import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from gamma_mixture_em import fit_gamma_mixture

# =========================================================================
# Device
# =========================================================================

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print(f'Using device: {DEVICE}')

# =========================================================================
# Hyperparameters
# =========================================================================

BATCH_SIZE    = 1024
LEARNING_RATE = 2.5e-4  # was 1e-3; lowered 2026-08-04 after nsamps 4x increase
                        # caused the lead24h retrain to overshoot within
                        # epoch 1 (best val NLL at epoch 1, monotonically
                        # worse after) -- ~4x more optimizer steps/epoch at
                        # the old LR, so scaling LR down proportionally
MAX_EPOCHS    = 75
LR_PATIENCE   = 5          # ReduceLROnPlateau patience
ES_PATIENCE   = 5          # early-stopping patience

DROPOUT_P     = 0.15    # 2026-08-05: added alongside the sampling-redundancy
                        # fix (reduced nsamps + spatial block-cap thinning in
                        # sample_6hourly_prob_mrms.py) as a belt-and-suspenders
                        # regularizer against the epoch-1-overfit pattern
                        # diagnosed at every lead.
WEIGHT_DECAY  = 1e-4    # L2 penalty on Adam, same motivation

# Train/validation split: sequential calendar-day blocks within each
# calendar month, so validation days are never adjacent to training days
# on both sides. Both 00Z and 12Z inits of a given day fall in the same
# split. Pattern resets at each month boundary (day-of-month 1).
VAL_BLOCK_STRIDE = 5   # every VAL_BLOCK_STRIDE-th day-of-month is held out
RANDOM_SEED       = 42

SHAPE_MIN = 0.1
SCALE_MIN = 0.01
MIN_SEPARATION = 0.5   # hard floor on (shape2 - shape1); mirrors
                       # pytorch_train_resunet_gamma_mixture.py's GammaMixtureNLLLoss

HIDDEN_SIZES = [72, 144, 72, 36, 12]
# For --texture models (73 inputs): with the original HIDDEN_SIZES, the
# first layer (73->72) actually COMPRESSES the input rather than expanding
# it as it did at 41 inputs (41->72, a 1.76x expansion) -- a concrete
# capacity bottleneck identified 2026-08, not just a hunch. Widen only the
# first layer (73*1.75~=128) to restore a comparable expansion ratio;
# leave the rest of the funnel unchanged so this is an isolated, attributable
# change on top of the window-size widening tested in the same round.
HIDDEN_SIZES_TEXTURE = [128, 144, 72, 36, 12]
FZ_HEAD_HIDDEN = 16   # dedicated fz branch width; small on purpose -- mirrors
                      # HRRRcal's dedicated_fz_head (Conv2d(64,32)->ReLU->
                      # Conv2d(32,1)), scaled down for this flat-feature context

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _locate_data_dir():
    for base in ['/data/resnet_data', '/data2/resnet_data']:
        candidate = os.path.join(base, 'prob_samples')
        if os.path.isdir(candidate):
            return candidate
    raise RuntimeError(
        "Cannot locate prob_samples directory. "
        "Expected /data/resnet_data/prob_samples or /data2/resnet_data/prob_samples."
    )

DATA_DIR  = _locate_data_dir()
TRAIN_DIR = os.path.join(SCRIPT_DIR, 'mlp_trainings')

FEATURE_VARS = [
    'fraction_zero', 'mixture_weight',
    'gamma_shape1',  'gamma_scale1',
    'gamma_shape2',  'gamma_scale2',
]
SEASONAL_VARS = ['sample_cos_doy', 'sample_sin_doy']
TERRAIN_VARS  = ['sample_local_std']
N_HOUR_FEATS  = 2   # cos/sin(hour-of-day), derived from sample_date at load
                    # time (2026-08: GRAF init cycle -- 00/06/12/18Z -- may
                    # carry systematic bias/climatology differences the
                    # model previously had no way to see).
N_INPUT = 36 + len(SEASONAL_VARS) + len(TERRAIN_VARS) + N_HOUR_FEATS

# Texture/persistence/terrain-gradient features (2026-08): computed from the
# RAW GRAF field (not the fitted gamma-mixture parameters, to avoid
# reintroducing the same summarization bottleneck those parameters already
# impose) by save_graf_texture_features.py, plus static terrain-gradient
# fields already used by the 1-h ResUNet. Requested via the --texture CLI
# flag, orthogonal to the loss-variant selection (see main()).
TEXTURE_SPATIAL_VARS = [
    'sample_wet_area_fraction', 'sample_peak_to_mean_ratio', 'sample_coeff_variation',
]   # 3 vars x 6 hours = 18
TEXTURE_TEMPORAL_VARS = [
    'sample_wetdry_jaccard', 'sample_zscore_pattern_corr',
]   # 2 vars x 5 consecutive-hour pairs = 10
RAW_PRECIP_VARS = ['sample_graf_precip_6h']   # 1
TERRAIN_GRAD_VARS = [
    'sample_terrain_height_local_difference',
    'sample_dterrain_dlon_smoothed', 'sample_dterrain_dlat_smoothed',
]   # 3
N_TEXTURE_FEATS = (6 * len(TEXTURE_SPATIAL_VARS) + 5 * len(TEXTURE_TEMPORAL_VARS)
                  + len(RAW_PRECIP_VARS) + len(TERRAIN_GRAD_VARS))   # 18+10+1+3 = 32
N_INPUT_TEXTURE = N_INPUT + N_TEXTURE_FEATS   # 41 + 32 = 73

# =========================================================================
# Model
# =========================================================================

class GammaMixtureMLP(nn.Module):
    """
    MLP that maps 36 hourly gamma-mixture features, 2 cyclic day-of-year
    features (cos/sin), and 1 local terrain-roughness feature to 6
    parameters for the 6-hourly zero-inflated two-component Gamma mixture
    distribution.

    Architecture: 38 → 72 → 144 → 72 → 36 → 12 → 6
    Activations:  ReLU after each BatchNorm layer.

    Label-switching fix: shape2 is reparameterized as
    shape1 + softplus(offset) + min_separation, a hard structural
    constraint (not a post-hoc sort) that guarantees component 2 is
    always the heavier tail and can never collapse onto component 1.
    Mirrors the equivalent constraint in the hourly ResUNet's
    GammaMixtureNLLLoss (pytorch_train_resunet_gamma_mixture.py).
    """

    def __init__(self, hidden_sizes=HIDDEN_SIZES,
                 shape_min=SHAPE_MIN, scale_min=SCALE_MIN, n_input=N_INPUT,
                 min_separation=MIN_SEPARATION,
                 dedicated_fz_head=False, fz_head_hidden=FZ_HEAD_HIDDEN):
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
                             nn.Dropout(DROPOUT_P)]

        # See GammaMixtureGRU for the rationale; dedicated_fz_head=False
        # preserves the original single-`net` structure exactly.
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

        raw = self.net(x)                                      # (batch, 6)
        frac_zero  = torch.sigmoid(raw[:, 0])
        mix_weight = torch.sigmoid(raw[:, 1])
        shape1     = self.shape_min + F.softplus(raw[:, 2])
        scale1     = self.scale_min + F.softplus(raw[:, 3])

        # Hard ordering constraint: shape2 = shape1 + softplus(offset) +
        # min_separation.  Guarantees shape2 > shape1 by construction, so
        # component 2 is always the heavier tail -- no post-hoc swap needed.
        shape2_offset = F.softplus(raw[:, 4])
        shape2        = shape1 + shape2_offset + self.min_separation
        scale2        = self.scale_min + F.softplus(raw[:, 5])

        return frac_zero, mix_weight, shape1, scale1, shape2, scale2


FILM_HIDDEN = 16


class GammaMixtureMLPFiLM(nn.Module):
    """
    Same hidden-layer sizes and 6-parameter output head as GammaMixtureMLP,
    but local terrain roughness (the last input column) is not concatenated
    into the trunk. Instead it conditions the trunk via FiLM: a small
    generator network maps terrain -> per-hidden-layer (gamma, beta), applied
    as h' = (1+gamma)*h + beta right after each BatchNorm (before ReLU).

    FiLM heads are zero-initialized so gamma=1, beta=0 at the start of
    training -- the model begins exactly equivalent to a terrain-blind trunk
    (matching GammaMixtureMLP's non-terrain layers) and only diverges where
    the data rewards using terrain.
    """

    def __init__(self, hidden_sizes=HIDDEN_SIZES,
                 shape_min=SHAPE_MIN, scale_min=SCALE_MIN, n_input=N_INPUT,
                 min_separation=MIN_SEPARATION, film_hidden=FILM_HIDDEN):
        super().__init__()
        self.shape_min = shape_min
        self.scale_min = scale_min
        self.min_separation = min_separation
        self.trunk_input = n_input - 1   # last column is the FiLM conditioning var

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
            h = F.dropout(h, p=DROPOUT_P, training=self.training)

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
N_HOURLY_FEATS_PER_HOUR = 9   # 6 gamma-mixture params (FEATURE_VARS) + 3
                              # texture-spatial stats (TEXTURE_SPATIAL_VARS)


def _gru_column_indices(n_input):
    """
    Column indices (into the flat N_INPUT_TEXTURE=73 feature vector built by
    load_data()/apply_mlp_fulldomain(), in that fixed block order) that
    belong to the per-hour sequence (hourly_idx, shape (6,9)) vs. everything
    else (static_idx, shape (19,)) -- season/hour-of-day/local terrain
    roughness (constant across the 6-hour window, not per-hour-varying),
    the pairwise temporal-persistence features, raw 6h GRAF precip, and the
    static terrain-gradient scalars.

    Hardcoded to this specific, currently-fixed block layout (hourly_feats,
    seasonal_feats, terrain_feats, hour_feats, texture_spatial_feats,
    texture_temporal_feats, raw_precip_feat, terrain_grad_feats -- see
    load_data()) rather than derived generically, since this model
    architecture is specific to the --texture (73-input) feature set; if
    that layout ever changes, this must be updated to match. The internal
    assert below is a cheap safety net against silent drift.
    """
    if n_input != N_INPUT_TEXTURE:
        raise ValueError(
            f'GammaMixtureGRU requires n_input=={N_INPUT_TEXTURE} (the --texture '
            f'feature set); got n_input={n_input}. Train with --texture.')

    # hourly_feats block [0:36]: FEATURE_VARS (6 vars) x 6 hours, var v/hour h -> v*6+h
    # texture_spatial_feats block [41:59]: TEXTURE_SPATIAL_VARS (3 vars) x 6 hours, offset 41
    hourly_idx = [
        [v * 6 + h for v in range(len(FEATURE_VARS))]
        + [41 + v * 6 + h for v in range(len(TEXTURE_SPATIAL_VARS))]
        for h in range(6)
    ]
    used = set(i for row in hourly_idx for i in row)
    static_idx = [i for i in range(n_input) if i not in used]

    assert sorted(used | set(static_idx)) == list(range(n_input)), \
        'GRU column-index derivation does not partition the full feature vector'
    return hourly_idx, static_idx


class GammaMixtureGRU(nn.Module):
    """
    Replaces GammaMixtureMLP's flat concatenation of the 6 hourly
    gamma-mixture params + texture-spatial stats with a small GRU over the
    6-hour sequence, so the model can learn cross-hour temporal relationships
    directly instead of relying solely on the hand-engineered pairwise
    persistence features (sample_wetdry_jaccard/sample_zscore_pattern_corr,
    still kept as additional static inputs here, not replaced -- this is an
    additive change, not a substitution, to avoid confounding "did the GRU
    help" with "did removing the hand-crafted features hurt").

    Requires n_input==N_INPUT_TEXTURE (73) -- see _gru_column_indices().
    Everything downstream of the GRU (trunk + 6-parameter head) is identical
    in structure to GammaMixtureMLP.
    """

    def __init__(self, hidden_sizes=HIDDEN_SIZES_TEXTURE,
                 shape_min=SHAPE_MIN, scale_min=SCALE_MIN, n_input=N_INPUT_TEXTURE,
                 min_separation=MIN_SEPARATION, gru_hidden=GRU_HIDDEN,
                 dedicated_fz_head=False, fz_head_hidden=FZ_HEAD_HIDDEN):
        super().__init__()
        self.shape_min = shape_min
        self.scale_min = scale_min
        self.min_separation = min_separation
        self.dedicated_fz_head = dedicated_fz_head

        hourly_idx, static_idx = _gru_column_indices(n_input)
        self.register_buffer('hourly_idx', torch.tensor(hourly_idx, dtype=torch.long))  # (6,9)
        self.register_buffer('static_idx', torch.tensor(static_idx, dtype=torch.long))  # (19,)

        self.gru = nn.GRU(input_size=N_HOURLY_FEATS_PER_HOUR, hidden_size=gru_hidden,
                          num_layers=1, batch_first=True)

        n_static = len(static_idx)
        layer_sizes = [gru_hidden + n_static] + hidden_sizes
        trunk_layers = []
        for in_sz, out_sz in zip(layer_sizes, layer_sizes[1:]):
            trunk_layers += [nn.Linear(in_sz, out_sz),
                             nn.BatchNorm1d(out_sz),
                             nn.ReLU(),
                             nn.Dropout(DROPOUT_P)]

        # dedicated_fz_head=False preserves the exact original single-`net`
        # structure (so every checkpoint trained before this option existed
        # keeps loading unchanged); =True splits the final layer into a
        # small dedicated fz branch (mirrors HRRRcal's AttnResUNetHRRR
        # dedicated_fz_head: frac_zero gets its own weights off the shared
        # trunk features, instead of sharing one linear projection with the
        # other 5 params) plus a main head for the rest.
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
        seq    = x[:, self.hourly_idx]    # (batch, 6, 9)
        static = x[:, self.static_idx]    # (batch, 19)

        _, h_n = self.gru(seq)            # h_n: (num_layers=1, batch, gru_hidden)
        gru_summary = h_n[-1]             # (batch, gru_hidden)
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
        else:
            raw = self.net(combined)
            frac_zero  = torch.sigmoid(raw[:, 0])
            mix_weight = torch.sigmoid(raw[:, 1])
            shape1     = self.shape_min + F.softplus(raw[:, 2])
            scale1     = self.scale_min + F.softplus(raw[:, 3])
            shape2_offset = F.softplus(raw[:, 4])
            shape2        = shape1 + shape2_offset + self.min_separation
            scale2        = self.scale_min + F.softplus(raw[:, 5])

        return frac_zero, mix_weight, shape1, scale1, shape2, scale2


BERNSTEIN_DEGREE = 10   # K; produces K+1=11 monotonic coefficients for the
                        # conditional-positive quantile function. Kept modest
                        # per the lightweight-by-default preference -- see
                        # feedback_6h_mlp_lightweight_preference in memory.


def _bernstein_basis_matrix(taus, degree):
    """(n_tau,) tau values in [0,1] -> (n_tau, degree+1) Bernstein basis
    matrix, B_k(tau) = C(degree,k) * tau^k * (1-tau)^(degree-k). Pure numpy;
    used both to build the fixed training tau-grid basis (below) and,
    identically, by reliability_6hourly_mlp_3panel.py's exceedance-
    probability inversion (evaluated at per-pixel, not fixed, tau values --
    same formula, just called elementwise there instead of on a grid)."""
    from scipy.special import comb
    K = degree
    taus = np.asarray(taus, dtype=np.float64)
    basis = np.zeros((len(taus), K + 1), dtype=np.float32)
    for k in range(K + 1):
        basis[:, k] = comb(K, k) * (taus ** k) * ((1.0 - taus) ** (K - k))
    return basis


class BernsteinGRU(nn.Module):
    """
    Same GRU-over-6-hours + trunk backbone as GammaMixtureGRU (identical
    hourly/static column split, identical trunk depth/width), but replaces
    the 2-component Gamma mixture's conditional-positive parameterization
    with a degree-BERNSTEIN_DEGREE Bernstein-polynomial quantile function.

    frac_zero keeps the EXACT same treatment as GammaMixtureGRU (sigmoid of
    one logit, trained with the same Bernoulli/logistic term via
    bernstein_hurdle_loss below) -- only the conditional-on-positive
    representation changes. This is a deliberate scoping decision (Tom,
    2026-08-09): naive quantile regression performed poorly in Tom's much
    older 1h-model notes, apparently because a single smooth quantile
    function can't represent the true mixed distribution's flat-then-kink
    shape (flat at 0 for the whole frac_zero probability mass, then a kink,
    then the continuous part) without smearing mass around the zero
    boundary. Keeping frac_zero's own hurdle-style handling untouched avoids
    re-deriving that delicate part from scratch inside the new family.

    Output: frac_zero (batch,), coeffs (batch, BERNSTEIN_DEGREE+1) -- the
    monotonic non-decreasing Bernstein coefficients b_0<=b_1<=...<=b_K of
    Q_pos(tau) = sum_k b_k * BernsteinBasis_k(tau), i.e. the quantile
    function of (Y | Y>0). Monotonicity (a valid quantile function must be
    non-decreasing) and non-negativity (b_0 = Q_pos(0) >= 0, since positive
    precipitation amounts are strictly positive) are both guaranteed by
    construction: raw increments pass through softplus (>0) and are
    cumsum'd (so the running sum is non-decreasing), rather than enforced
    via a penalty term that could be violated.
    """

    def __init__(self, hidden_sizes=HIDDEN_SIZES_TEXTURE, n_input=N_INPUT_TEXTURE,
                 degree=BERNSTEIN_DEGREE, gru_hidden=GRU_HIDDEN):
        super().__init__()
        self.degree = degree
        # No shape_min/scale_min for this family -- kept as None so
        # save_checkpoint()'s generic getattr(model, 'shape_min', None)
        # stays harmless rather than needing a Bernstein-specific branch.
        self.shape_min = None
        self.scale_min = None

        hourly_idx, static_idx = _gru_column_indices(n_input)
        self.register_buffer('hourly_idx', torch.tensor(hourly_idx, dtype=torch.long))
        self.register_buffer('static_idx', torch.tensor(static_idx, dtype=torch.long))

        self.gru = nn.GRU(input_size=N_HOURLY_FEATS_PER_HOUR, hidden_size=gru_hidden,
                          num_layers=1, batch_first=True)

        n_static = len(static_idx)
        layer_sizes = [gru_hidden + n_static] + hidden_sizes
        trunk_layers = []
        for in_sz, out_sz in zip(layer_sizes, layer_sizes[1:]):
            trunk_layers += [nn.Linear(in_sz, out_sz),
                             nn.BatchNorm1d(out_sz),
                             nn.ReLU(),
                             nn.Dropout(DROPOUT_P)]
        trunk_layers.append(nn.Linear(hidden_sizes[-1], 1 + (degree + 1)))
        self.net = nn.Sequential(*trunk_layers)

    def forward(self, x):
        seq    = x[:, self.hourly_idx]    # (batch, 6, 9)
        static = x[:, self.static_idx]    # (batch, 19)

        _, h_n = self.gru(seq)
        gru_summary = h_n[-1]
        combined = torch.cat([gru_summary, static], dim=1)

        raw = self.net(combined)
        frac_zero  = torch.sigmoid(raw[:, 0])
        increments = F.softplus(raw[:, 1:])           # (batch, degree+1), each > 0
        coeffs     = torch.cumsum(increments, dim=1)   # non-decreasing; coeffs[:,0] = increments[:,0] > 0

        return frac_zero, coeffs


# =========================================================================
# NLL Loss
# =========================================================================

def nll_loss(frac_zero, mix_weight, shape1, scale1, shape2, scale2, y):
    """
    Negative log-likelihood for a zero-inflated two-component Gamma mixture.

    torch.igamma has no gradient w.r.t. its shape argument, so CRPS is not
    differentiable.  NLL only requires lgamma / log, which are fully
    differentiable in PyTorch.

    Case y = 0 :  NLL = -log(frac_zero)
    Case y > 0 :  NLL = -log(1 - frac_zero)
                        - log[ w * GammaPDF(y; α1, θ1)
                               + (1-w) * GammaPDF(y; α2, θ2) ]

    log GammaPDF(y; α, θ) = (α-1)*log(y) - y/θ - α*log(θ) - lgamma(α)

    Parameters
    ----------
    frac_zero, mix_weight, shape1, scale1, shape2, scale2 : (batch,)
    y : (batch,)   observed 6-hourly MRMS precipitation (mm)

    Returns
    -------
    scalar mean NLL
    """
    eps = 1e-7

    def log_gamma_pdf(y_pos, alpha, theta):
        return ((alpha - 1.0) * torch.log(y_pos)
                - y_pos / theta
                - alpha * torch.log(theta)
                - torch.lgamma(alpha))

    zero_mask = (y == 0.0)
    pos_mask  = ~zero_mask

    nll = torch.empty_like(y)

    # --- y = 0 ---
    nll[zero_mask] = -torch.log(frac_zero[zero_mask] + eps)

    # --- y > 0 ---
    if pos_mask.any():
        y_p   = y[pos_mask]
        fz_p  = frac_zero[pos_mask]
        mw_p  = mix_weight[pos_mask]
        s1_p  = shape1[pos_mask]
        sc1_p = scale1[pos_mask]
        s2_p  = shape2[pos_mask]
        sc2_p = scale2[pos_mask]

        log_pdf1 = log_gamma_pdf(y_p, s1_p, sc1_p)
        log_pdf2 = log_gamma_pdf(y_p, s2_p, sc2_p)

        # log-mixture via logsumexp for numerical stability
        log_mix = torch.logaddexp(
            torch.log(mw_p + eps)       + log_pdf1,
            torch.log(1.0 - mw_p + eps) + log_pdf2,
        )

        nll[pos_mask] = -torch.log(1.0 - fz_p + eps) - log_mix

    nll = torch.clamp(nll, max=100.0)
    return nll.mean()


# =========================================================================
# CRPS loss (discretized, via reparameterized samples)
# =========================================================================

# Non-uniform bin edges for the discretized CRPS Riemann sum: fine near
# zero where most of the 6-h precip mass and CRPS sensitivity lives,
# coarser in the tail. First edge must be 0.0.
CRPS_BIN_EDGES = np.concatenate([
    [0.0],
    np.arange(0.25, 20.0, 0.25),
    np.arange(20.0, 60.0, 1.0),
    np.arange(60.0, 200.01, 5.0),
]).astype(np.float32)

CRPS_N_SAMPLES  = 32     # reparameterized Monte Carlo draws per Gamma component
CRPS_TEMPERATURE = 0.25  # mm; width of the sigmoid soft-indicator used to
                         # build a differentiable empirical CDF from samples


def crps_loss(frac_zero, mix_weight, shape1, scale1, shape2, scale2, y,
              n_samples=CRPS_N_SAMPLES, temperature=CRPS_TEMPERATURE,
              bin_edges=None):
    """
    Discretized CRPS for the zero-inflated two-component Gamma mixture.

    torch.special.gammainc's gradient w.r.t. its shape argument is not
    implemented in PyTorch (confirmed empirically, 2026-08), so the
    mixture's closed-form CDF can't be backpropagated through directly --
    that's why nll_loss() above uses the PDF (lgamma-based, fully
    differentiable) instead. torch.distributions.Gamma.rsample(), however,
    DOES support gradients through both shape and rate via implicit
    reparameterization (Figurnov et al. 2018), so CRPS is approximated
    here as a Riemann sum over the squared CDF difference

        CRPS(F, y) ~= sum_i (F_hat(x_i) - 1{x_i >= y})^2 * dx_i

    (the same discretized-CDF-difference structure as a masked ordinal/
    categorical CRPS loss, e.g. Ghazvinian et al.-style, adapted from a
    softmax head to this parametric mixture head), where F_hat(x) is a
    smoothed empirical CDF built from reparameterized per-component
    samples:

        F_hat(x) = frac_zero + (1-frac_zero) *
                   (w * mean_k[sigmoid((x - g1_k)/T)]
                    + (1-w) * mean_k[sigmoid((x - g2_k)/T)])

    Only the per-component CDF is approximated via samples; the mixture
    weight and zero-inflation terms mix the two component CDFs linearly
    (no relaxation needed for those, since they're already continuous
    scalars, not a discrete choice).

    Parameters
    ----------
    frac_zero, mix_weight, shape1, scale1, shape2, scale2 : (batch,)
    y : (batch,)   observed 6-hourly MRMS precipitation (mm)
    n_samples : int   Monte Carlo draws per component per call
    temperature : float   soft-indicator width (mm); smaller = closer to
        the true (non-differentiable) step CDF but noisier gradients
    bin_edges : optional override of CRPS_BIN_EDGES (mm), ascending, [0]=0.0

    Returns
    -------
    scalar mean discretized CRPS (mm)
    """
    device = y.device
    edges = torch.as_tensor(
        CRPS_BIN_EDGES if bin_edges is None else bin_edges,
        dtype=torch.float32, device=device)
    eval_pts = edges[1:]                 # (nbins,) right edge of each bin
    dx       = torch.diff(edges)         # (nbins,) bin widths

    rate1 = 1.0 / scale1
    rate2 = 1.0 / scale2
    g1 = torch.distributions.Gamma(shape1, rate1).rsample((n_samples,))  # (K, batch)
    g2 = torch.distributions.Gamma(shape2, rate2).rsample((n_samples,))  # (K, batch)

    # (nbins, K, batch) -> mean over K -> (nbins, batch)
    cdf1 = torch.sigmoid((eval_pts.view(-1, 1, 1) - g1.unsqueeze(0)) / temperature).mean(dim=1)
    cdf2 = torch.sigmoid((eval_pts.view(-1, 1, 1) - g2.unsqueeze(0)) / temperature).mean(dim=1)

    fz = frac_zero.unsqueeze(0)
    mw = mix_weight.unsqueeze(0)
    cdf_mix = fz + (1.0 - fz) * (mw * cdf1 + (1.0 - mw) * cdf2)   # (nbins, batch)

    obs_cdf = (eval_pts.view(-1, 1) >= y.view(1, -1)).float()      # (nbins, batch)

    sq_diff = (obs_cdf - cdf_mix) ** 2
    per_sample_crps = (sq_diff * dx.view(-1, 1)).sum(dim=0)        # (batch,)
    return per_sample_crps.mean()


def hurdle_loss(frac_zero, mix_weight, shape1, scale1, shape2, scale2, y,
                n_samples=CRPS_N_SAMPLES, temperature=CRPS_TEMPERATURE,
                bin_edges=None, crps_weight=1.0):
    """
    Hurdle-model loss (Cragg 1971): exact NLL for the Bernoulli zero/
    positive split, plus discretized CRPS (via reparameterized samples,
    see crps_loss() above) for the conditional-on-positive two-component
    Gamma mixture. A hurdle model's joint likelihood factorizes exactly
    into a discrete part and a continuous part with no cross terms:

        loss = NLL_Bernoulli(1{y=0}; frac_zero)
               + 1{y>0} * CRPS(conditional mixture, y)

    so scoring each part with a different (but each individually
    well-motivated) rule is not an ad hoc hybrid -- it's a legitimate
    choice within a standard decomposition. This specifically targets the
    gradient-vanishing-near-extremes property of CRPS/Brier-family scores
    (see crps_loss() docstring): NLL keeps a strong, non-vanishing
    gradient on frac_zero even as it approaches 0 or 1, which matters
    here because 69% of training samples are exact zeros.

    The conditional mixture's CDF is NOT multiplied by (1-frac_zero) --
    unlike crps_loss(), this scores the distribution given that y>0 is
    already known, so it's a proper CDF on (0, inf) in its own right.

    Parameters
    ----------
    frac_zero, mix_weight, shape1, scale1, shape2, scale2 : (batch,)
    y : (batch,)   observed 6-hourly MRMS precipitation (mm)
    crps_weight : float   relative weight of the CRPS term (NLL and CRPS
        are on different scales -- nats vs. mm -- with no canonical
        exchange rate; 1.0 is an unweighted sum, tune if one term
        dominates during training)

    Returns
    -------
    scalar mean hurdle loss
    """
    eps = 1e-7
    zero_mask = (y == 0.0)
    pos_mask  = ~zero_mask

    bernoulli_nll = torch.empty_like(y)
    bernoulli_nll[zero_mask] = -torch.log(frac_zero[zero_mask] + eps)
    bernoulli_nll[pos_mask]  = -torch.log(1.0 - frac_zero[pos_mask] + eps)
    bernoulli_nll = torch.clamp(bernoulli_nll, max=100.0)
    loss_bernoulli = bernoulli_nll.mean()

    if pos_mask.any():
        y_p   = y[pos_mask]
        mw_p  = mix_weight[pos_mask]
        s1_p  = shape1[pos_mask]
        sc1_p = scale1[pos_mask]
        s2_p  = shape2[pos_mask]
        sc2_p = scale2[pos_mask]

        device = y.device
        edges = torch.as_tensor(
            CRPS_BIN_EDGES if bin_edges is None else bin_edges,
            dtype=torch.float32, device=device)
        eval_pts = edges[1:]
        dx       = torch.diff(edges)

        g1 = torch.distributions.Gamma(s1_p, 1.0 / sc1_p).rsample((n_samples,))
        g2 = torch.distributions.Gamma(s2_p, 1.0 / sc2_p).rsample((n_samples,))

        cdf1 = torch.sigmoid((eval_pts.view(-1, 1, 1) - g1.unsqueeze(0)) / temperature).mean(dim=1)
        cdf2 = torch.sigmoid((eval_pts.view(-1, 1, 1) - g2.unsqueeze(0)) / temperature).mean(dim=1)

        mw_v = mw_p.unsqueeze(0)
        cdf_cond = mw_v * cdf1 + (1.0 - mw_v) * cdf2          # (nbins, n_pos); no frac_zero term

        obs_cdf = (eval_pts.view(-1, 1) >= y_p.view(1, -1)).float()
        sq_diff = (obs_cdf - cdf_cond) ** 2
        per_sample_crps = (sq_diff * dx.view(-1, 1)).sum(dim=0)
        loss_crps = per_sample_crps.mean()
    else:
        loss_crps = torch.zeros((), device=y.device)

    return loss_bernoulli + crps_weight * loss_crps


# =========================================================================
# Reliability-weighted loss add-on (differentiable REL term from the
# Murphy 1973 / Hersbach 2000 CRPS decomposition, CRPS = RELI - RESO + UNC)
# =========================================================================
#
# Since UNC doesn't depend on the forecast, a weighted combination
# w1*RELI - w2*RESO is, up to an additive constant, algebraically identical
# to w2*base_loss + (w1-w2)*RELI -- "reweight reliability vs. resolution"
# and "base loss plus an extra reliability penalty" are the same loss
# family. This implements the latter: base_loss_fn(...) + reli_weight *
# RELI, where RELI is Murphy (1973)'s per-threshold Brier-score reliability
# term at fixed exceedance thresholds (matching the reliability diagrams'
# own 0.25/2.5/10mm thresholds), soft-binned by forecast probability (11
# nodes, 0-100% by 10%, matching reliability_6hourly_mlp_3panel.py's
# ncats=11) so it stays differentiable.
#
# Since this is additive on top of the base loss (not a replacement),
# collapsing to a low-resolution/climatology-like forecast is not free --
# it still pays the base loss's own resolution-sensitive cost. This only
# shifts where on the achievable reliability/resolution tradeoff curve
# training settles, given a strictly proper base loss already has RELI=0
# at its (unreachable, in finite-sample practice) population optimum.

RELI_THRESHOLDS = torch.tensor([0.25, 2.5, 10.0])   # mm; matches the 3
                                                     # evaluation thresholds
RELI_PROB_NODES  = torch.linspace(0.0, 1.0, 11)      # matches ncats=11 in
                                                      # reliability_6hourly_mlp_3panel.py
RELI_KERNEL_BW   = 0.1                               # soft-bin bandwidth,
                                                      # matches the 10%-wide
                                                      # hard bins it approximates

_reli_component_log = []   # populated only when reli_weight > 0; read/cleared
                          # by main() once per run_epoch() call, for the
                          # per-epoch diagnostic printout


def exceedance_prob_mc(mix_weight, shape1, scale1, shape2, scale2, thresholds,
                       n_samples=CRPS_N_SAMPLES, temperature=CRPS_TEMPERATURE):
    """
    Differentiable P(Y_pos >= thresh) for the conditional-on-positive
    two-component Gamma mixture, at each of `thresholds`, via the same
    reparameterized-sample + sigmoid soft-indicator trick crps_loss() uses
    (torch.special.gammaincc's gradient w.r.t. shape is not implemented in
    PyTorch, so the closed-form survival function can't be backpropped
    through directly).

    Parameters
    ----------
    mix_weight, shape1, scale1, shape2, scale2 : (batch,)
    thresholds : (T,) tensor, mm

    Returns
    -------
    (T, batch) tensor of P(Y_pos >= thresh)
    """
    g1 = torch.distributions.Gamma(shape1, 1.0 / scale1).rsample((n_samples,))  # (K,batch)
    g2 = torch.distributions.Gamma(shape2, 1.0 / scale2).rsample((n_samples,))
    thr = thresholds.to(g1.device).view(-1, 1, 1)                              # (T,1,1)
    sf1 = torch.sigmoid((g1.unsqueeze(0) - thr) / temperature).mean(dim=1)      # (T,batch)
    sf2 = torch.sigmoid((g2.unsqueeze(0) - thr) / temperature).mean(dim=1)
    mw = mix_weight.unsqueeze(0)
    return mw * sf1 + (1.0 - mw) * sf2


def reliability_penalty(frac_zero, mix_weight, shape1, scale1, shape2, scale2, y,
                        thresholds=None, prob_nodes=None, kernel_bw=RELI_KERNEL_BW,
                        n_samples=CRPS_N_SAMPLES, thresh_weights=None):
    """
    Differentiable approximation of Murphy (1973)'s per-threshold Brier-
    score reliability term:

        REL(thresh) = sum_k  n_k * (mean_p_k - obs_freq_k)^2  / N

    computed at each of `thresholds` and averaged, using soft (Gaussian-
    kernel) bin membership in place of hard binning so it's differentiable
    w.r.t. the forecast probability p. This is a per-threshold analogue of
    Hersbach (2000)'s CRPS-level RELI term (Eqs. 10-12), not the full
    ensemble-rank-binned version -- tractable here since exceedance
    probability at a fixed threshold is already what apply_mlp_fulldomain()
    computes for the reliability diagrams downstream, so training-time and
    eval-time are measuring the same quantity.

    Parameters
    ----------
    frac_zero, mix_weight, shape1, scale1, shape2, scale2 : (batch,)
    y : (batch,)   observed 6-hourly MRMS precipitation (mm)
    thresh_weights : optional (T,) tensor; if given, a weighted (rather
        than equal) average of the per-threshold REL values, normalised to
        sum to 1. Default (None) is an equal-weighted mean.

    Returns
    -------
    scalar: (weighted) mean REL across `thresholds`
    """
    if thresholds is None:
        thresholds = RELI_THRESHOLDS
    if prob_nodes is None:
        prob_nodes = RELI_PROB_NODES
    thresholds = thresholds.to(y.device)
    prob_nodes = prob_nodes.to(y.device)

    p_pos = exceedance_prob_mc(mix_weight, shape1, scale1, shape2, scale2,
                               thresholds, n_samples=n_samples)   # (T, batch)
    fz = frac_zero.unsqueeze(0)
    p = (1.0 - fz) * p_pos                                        # (T, batch)

    obs = (y.unsqueeze(0) >= thresholds.view(-1, 1)).float()      # (T, batch)

    diff = p.unsqueeze(-1) - prob_nodes.view(1, 1, -1)             # (T, batch, K)
    w = torch.exp(-0.5 * (diff / kernel_bw) ** 2)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)

    n_k    = w.sum(dim=1)                                          # (T, K)
    pbar_k = (w * p.unsqueeze(-1)).sum(dim=1) / (n_k + 1e-8)
    obar_k = (w * obs.unsqueeze(-1)).sum(dim=1) / (n_k + 1e-8)

    rel_per_thresh = (n_k * (pbar_k - obar_k) ** 2).sum(dim=1) / p.shape[1]   # (T,)

    if thresh_weights is None:
        return rel_per_thresh.mean()
    tw = thresh_weights.to(y.device)
    tw = tw / (tw.sum() + 1e-8)
    return (rel_per_thresh * tw).sum()


def make_reliability_weighted_loss(base_loss_fn, reli_weight, thresh_weights=None):
    """Wraps any of nll_loss/crps_loss/hurdle_loss (same 7-arg signature) to
    add reli_weight * reliability_penalty(...) on top. Logs the raw
    (unweighted-by-reli_weight, but already thresh_weights-combined) REL
    value to _reli_component_log for main()'s per-epoch diagnostic printout.

    thresh_weights lets the REL term emphasize specific exceedance
    thresholds instead of the default equal average across all 3 -- added
    2026-08-12 after an equal-weighted pilot showed the optimizer
    preferentially "spends" a large reli_weight on whichever threshold has
    the most samples / least noisy estimate (0.25mm), at the expense of the
    thresholds with thinner data (2.5/10mm) -- exactly backwards when 10mm
    is the threshold that actually needs help."""
    def wrapped(frac_zero, mix_weight, shape1, scale1, shape2, scale2, y):
        base = base_loss_fn(frac_zero, mix_weight, shape1, scale1, shape2, scale2, y)
        rel  = reliability_penalty(frac_zero, mix_weight, shape1, scale1, shape2, scale2, y,
                                   thresh_weights=thresh_weights)
        _reli_component_log.append(rel.item())
        return base + reli_weight * rel
    return wrapped


# =========================================================================
# Bernstein-quantile hurdle loss (for BernsteinGRU)
# =========================================================================

N_QUANTILE_LOSS_POINTS = 39   # fixed tau-grid density for the pinball-loss
                              # training objective below
_TAU_GRID_NP = np.linspace(0.025, 0.975, N_QUANTILE_LOSS_POINTS).astype(np.float32)
_TAU_BASIS_NP = _bernstein_basis_matrix(_TAU_GRID_NP, BERNSTEIN_DEGREE)   # (n_tau, degree+1)
_TAU_GRID_T  = torch.tensor(_TAU_GRID_NP)     # (n_tau,)
_TAU_BASIS_T = torch.tensor(_TAU_BASIS_NP)    # (n_tau, degree+1)


def bernstein_hurdle_loss(frac_zero, coeffs, y):
    """
    Hurdle-style loss for BernsteinGRU: exact Bernoulli NLL for the zero/
    positive split (identical in form to hurdle_loss's zero term above --
    same reasoning: NLL keeps a non-vanishing gradient on frac_zero even
    near 0/1), plus an average pinball (quantile) loss over a fixed
    39-point tau-grid for the conditional-on-positive Bernstein quantile
    function.

    The grid-pinball form is a discretized approximation to the CRPS of a
    quantile function (CRPS = integral over tau of the pinball loss at that
    tau -- see Gneiting & Ranjan 2011's quantile decomposition of CRPS);
    an exact closed form exists for Bernstein quantile functions specifically
    (via incomplete-beta-function bookkeeping, as in the BQN literature) but
    the grid form is simpler to implement and verify correctness of, and
    converges to the same quantity as the grid density increases. Not
    weighted/combined with anything else beyond the unweighted sum used by
    hurdle_loss, for the same reason: no canonical exchange rate between
    the Bernoulli NLL (nats) and the pinball loss (mm) scales, so start
    from the simplest unweighted choice and only add a weight if training
    dynamics show one term dominating.

    Parameters
    ----------
    frac_zero : (batch,)
    coeffs    : (batch, BERNSTEIN_DEGREE+1)  monotonic Bernstein coefficients
    y : (batch,)   observed 6-hourly MRMS precipitation (mm)
    """
    eps = 1e-7
    zero_mask = (y == 0.0)
    pos_mask  = ~zero_mask

    bernoulli_nll = torch.empty_like(y)
    bernoulli_nll[zero_mask] = -torch.log(frac_zero[zero_mask] + eps)
    bernoulli_nll[pos_mask]  = -torch.log(1.0 - frac_zero[pos_mask] + eps)
    bernoulli_nll = torch.clamp(bernoulli_nll, max=100.0)
    loss_bernoulli = bernoulli_nll.mean()

    if pos_mask.any():
        device = y.device
        basis = _TAU_BASIS_T.to(device)      # (n_tau, degree+1)
        taus  = _TAU_GRID_T.to(device)        # (n_tau,)

        coeffs_pos = coeffs[pos_mask]                    # (n_pos, degree+1)
        y_pos      = y[pos_mask].unsqueeze(1)             # (n_pos, 1)
        q_pred     = coeffs_pos @ basis.T                  # (n_pos, n_tau)
        diff       = y_pos - q_pred                          # (n_pos, n_tau)
        pinball    = torch.maximum(taus * diff, (taus - 1.0) * diff)
        loss_positive = pinball.mean()
    else:
        loss_positive = torch.zeros((), device=y.device)

    return loss_bernoulli + loss_positive


# =========================================================================
# Data loading
# =========================================================================

def load_data(clead, use_texture=False, precip_transform='log1p'):
    """
    Load all prob_MRMS_samples_*_lead{clead}h.nc files from DATA_DIR.

    Parameters
    ----------
    use_texture : bool  if True, also load the texture/persistence/terrain-
        gradient feature blocks (N_INPUT_TEXTURE columns total); these
        require prob_MRMS_samples_*_lead{clead}h.nc to have been generated
        by a sample_6hourly_prob_mrms.py that read save_graf_texture_features.py
        output -- older sample files will raise KeyError below.
    precip_transform : 'log1p' or 'sqrt'  transform applied to the raw
        sample_graf_precip_6h feature only (2026-08-10 experiment: the 1h
        ResUNet uses sqrt/power=0.5 on GRAF precip, this MLP has always used
        log1p -- testing whether matching the 1h model's transform helps or
        whether the two features' different math properties near zero/in
        the tail, discussed at length with Tom, actually matter more than
        consistency). Affects only this one feature; every other transform
        in this function is unchanged.

    Returns
    -------
    features : np.ndarray  (N, N_INPUT or N_INPUT_TEXTURE)
        36 hourly params + cos/sin day-of-year + local terrain roughness
        + cos/sin hour-of-day [+ texture spatial/temporal + raw GRAF 6h
        total + terrain gradient/deviation, appended in that fixed order
        if use_texture].  Column order must exactly match
        apply_mlp_fulldomain() in reliability_6hourly_mlp_3panel.py.
    targets  : np.ndarray  (N,)
    dates    : np.ndarray  (N,)   init date/time as YYYYMMDDHH (int)
    """
    pattern = os.path.join(DATA_DIR, f'prob_MRMS_samples_*_lead{clead}h.nc')
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f'No data files found matching:\n  {pattern}\n'
            f'Run sample_6hourly_prob_mrms.py first.')

    print(f'Found {len(files)} data file(s):')
    for f in files:
        print(f'  {f}')

    required_vars = ['sample_date'] + SEASONAL_VARS + TERRAIN_VARS
    if use_texture:
        required_vars += TEXTURE_SPATIAL_VARS + TEXTURE_TEMPORAL_VARS + RAW_PRECIP_VARS + TERRAIN_GRAD_VARS

    feat_list   = []
    target_list = []
    date_list   = []

    for fpath in files:
        with Dataset(fpath, 'r') as ds:
            for vname in required_vars:
                if vname not in ds.variables:
                    raise KeyError(
                        f'{fpath} has no "{vname}" variable. '
                        f'Re-run sample_6hourly_prob_mrms.py to regenerate it.')
            blocks = []
            for vname in FEATURE_VARS:
                arr = ds[vname][:].data.astype(np.float32)   # (nsamples, 6)
                blocks.append(arr)
            hourly_feats = np.concatenate(blocks, axis=1)      # (nsamples, 36)
            seasonal_feats = np.stack(
                [ds[vname][:].data.astype(np.float32) for vname in SEASONAL_VARS],
                axis=1)                                        # (nsamples, 2)
            terrain_feats = np.stack(
                [np.log1p(ds[vname][:].data.astype(np.float32)) for vname in TERRAIN_VARS],
                axis=1)                                        # (nsamples, 1); log1p tames terrain's skewed tail
            sample_date  = ds['sample_date'][:].data.astype(np.int64)
            hour_of_day  = (sample_date % 100).astype(np.float32)   # cycle: 00/06/12/18
            angle        = 2.0 * math.pi * hour_of_day / 24.0
            hour_feats   = np.stack([np.cos(angle), np.sin(angle)], axis=1)  # (nsamples, 2)

            all_blocks = [hourly_feats, seasonal_feats, terrain_feats, hour_feats]

            if use_texture:
                # Spatial (already bounded [0,1] wet_area_fraction aside;
                # peak_to_mean/coeff_variation are skewed ratios -> log1p)
                texture_spatial_blocks = []
                for vname in TEXTURE_SPATIAL_VARS:
                    arr = ds[vname][:].data.astype(np.float32)   # (nsamples, 6)
                    if vname in ('sample_peak_to_mean_ratio', 'sample_coeff_variation'):
                        arr = np.log1p(arr)
                    texture_spatial_blocks.append(arr)
                texture_spatial_feats = np.concatenate(texture_spatial_blocks, axis=1)  # (nsamples, 18)

                # Temporal: Jaccard and z-scored correlation are both
                # already bounded ([0,1] and [-1,1] resp.) -- no transform.
                texture_temporal_feats = np.concatenate(
                    [ds[vname][:].data.astype(np.float32) for vname in TEXTURE_TEMPORAL_VARS],
                    axis=1)   # (nsamples, 10)

                # Raw GRAF 6h total: skewed/zero-inflated like target_precip_6h.
                # precip_transform selects log1p (default) or sqrt -- see
                # load_data()'s docstring for why this is worth testing.
                raw_precip_vals = ds[RAW_PRECIP_VARS[0]][:].data.astype(np.float32)
                if precip_transform == 'sqrt':
                    raw_precip_feat = np.sqrt(raw_precip_vals)[:, np.newaxis]
                else:
                    raw_precip_feat = np.log1p(raw_precip_vals)[:, np.newaxis]  # (nsamples, 1)

                # Terrain gradient/deviation: deviation is signed and
                # heavy-tailed (sign-preserving log1p); dlon/dlat are
                # already small/well-scaled (ResUNet normalizes them by a
                # fixed max of 0.02) -- left untransformed.
                terrain_diff = ds[TERRAIN_GRAD_VARS[0]][:].data.astype(np.float32)
                terrain_diff = np.sign(terrain_diff) * np.log1p(np.abs(terrain_diff))
                terrain_dlon = ds[TERRAIN_GRAD_VARS[1]][:].data.astype(np.float32)
                terrain_dlat = ds[TERRAIN_GRAD_VARS[2]][:].data.astype(np.float32)
                terrain_grad_feats = np.stack([terrain_diff, terrain_dlon, terrain_dlat], axis=1)  # (nsamples, 3)

                all_blocks += [texture_spatial_feats, texture_temporal_feats,
                              raw_precip_feat, terrain_grad_feats]

            feat_list.append(np.concatenate(all_blocks, axis=1))
            target_list.append(ds['target_precip_6h'][:].data.astype(np.float32))
            date_list.append(sample_date)

    features = np.concatenate(feat_list,  axis=0)   # (N, N_INPUT or N_INPUT_TEXTURE)
    targets  = np.concatenate(target_list, axis=0)  # (N,)
    dates    = np.concatenate(date_list,   axis=0)  # (N,)
    print(f'Total samples: {len(targets):,}')
    print(f'  Wet fraction : {(targets > 0).mean():.3f}')
    print(f'  Mean precip  : {targets.mean():.3f} mm')
    print(f'  Max precip   : {targets.max():.3f} mm')
    return features, targets, dates


def split_by_day_block(dates, stride=VAL_BLOCK_STRIDE):
    """
    Assign each sample to train or validation using sequential calendar-day
    blocks, resetting at each month boundary: day-of-month 1..(stride-1) are
    training, every stride-th day-of-month (5, 10, 15, ...) is validation.
    Both 00Z and 12Z inits of the same date land in the same split.

    Parameters
    ----------
    dates  : np.ndarray (N,)  init date/time as YYYYMMDDHH (int)
    stride : int               validation block period, in days

    Returns
    -------
    train_idx, val_idx : np.ndarray of indices into `dates`
    """
    day_of_month = (dates // 100) % 100
    val_mask     = (day_of_month % stride == 0)

    train_idx = np.where(~val_mask)[0]
    val_idx   = np.where(val_mask)[0]
    return train_idx, val_idx


# =========================================================================
# Climatological output-layer initialisation
# =========================================================================

def _inv_softplus(y, min_val=0.0):
    """Inverse of softplus shifted by min_val.  Solves min_val + softplus(x) = y."""
    z = y - min_val          # should be > 0
    z = max(z, 1e-6)
    # softplus(x) = log(1+exp(x)); inverse: x = log(exp(z)-1)
    if z > 20.0:
        return float(z)      # softplus(x) ≈ x for large x
    return float(math.log(math.exp(z) - 1.0))


MAX_EM_WET_SAMPLES = 50000   # sub-sample wet targets for EM speed, matches
                             # pytorch_train_resunet_gamma_mixture.py's precedent


def fit_em_climatology(targets, min_separation):
    """
    Fit a real 2-component Gamma mixture (EM) to the observed 6-hourly
    totals, mirroring pytorch_train_resunet_gamma_mixture.py's
    compute_climatology()/initialize_output_layer() -- but applied to the
    6-hourly target instead of hourly, so the MLP's two components start
    from genuinely different light/heavy regimes instead of an identical
    (symmetric) pair.
    """
    frac_zero_clim = np.clip(float((targets == 0).mean()), 1e-4, 1 - 1e-4)
    wet = targets[targets > 0].astype(np.float64)

    if len(wet) < 100:
        print('WARNING: too few wet samples for EM climatology fit; '
              'falling back to a fixed light/heavy split.')
        return dict(frac_zero=frac_zero_clim, weight1=0.5,
                    shape1=0.8, scale1=1.0, shape2=0.8 + min_separation + 1.0, scale2=3.0)

    if len(wet) > MAX_EM_WET_SAMPLES:
        rng = np.random.default_rng(RANDOM_SEED)
        wet = rng.choice(wet, size=MAX_EM_WET_SAMPLES, replace=False)

    try:
        weights, shapes, scales, model = fit_gamma_mixture(
            wet, n_components=2, init_method='moments', verbose=False, max_iter=500)
        sort_idx = np.argsort(shapes)      # light -> heavy
        weights, shapes, scales = weights[sort_idx], shapes[sort_idx], scales[sort_idx]
        weight1, shape1, scale1 = float(weights[0]), float(shapes[0]), float(scales[0])
        shape2, scale2 = float(shapes[1]), float(scales[1])
        print(f'EM climatology converged after {model.n_iter_} iterations '
              f'(log-lik={model.loglik_:.2f})')
    except Exception as exc:
        print(f'WARNING: EM fit failed ({exc}); falling back to percentile split.')
        weight1, shape1, scale1 = 0.5, 0.8, float(np.percentile(wet, 25))
        shape2, scale2 = 0.8 + min_separation + 1.0, float(np.percentile(wet, 75))

    # Enforce the model's own hard-separation floor so the offset target
    # below is guaranteed positive.
    shape2 = max(shape2, shape1 + min_separation + 0.1)

    return dict(frac_zero=frac_zero_clim, weight1=weight1,
                shape1=shape1, scale1=scale1, shape2=shape2, scale2=scale2)


def init_output_layer(model, targets):
    """
    Initialise the bias of the final Linear layer from an EM-fit 2-
    component Gamma mixture climatology of the observed 6-hourly totals
    (see fit_em_climatology), rather than a symmetric single-Gamma
    climatology -- the two components start out already distinguishable,
    matching the hourly ResUNet's initialization strategy.
    """
    clim = fit_em_climatology(targets, model.min_separation)

    logit_fz = math.log(clim['frac_zero'] / (1.0 - clim['frac_zero']))
    w1 = np.clip(clim['weight1'], 1e-4, 1 - 1e-4)
    logit_w1 = math.log(w1 / (1.0 - w1))

    b2 = _inv_softplus(clim['shape1'], SHAPE_MIN)
    b3 = _inv_softplus(clim['scale1'], SCALE_MIN)
    shape2_offset_target = clim['shape2'] - clim['shape1'] - model.min_separation
    b4 = _inv_softplus(max(shape2_offset_target, 0.1), 0.0)
    b5 = _inv_softplus(clim['scale2'], SCALE_MIN)

    with torch.no_grad():
        if getattr(model, 'dedicated_fz_head', False):
            # Two separate final layers: fz_head's last Linear (1 output)
            # and main_head (5 outputs: mix_weight, shape1, scale1,
            # shape2_offset, scale2), indices shifted by one vs. the
            # single-head case since frac_zero isn't index 0 here.
            model.fz_head[-1].bias[0].fill_(logit_fz)
            model.main_head.bias[0].fill_(logit_w1)
            model.main_head.bias[1].fill_(b2)
            model.main_head.bias[2].fill_(b3)
            model.main_head.bias[3].fill_(b4)
            model.main_head.bias[4].fill_(b5)
        else:
            final_layer = model.net[-1] if hasattr(model, 'net') else model.output_layer
            final_layer.bias[0].fill_(logit_fz)       # frac_zero
            final_layer.bias[1].fill_(logit_w1)       # mix_weight -> weight1
            final_layer.bias[2].fill_(b2)             # shape1
            final_layer.bias[3].fill_(b3)             # scale1
            final_layer.bias[4].fill_(b4)             # shape2_offset
            final_layer.bias[5].fill_(b5)             # scale2

    print(f"EM climatology init: frac_zero={clim['frac_zero']:.3f}  "
          f"weight1={clim['weight1']:.3f}  "
          f"comp1(shape={clim['shape1']:.3f}, scale={clim['scale1']:.3f}, "
          f"mean={clim['shape1']*clim['scale1']:.3f}mm)  "
          f"comp2(shape={clim['shape2']:.3f}, scale={clim['scale2']:.3f}, "
          f"mean={clim['shape2']*clim['scale2']:.3f}mm)")


def init_bernstein_output_layer(model, targets):
    """
    Analogous to init_output_layer(), but for BernsteinGRU: instead of
    EM-fitting a Gamma mixture, this just needs empirical QUANTILES of the
    observed positive 6-hourly totals -- a simpler climatology fit than the
    Gamma case, since quantiles are exactly what the Bernstein coefficients
    represent (no distributional-family fitting needed at all).

    Sets the final-layer bias so that, before any input-dependent
    variation, the model's default output already matches the empirical
    climatological quantiles at tau=k/degree, k=0..degree, and frac_zero
    matches the empirical zero fraction -- same spirit as
    init_output_layer(), adapted to this parameterization.
    """
    frac_zero_clim = float(np.mean(targets == 0.0))
    frac_zero_clim = np.clip(frac_zero_clim, 1e-4, 1 - 1e-4)
    logit_fz = math.log(frac_zero_clim / (1.0 - frac_zero_clim))

    wet = targets[targets > 0.0]
    degree = model.degree
    taus = np.array([k / degree for k in range(degree + 1)], dtype=np.float64)
    quantiles = np.quantile(wet, taus)
    # Guard against ties/non-strict-increase at the low end (common with a
    # discretized/rounded MRMS record) -- increments must be > 0 for the
    # inverse-softplus below to be well-defined.
    quantiles = np.maximum.accumulate(quantiles)

    increments_target = np.empty(degree + 1)
    increments_target[0] = max(quantiles[0], 1e-3)
    increments_target[1:] = np.maximum(np.diff(quantiles), 1e-3)
    biases = [_inv_softplus(v, 0.0) for v in increments_target]

    with torch.no_grad():
        final_layer = model.net[-1]
        final_layer.bias[0].fill_(logit_fz)
        for k, b in enumerate(biases):
            final_layer.bias[1 + k].fill_(b)

    print(f'Bernstein climatology init: frac_zero={frac_zero_clim:.3f}  '
         f'quantiles(tau=0,{1/degree:.2f},...,1)={np.round(quantiles, 2)}')


# =========================================================================
# Checkpoint helpers
# =========================================================================

def checkpoint_path(clead, variant=None, use_texture=False, use_gru=False, use_fzhead=False,
                    use_bernstein=False, seed=None, use_sqrt_precip=False, reli_weight=0.0,
                    reli_thresh_weights=None):
    os.makedirs(TRAIN_DIR, exist_ok=True)
    suffix = f'_{variant}' if variant else ''
    suffix += '_texture' if use_texture else ''
    suffix += '_gru' if use_gru else ''
    suffix += '_bernstein' if use_bernstein else ''
    suffix += '_fzhead' if use_fzhead else ''
    suffix += '_sqrtprecip' if use_sqrt_precip else ''
    suffix += f'_reliw{reli_weight:g}' if reli_weight else ''
    if reli_weight and reli_thresh_weights is not None:
        suffix += '_thw' + '-'.join(f'{w:g}' for w in reli_thresh_weights)
    suffix += f'_seed{seed}' if seed is not None else ''
    return os.path.join(TRAIN_DIR, f'6h_mlp_lead{clead}h{suffix}.pth')


def save_checkpoint(path, model, optimizer, scheduler, epoch,
                    best_val_nll, feat_mean, feat_std, clead, architecture='concat',
                    loss_type='NLL', n_input=N_INPUT, hidden_sizes=HIDDEN_SIZES,
                    dedicated_fz_head=False, bernstein_degree=None, precip_transform='log1p',
                    reli_weight=0.0, reli_thresh_weights=None):
    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch':          epoch,
        'best_val_nll':  best_val_nll,   # holds CRPS value when loss_type='CRPS'
        'dedicated_fz_head': dedicated_fz_head,
        'bernstein_degree': bernstein_degree,   # None unless architecture=='bernstein_gru'
        'precip_transform': precip_transform,   # 'log1p' (default) or 'sqrt'
        'reli_weight':    reli_weight,   # training-time only; doesn't affect forward()
        'reli_thresh_weights': reli_thresh_weights,   # training-time only; doesn't affect forward()
        'feature_mean':   feat_mean,
        'feature_std':    feat_std,
        'shape_min':      getattr(model, 'shape_min', None),
        'scale_min':      getattr(model, 'scale_min', None),
        'hidden_sizes':   hidden_sizes,
        'n_input':        n_input,
        'clead':          clead,
        'architecture':   architecture,
        'loss_type':      loss_type,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt['epoch'], ckpt['best_val_nll'], ckpt['feature_mean'], ckpt['feature_std']


# =========================================================================
# Training
# =========================================================================

def make_loaders(features, targets, feat_mean, feat_std, train_idx, val_idx):
    """Normalise and split into train/val DataLoaders using precomputed
    day-block indices (see split_by_day_block)."""
    feats_norm = (features - feat_mean) / feat_std

    X = torch.tensor(feats_norm, dtype=torch.float32)
    y = torch.tensor(targets,    dtype=torch.float32)

    train_idx = torch.as_tensor(train_idx, dtype=torch.long)
    val_idx   = torch.as_tensor(val_idx,   dtype=torch.long)

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds   = TensorDataset(X[val_idx],   y[val_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    return train_loader, val_loader


def run_epoch(loader, model, loss_fn, optimizer=None, is_bernstein=False):
    """One forward pass over loader; returns mean loss (NLL/CRPS/hurdle for
    the Gamma-mixture models, or the Bernstein-hurdle loss for BernsteinGRU
    -- is_bernstein selects which output arity to unpack, since BernsteinGRU
    returns (frac_zero, coeffs) rather than the 6 Gamma-mixture params)."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    n_batches  = 0

    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            if is_bernstein:
                frac_zero, coeffs = model(X_batch)
                loss = loss_fn(frac_zero, coeffs, y_batch)
            else:
                fz, mw, s1, sc1, s2, sc2 = model(X_batch)
                loss = loss_fn(fz, mw, s1, sc1, s2, sc2, y_batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / n_batches


# =========================================================================
# Main
# =========================================================================

def main():
    argv = sys.argv[1:]
    use_texture     = '--texture' in argv
    use_gru         = '--gru' in argv
    use_fzhead      = '--fzhead' in argv
    use_bernstein   = '--bernstein' in argv
    use_sqrt_precip = '--sqrt-precip' in argv
    seed = RANDOM_SEED
    seed_explicit = False
    reli_weight = 0.0
    reli_thresh_weights = None
    for a in list(argv):
        if a.startswith('--seed='):
            seed = int(a.split('=', 1)[1])
            seed_explicit = True
        elif a.startswith('--reli-weight='):
            reli_weight = float(a.split('=', 1)[1])
        elif a.startswith('--reli-thresh-weights='):
            reli_thresh_weights = torch.tensor(
                [float(w) for w in a.split('=', 1)[1].split(',')], dtype=torch.float32)
            if len(reli_thresh_weights) != len(RELI_THRESHOLDS):
                print(f'--reli-thresh-weights must have {len(RELI_THRESHOLDS)} comma-separated '
                      f'values, matching RELI_THRESHOLDS={RELI_THRESHOLDS.tolist()} (mm)')
                sys.exit(1)
    argv = [a for a in argv if a not in ('--texture', '--gru', '--fzhead', '--bernstein', '--sqrt-precip')
           and not a.startswith('--seed=') and not a.startswith('--reli-weight=')
           and not a.startswith('--reli-thresh-weights=')]
    precip_transform = 'sqrt' if use_sqrt_precip else 'log1p'

    # Explicit seeding for reproducible multi-seed reruns (2026-08): PyTorch's
    # weight init and DataLoader shuffling otherwise draw from an unseeded
    # global RNG, so repeated runs of the same config aren't directly
    # comparable/reproducible without this.
    torch.manual_seed(seed)
    np.random.seed(seed)

    if use_gru and not use_texture:
        print('--gru requires --texture (GammaMixtureGRU is defined only for the '
              '73-input --texture feature set)')
        sys.exit(1)

    if use_bernstein and not use_gru:
        print('--bernstein requires --gru (BernsteinGRU reuses the same hourly/'
              'static column split and GRU-over-6-hours backbone as GammaMixtureGRU)')
        sys.exit(1)

    if use_bernstein and use_fzhead:
        print('--bernstein and --fzhead are not supported together (BernsteinGRU '
              'does not implement a dedicated fz branch -- frac_zero already gets '
              'its own hurdle-style treatment inseparable from the shared trunk)')
        sys.exit(1)

    if len(argv) not in (1, 2):
        print('Usage: python train_6hourly_mlp.py <clead> [film|crps|hurdle] [--texture] [--gru] [--fzhead] [--bernstein] [--sqrt-precip] [--reli-weight=W] [--reli-thresh-weights=w0.25,w2.5,w10]')
        sys.exit(1)

    clead   = int(argv[0])
    variant = argv[1] if len(argv) == 2 else None
    if variant not in (None, 'film', 'crps', 'hurdle'):
        print(f'Unknown variant {variant!r} (expected "film", "crps", "hurdle", or omit)')
        sys.exit(1)
    if use_gru and variant == 'film':
        print('--gru and "film" are not supported together (both condition the '
              'trunk differently; pick one)')
        sys.exit(1)
    if use_fzhead and variant == 'film':
        print('--fzhead and "film" are not supported together (GammaMixtureMLPFiLM '
              'does not implement a dedicated fz branch)')
        sys.exit(1)
    if use_bernstein and variant is not None:
        print(f'--bernstein is not supported with a loss variant ({variant!r}) -- it '
              f'has its own loss (bernstein_hurdle_loss); omit the variant argument')
        sys.exit(1)

    if use_bernstein and reli_weight:
        print('--reli-weight is not supported with --bernstein (reliability_penalty() '
              'assumes the 6-parameter Gamma-mixture output, not BernsteinGRU\'s '
              '(frac_zero, coeffs) arity)')
        sys.exit(1)

    # 'film' selects the architecture (terrain via FiLM conditioning);
    # 'crps' and 'hurdle' select the loss, with the plain concat
    # architecture. These are mutually exclusive with 'film' for now --
    # nothing stops combining them later, but this experiment only tests
    # loss choice on the architecture already in production. --texture is
    # an orthogonal, independent axis (feature set), so it composes freely
    # with any of the above, e.g. "12 hurdle --texture". --gru is a further
    # orthogonal axis (temporal encoding via a GRU over the 6-hour sequence
    # instead of flat concatenation) that requires --texture and is not
    # combined with 'film'. --fzhead is yet another orthogonal axis (a
    # dedicated small branch for frac_zero, off the shared trunk, instead
    # of sharing the final linear layer with the other 5 params) that
    # composes with any of the above except 'film'. --bernstein replaces
    # the conditional-positive Gamma-mixture parameterization with a
    # Bernstein-quantile function (see BernsteinGRU/bernstein_hurdle_loss);
    # it requires --gru, and brings its own loss, so it's incompatible with
    # the variant argument and with --fzhead.
    use_crps   = (variant == 'crps')
    use_hurdle = (variant == 'hurdle')
    if use_bernstein:
        loss_fn   = bernstein_hurdle_loss
        loss_name = 'BERNSTEIN_HURDLE'
    else:
        loss_fn   = hurdle_loss if use_hurdle else (crps_loss if use_crps else nll_loss)
        loss_name = 'HURDLE' if use_hurdle else ('CRPS' if use_crps else 'NLL')
        if reli_weight:
            loss_fn   = make_reliability_weighted_loss(loss_fn, reli_weight, reli_thresh_weights)
            tw_str = ('[' + ','.join(f'{w:g}' for w in reli_thresh_weights) + ']'
                     if reli_thresh_weights is not None else 'equal')
            loss_name = f'{loss_name}+{reli_weight:g}*RELI(tw={tw_str})'
    n_input      = N_INPUT_TEXTURE if use_texture else N_INPUT
    hidden_sizes = HIDDEN_SIZES_TEXTURE if use_texture else HIDDEN_SIZES
    architecture = ('bernstein_gru' if use_bernstein else
                    'gru' if use_gru else (variant if variant == 'film' else 'concat'))

    print(f'Training 6-hourly MLP for lead time {clead} h'
          f'{"  (FiLM terrain conditioning)" if variant == "film" else ""}'
          f'{"  (CRPS loss)" if use_crps else ""}'
          f'{"  (hurdle loss: NLL fz + CRPS conditional)" if use_hurdle else ""}'
          f'{"  (+texture/persistence/terrain-gradient features)" if use_texture else ""}'
          f'{"  (GRU temporal encoder)" if use_gru else ""}'
          f'{"  (dedicated fz head)" if use_fzhead else ""}'
          f'{"  (Bernstein-quantile conditional-positive head)" if use_bernstein else ""}'
          f'{"  (sqrt precip transform)" if use_sqrt_precip else ""}'
          f'{f"  (+{reli_weight:g}*reliability penalty)" if reli_weight else ""}')
    print(f'Data dir:     {DATA_DIR}')
    print(f'N_INPUT:      {n_input}')
    print(f'Hidden sizes: {hidden_sizes}')
    print(f'Architecture: {architecture}')
    print(f'Dedicated fz head: {use_fzhead}')
    if use_bernstein:
        print(f'Bernstein degree: {BERNSTEIN_DEGREE}')
    ckpt_path_preview = checkpoint_path(clead, variant, use_texture, use_gru, use_fzhead, use_bernstein,
                                        seed if seed_explicit else None, use_sqrt_precip, reli_weight,
                                        reli_thresh_weights)
    print(f'Checkpoint:   {ckpt_path_preview}')
    print()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    features, targets, dates = load_data(clead, use_texture=use_texture, precip_transform=precip_transform)

    # ------------------------------------------------------------------
    # 2. Split into train/val by sequential calendar-day blocks, and
    #    compute normalisation stats from the training split
    # ------------------------------------------------------------------
    train_idx, val_idx = split_by_day_block(dates)
    n, n_train, n_val = len(targets), len(train_idx), len(val_idx)

    feat_mean = features[train_idx].mean(axis=0).astype(np.float32)  # (n_input,)
    feat_std  = features[train_idx].std(axis=0).astype(np.float32)
    feat_std  = np.where(feat_std < 1e-8, 1.0, feat_std)             # avoid /0

    # ------------------------------------------------------------------
    # 3. Build model and optimiser
    # ------------------------------------------------------------------
    if use_bernstein:
        model = BernsteinGRU(n_input=n_input, hidden_sizes=hidden_sizes).to(DEVICE)
    elif use_gru:
        model = GammaMixtureGRU(n_input=n_input, hidden_sizes=hidden_sizes,
                                dedicated_fz_head=use_fzhead).to(DEVICE)
    elif variant == 'film':
        model = GammaMixtureMLPFiLM(n_input=n_input, hidden_sizes=hidden_sizes).to(DEVICE)
    else:
        model = GammaMixtureMLP(n_input=n_input, hidden_sizes=hidden_sizes,
                                dedicated_fz_head=use_fzhead).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_PATIENCE)

    start_epoch    = 0
    best_val_loss  = float('inf')
    no_improve     = 0

    # ------------------------------------------------------------------
    # 4. Resume from checkpoint if available
    # ------------------------------------------------------------------
    ckpt_path = checkpoint_path(clead, variant, use_texture, use_gru, use_fzhead, use_bernstein,
                                seed if seed_explicit else None, use_sqrt_precip, reli_weight,
                                reli_thresh_weights)
    if os.path.exists(ckpt_path):
        print(f'Resuming from checkpoint: {ckpt_path}')
        start_epoch, best_val_loss, feat_mean, feat_std = \
            load_checkpoint(ckpt_path, model, optimizer, scheduler)
        start_epoch += 1
        print(f'  Resuming at epoch {start_epoch}, best val {loss_name}={best_val_loss:.6f}')
    else:
        # Initialise output layer with climatology
        if use_bernstein:
            init_bernstein_output_layer(model, targets[train_idx])
        else:
            init_output_layer(model, targets[train_idx])

    # ------------------------------------------------------------------
    # 5. Build DataLoaders using final normalisation stats
    # ------------------------------------------------------------------
    train_loader, val_loader = make_loaders(features, targets, feat_mean, feat_std,
                                             train_idx, val_idx)

    print(f'\nTraining: {n_train:,} samples, Validation: {n_val:,} samples')
    print(f'Batch size: {BATCH_SIZE}, Max epochs: {MAX_EPOCHS}')
    print()

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, MAX_EPOCHS):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, is_bernstein=use_bernstein)
        train_rel  = float(np.mean(_reli_component_log)) if reli_weight else None
        _reli_component_log.clear()
        val_loss   = run_epoch(val_loader,   model, loss_fn, optimizer=None, is_bernstein=use_bernstein)
        val_rel    = float(np.mean(_reli_component_log)) if reli_weight else None
        _reli_component_log.clear()

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]['lr']

        improved = val_loss < best_val_loss
        tag = ' *' if improved else ''
        rel_str = f'  train_RELI={train_rel:.6f}  val_RELI={val_rel:.6f}' if reli_weight else ''
        print(f'Epoch {epoch+1:3d}/{MAX_EPOCHS}  '
              f'train {loss_name}={train_loss:.6f}  '
              f'val {loss_name}={val_loss:.6f}  '
              f'lr={lr_now:.2e}{rel_str}{tag}')

        if improved:
            best_val_loss = val_loss
            no_improve    = 0
            save_checkpoint(ckpt_path, model, optimizer, scheduler,
                            epoch, best_val_loss,
                            feat_mean, feat_std, clead,
                            architecture=architecture,
                            loss_type=loss_name, n_input=n_input, hidden_sizes=hidden_sizes,
                            dedicated_fz_head=use_fzhead,
                            bernstein_degree=(BERNSTEIN_DEGREE if use_bernstein else None),
                            precip_transform=precip_transform,
                            reli_weight=reli_weight, reli_thresh_weights=reli_thresh_weights)
            print(f'  Checkpoint saved.')
        else:
            no_improve += 1
            if no_improve >= ES_PATIENCE:
                print(f'Early stopping: no improvement for {ES_PATIENCE} epochs.')
                break

    print(f'\nDone. Best val {loss_name} = {best_val_loss:.6f}')
    print(f'Checkpoint: {ckpt_path}')


if __name__ == '__main__':
    main()
