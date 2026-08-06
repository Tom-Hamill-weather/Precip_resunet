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

Input features (36 total):
    Six consecutive hourly gamma-mixture parameters (clead-5 … clead),
    grouped by variable:
        fraction_zero  × 6 hours   (cols  0-5)
        mixture_weight × 6 hours   (cols  6-11)
        gamma_shape1   × 6 hours   (cols 12-17)
        gamma_scale1   × 6 hours   (cols 18-23)
        gamma_shape2   × 6 hours   (cols 24-29)
        gamma_scale2   × 6 hours   (cols 30-35)

Output parameters (6 per sample):
    fraction_zero, mixture_weight, shape1, scale1, shape2, scale2

Loss function:
    NLL for zero-inflated two-component Gamma mixture over the
    zero-inflated two-component Gamma mixture CDF.

Checkpoint saved to:
    mlp_trainings/6h_mlp_lead{clead}h.pth

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
N_INPUT = 36 + len(SEASONAL_VARS) + len(TERRAIN_VARS)

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
                 min_separation=MIN_SEPARATION):
        super().__init__()
        self.shape_min = shape_min
        self.scale_min = scale_min
        self.min_separation = min_separation

        layer_sizes = [n_input] + hidden_sizes
        layers = []
        for in_sz, out_sz in zip(layer_sizes, layer_sizes[1:]):
            layers += [nn.Linear(in_sz, out_sz),
                       nn.BatchNorm1d(out_sz),
                       nn.ReLU(),
                       nn.Dropout(DROPOUT_P)]
        layers.append(nn.Linear(hidden_sizes[-1], 6))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
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
# Data loading
# =========================================================================

def load_data(clead):
    """
    Load all prob_MRMS_samples_*_lead{clead}h.nc files from DATA_DIR.

    Returns
    -------
    features : np.ndarray  (N, N_INPUT)   36 hourly params + cos/sin day-of-year
                                           + local terrain roughness
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

    feat_list   = []
    target_list = []
    date_list   = []

    for fpath in files:
        with Dataset(fpath, 'r') as ds:
            for vname in ['sample_date'] + SEASONAL_VARS + TERRAIN_VARS:
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
            feat_list.append(np.concatenate(
                [hourly_feats, seasonal_feats, terrain_feats], axis=1))
            target_list.append(ds['target_precip_6h'][:].data.astype(np.float32))
            date_list.append(ds['sample_date'][:].data.astype(np.int64))

    features = np.concatenate(feat_list,  axis=0)   # (N, N_INPUT)
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


# =========================================================================
# Checkpoint helpers
# =========================================================================

def checkpoint_path(clead, variant=None):
    os.makedirs(TRAIN_DIR, exist_ok=True)
    suffix = f'_{variant}' if variant else ''
    return os.path.join(TRAIN_DIR, f'6h_mlp_lead{clead}h{suffix}.pth')


def save_checkpoint(path, model, optimizer, scheduler, epoch,
                    best_val_nll, feat_mean, feat_std, clead, architecture='concat'):
    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch':          epoch,
        'best_val_nll':  best_val_nll,
        'feature_mean':   feat_mean,
        'feature_std':    feat_std,
        'shape_min':      model.shape_min,
        'scale_min':      model.scale_min,
        'hidden_sizes':   HIDDEN_SIZES,
        'n_input':        N_INPUT,
        'clead':          clead,
        'architecture':   architecture,
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


def run_epoch(loader, model, optimizer=None):
    """One forward pass over loader; returns mean NLL."""
    training = optimizer is not None
    model.train(training)
    total_nll = 0.0
    n_batches  = 0

    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            fz, mw, s1, sc1, s2, sc2 = model(X_batch)
            loss = nll_loss(fz, mw, s1, sc1, s2, sc2, y_batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_nll += loss.item()
            n_batches  += 1

    return total_nll / n_batches


# =========================================================================
# Main
# =========================================================================

def main():
    if len(sys.argv) not in (2, 3):
        print('Usage: python train_6hourly_mlp.py <clead> [film]')
        sys.exit(1)

    clead   = int(sys.argv[1])
    variant = sys.argv[2] if len(sys.argv) == 3 else None
    if variant not in (None, 'film'):
        print(f'Unknown variant {variant!r} (expected "film" or omit)')
        sys.exit(1)

    print(f'Training 6-hourly MLP for lead time {clead} h'
          f'{"  (FiLM terrain conditioning)" if variant == "film" else ""}')
    print(f'Data dir:     {DATA_DIR}')
    print(f'Checkpoint:   {checkpoint_path(clead, variant)}')
    print()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    features, targets, dates = load_data(clead)

    # ------------------------------------------------------------------
    # 2. Split into train/val by sequential calendar-day blocks, and
    #    compute normalisation stats from the training split
    # ------------------------------------------------------------------
    train_idx, val_idx = split_by_day_block(dates)
    n, n_train, n_val = len(targets), len(train_idx), len(val_idx)

    feat_mean = features[train_idx].mean(axis=0).astype(np.float32)  # (36,)
    feat_std  = features[train_idx].std(axis=0).astype(np.float32)
    feat_std  = np.where(feat_std < 1e-8, 1.0, feat_std)             # avoid /0

    # ------------------------------------------------------------------
    # 3. Build model and optimiser
    # ------------------------------------------------------------------
    model     = (GammaMixtureMLPFiLM() if variant == 'film'
                 else GammaMixtureMLP()).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_PATIENCE)

    start_epoch    = 0
    best_val_nll  = float('inf')
    no_improve     = 0

    # ------------------------------------------------------------------
    # 4. Resume from checkpoint if available
    # ------------------------------------------------------------------
    ckpt_path = checkpoint_path(clead, variant)
    if os.path.exists(ckpt_path):
        print(f'Resuming from checkpoint: {ckpt_path}')
        start_epoch, best_val_nll, feat_mean, feat_std = \
            load_checkpoint(ckpt_path, model, optimizer, scheduler)
        start_epoch += 1
        print(f'  Resuming at epoch {start_epoch}, best val NLL={best_val_nll:.6f}')
    else:
        # Initialise output layer with climatology
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
        train_nll = run_epoch(train_loader, model, optimizer)
        val_nll   = run_epoch(val_loader,   model, optimizer=None)

        scheduler.step(val_nll)
        lr_now = optimizer.param_groups[0]['lr']

        improved = val_nll < best_val_nll
        tag = ' *' if improved else ''
        print(f'Epoch {epoch+1:3d}/{MAX_EPOCHS}  '
              f'train NLL={train_nll:.6f}  '
              f'val NLL={val_nll:.6f}  '
              f'lr={lr_now:.2e}{tag}')

        if improved:
            best_val_nll = val_nll
            no_improve    = 0
            save_checkpoint(ckpt_path, model, optimizer, scheduler,
                            epoch, best_val_nll,
                            feat_mean, feat_std, clead,
                            architecture=(variant or 'concat'))
            print(f'  Checkpoint saved.')
        else:
            no_improve += 1
            if no_improve >= ES_PATIENCE:
                print(f'Early stopping: no improvement for {ES_PATIENCE} epochs.')
                break

    print(f'\nDone. Best val NLL = {best_val_nll:.6f}')
    print(f'Checkpoint: {ckpt_path}')


if __name__ == '__main__':
    main()
