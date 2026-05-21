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
LEARNING_RATE = 1e-3
MAX_EPOCHS    = 75
LR_PATIENCE   = 5          # ReduceLROnPlateau patience
ES_PATIENCE   = 5          # early-stopping patience
VAL_FRAC      = 0.20
RANDOM_SEED   = 42

SHAPE_MIN = 0.1
SCALE_MIN = 0.01

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

# =========================================================================
# Model
# =========================================================================

class GammaMixtureMLP(nn.Module):
    """
    MLP that maps 36 hourly gamma-mixture features to 6 parameters for
    the 6-hourly zero-inflated two-component Gamma mixture distribution.

    Architecture: 36 → 72 → 144 → 72 → 36 → 12 → 6
    Activations:  ReLU after each BatchNorm layer.

    Label-switching fix: shape2 is constrained to exceed shape1 by at
    least 0.5, ensuring component 2 always represents the heavier tail.
    """

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
        raw = self.net(x)                                      # (batch, 6)
        frac_zero  = torch.sigmoid(raw[:, 0])
        mix_weight = torch.sigmoid(raw[:, 1])
        shape1     = self.shape_min + F.softplus(raw[:, 2])
        scale1     = self.scale_min + F.softplus(raw[:, 3])
        shape2     = self.shape_min + F.softplus(raw[:, 4])
        scale2     = self.scale_min + F.softplus(raw[:, 5])

        # Reorder so component 1 always has the smaller mean (drier).
        # swap is treated as a constant by autograd — gradients flow through
        # the parameter values, not through the ordering decision.
        swap = (shape1 * scale1 > shape2 * scale2).float()    # 1 where order is wrong
        shape1_out     = (1 - swap) * shape1  + swap * shape2
        scale1_out     = (1 - swap) * scale1  + swap * scale2
        shape2_out     = (1 - swap) * shape2  + swap * shape1
        scale2_out     = (1 - swap) * scale2  + swap * scale1
        mix_weight_out = (1 - swap) * mix_weight + swap * (1 - mix_weight)

        return frac_zero, mix_weight_out, shape1_out, scale1_out, shape2_out, scale2_out


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
    features : np.ndarray  (N, 36)
    targets  : np.ndarray  (N,)
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

    for fpath in files:
        with Dataset(fpath, 'r') as ds:
            blocks = []
            for vname in FEATURE_VARS:
                arr = ds[vname][:].data.astype(np.float32)   # (nsamples, 6)
                blocks.append(arr)
            feat_list.append(np.concatenate(blocks, axis=1))  # (nsamples, 36)
            target_list.append(ds['target_precip_6h'][:].data.astype(np.float32))

    features = np.concatenate(feat_list,  axis=0)   # (N, 36)
    targets  = np.concatenate(target_list, axis=0)  # (N,)
    print(f'Total samples: {len(targets):,}')
    print(f'  Wet fraction : {(targets > 0).mean():.3f}')
    print(f'  Mean precip  : {targets.mean():.3f} mm')
    print(f'  Max precip   : {targets.max():.3f} mm')
    return features, targets


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


def init_output_layer(model, targets):
    """
    Initialise the bias of the final Linear layer using climatology of targets.
    """
    wet = targets[targets > 0]
    frac_zero_clim = float((targets == 0).mean())
    frac_zero_clim = np.clip(frac_zero_clim, 1e-4, 1 - 1e-4)

    # Method of Moments Gamma fit to wet pixels
    mu     = float(wet.mean())
    sigma2 = float(wet.var())
    if sigma2 < 1e-8:
        sigma2 = 1e-8
    shape_clim = mu ** 2 / sigma2
    scale_clim = sigma2 / mu

    # logit(p) for frac_zero bias
    logit_fz = math.log(frac_zero_clim / (1.0 - frac_zero_clim))

    # Inverse-softplus for shape/scale biases
    b2  = _inv_softplus(shape_clim, SHAPE_MIN)
    b3  = _inv_softplus(scale_clim, SCALE_MIN)

    with torch.no_grad():
        final_layer = model.net[-1]               # last Linear(12→6)
        final_layer.bias[0].fill_(logit_fz)       # frac_zero
        final_layer.bias[1].fill_(0.0)            # mix_weight → 0.5
        final_layer.bias[2].fill_(b2)             # shape1
        final_layer.bias[3].fill_(b3)             # scale1
        final_layer.bias[4].fill_(b2)             # shape2
        final_layer.bias[5].fill_(b3)             # scale2

    print(f'Climatological init: frac_zero={frac_zero_clim:.3f}, '
          f'shape={shape_clim:.3f}, scale={scale_clim:.3f}')


# =========================================================================
# Checkpoint helpers
# =========================================================================

def checkpoint_path(clead):
    os.makedirs(TRAIN_DIR, exist_ok=True)
    return os.path.join(TRAIN_DIR, f'6h_mlp_lead{clead}h.pth')


def save_checkpoint(path, model, optimizer, scheduler, epoch,
                    best_val_nll, feat_mean, feat_std, clead):
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
        'clead':          clead,
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

def make_loaders(features, targets, feat_mean, feat_std):
    """Normalise and split into train/val DataLoaders."""
    feats_norm = (features - feat_mean) / feat_std

    X = torch.tensor(feats_norm, dtype=torch.float32)
    y = torch.tensor(targets,    dtype=torch.float32)

    n      = len(y)
    n_val  = int(n * VAL_FRAC)
    n_train = n - n_val

    rng = torch.Generator().manual_seed(RANDOM_SEED)
    perm = torch.randperm(n, generator=rng)

    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

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
    if len(sys.argv) != 2:
        print('Usage: python train_6hourly_mlp.py <clead>')
        sys.exit(1)

    clead = int(sys.argv[1])
    print(f'Training 6-hourly MLP for lead time {clead} h')
    print(f'Data dir:     {DATA_DIR}')
    print(f'Checkpoint:   {checkpoint_path(clead)}')
    print()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    features, targets = load_data(clead)

    # ------------------------------------------------------------------
    # 2. Compute normalisation stats from training split
    # ------------------------------------------------------------------
    n       = len(targets)
    n_val   = int(n * VAL_FRAC)
    n_train = n - n_val

    rng  = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(n)
    train_idx = perm[:n_train]

    feat_mean = features[train_idx].mean(axis=0).astype(np.float32)  # (36,)
    feat_std  = features[train_idx].std(axis=0).astype(np.float32)
    feat_std  = np.where(feat_std < 1e-8, 1.0, feat_std)             # avoid /0

    # ------------------------------------------------------------------
    # 3. Build model and optimiser
    # ------------------------------------------------------------------
    model     = GammaMixtureMLP().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_PATIENCE)

    start_epoch    = 0
    best_val_nll  = float('inf')
    no_improve     = 0

    # ------------------------------------------------------------------
    # 4. Resume from checkpoint if available
    # ------------------------------------------------------------------
    ckpt_path = checkpoint_path(clead)
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
    train_loader, val_loader = make_loaders(features, targets, feat_mean, feat_std)

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
                            feat_mean, feat_std, clead)
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
