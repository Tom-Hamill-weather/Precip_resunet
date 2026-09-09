"""pytorch_train_resunet_gamma_mixture_season.py

Season-pooled, FiLM lead-pooled training for the GRAF gamma-mixture
ResUNet, porting the two HRRRcal training-recipe choices Tom asked about
(HRRRcal/pytorch_train_hrrr_gamma_mixture.py):

  1. Train one model per calendar season (DJF/MAM/JJA/SON), pooling patches
     from every available year of that season rather than one ~8-month
     recency window per IC date. Leakage-safe 7-day-block holdout split
     (graf_season_index.build_seasonal_index).
  2. FiLM-condition the ResUNet on [sin_doy, cos_doy, sin_solar_hour,
     cos_solar_hour, lead/72] (resunet_film.AttnResUNetFiLM) so the single
     season model covers all lead times 3-72h instead of needing one model
     per lead.

Everything else (backbone architecture, 2-component Gamma mixture NLL loss,
EM-fit climatology bias init, Adam wd=0, sqrt power-transform, H/V-flip
augmentation) is unchanged from pytorch_train_resunet_gamma_mixture_v2.py --
those are the working GRAF recipe choices, reused via import, not
reimplemented.

Data source: zarr patch pools built by build_patch_pools_graf.py (NOT the
per-(IC-date,lead) pickles used by the existing per-lead training script --
see graf_season_index.py for the pool contract).

Usage:
    python pytorch_train_resunet_gamma_mixture_season.py --season JJA
    python pytorch_train_resunet_gamma_mixture_season.py --season SON \\
        --init-from /data/resnet_data/trainings/resunet_gamma_mixture_season_JJA_best.pth

Checkpoints: {TRAIN_DIR}/resunet_gamma_mixture_season_{season}_best.pth
"""

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import zarr
from torch.utils.data import DataLoader, Dataset

from gamma_mixture_em import fit_gamma_mixture
from graf_season_index import LEAD_MAX, SEASON_MONTHS, build_seasonal_index, make_cond
from pytorch_train_resunet_gamma_mixture_v2 import (
    AMP_DTYPE, BASE_DIR, BASE_LEARNING_RATE, BATCH_SIZE, DEVICE,
    EARLY_STOPPING_PATIENCE, GammaMixtureNLLLoss, NUM_EPOCHS, NUM_WORKERS,
    POWER_TRANSFORM, STABILITY_PARAMS, TRAIN_DIR, USE_AMP,
    initialize_output_layer,
)
from resunet_film import AttnResUNetFiLM

GRAF_SEASON_POOLS_DIR = os.path.join(BASE_DIR, 'graf_season_pools')

# Fixed normalization bounds (vs v2's per-run data-derived min/max): the
# season pool is far too large to scan for exact min/max, so use the same
# physical-floor values v2's data-derived bounds converge to in practice
# (2500 m terrain diff, 100% RH, 0.02 rad/m terrain gradient), scaled by
# POWER_TRANSFORM for the GRAF-precip-derived quantities. Stored in every
# checkpoint (mirrors HRRRcal's config-bounds-in-checkpoint pattern) so
# inference always matches training exactly.
_G75 = 75.0 ** POWER_TRANSFORM
NORM_BOUNDS = {
    'graf':         (0.0, _G75),
    'terrain_diff': (-2500.0, 2500.0),
    'gfs_r':        (0.0, 100.0),
    'terdiff_graf': (-_G75 * 2500.0, _G75 * 2500.0),
    'graf_rh':      (0.0, _G75 * 100.0),
    'dlon':         (-0.02, 0.02),
    'dlat':         (-0.02, 0.02),
}


class GRAFSeasonDataset(Dataset):
    """Lazily reads GRAF season/lead-pooled zarr patches (see
    graf_season_index.py for the pool contract). Mirrors HRRRcal's
    HRRRPatchDataset lazy-open-per-worker pattern
    (pytorch_train_hrrr_gamma_mixture.py:476-544)."""

    def __init__(self, index, train=False, power_transform=POWER_TRANSFORM):
        self.index = index  # list of (zarr_path, patch_idx, day, cycle, lead, lat, lon)
        self.train = train
        self.power_transform = power_transform
        self._zstore = {}  # per-worker-process zarr group cache, opened lazily

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _normalize(x, key):
        lo, hi = NORM_BOUNDS[key]
        denom = hi - lo if (hi - lo) > 1e-6 else 1.0
        return ((x - lo) / denom).astype(np.float32)

    def __getitem__(self, i):
        zarr_path, pidx, day, cycle, lead, lat, lon = self.index[i]
        if zarr_path not in self._zstore:
            self._zstore[zarr_path] = zarr.open_group(zarr_path, mode='r')
        grp = self._zstore[zarr_path]

        graf  = np.array(grp['GRAF'][pidx],         dtype=np.float32)
        mrms  = np.array(grp['MRMS'][pidx],         dtype=np.float32)
        qual  = np.array(grp['MRMS_qual'][pidx],    dtype=np.float32)
        diff  = np.array(grp['terrain_diff'][pidx], dtype=np.float32)
        dlon  = np.array(grp['dt_dlon'][pidx],      dtype=np.float32)
        dlat  = np.array(grp['dt_dlat'][pidx],      dtype=np.float32)
        gfs_r = np.array(grp['GFS_r'][pidx],        dtype=np.float32)

        if self.power_transform != 1.0:
            graf = np.power(np.clip(graf, 0.0, None), self.power_transform)
        terdiff_graf = graf * diff
        graf_rh = graf * gfs_r

        x = np.stack([
            self._normalize(graf,         'graf'),
            self._normalize(diff,         'terrain_diff'),
            self._normalize(gfs_r,        'gfs_r'),
            self._normalize(terdiff_graf, 'terdiff_graf'),
            self._normalize(graf_rh,      'graf_rh'),
            self._normalize(dlon,         'dlon'),
            self._normalize(dlat,         'dlat'),
        ], axis=0).astype(np.float32)

        y = mrms.copy()
        y[qual <= 0.01] = -1.0

        cond = make_cond(day, cycle, lead, lon, lead_max=LEAD_MAX)

        if self.train:
            x, y = self._augment(x, y)

        return torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(cond).float()

    @staticmethod
    def _augment(x, y):
        # Same H/V-flip-with-gradient-sign-flip augmentation as v2 (channels
        # 5=dlon, 6=dlat).
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=2).copy(); y = np.flip(y, axis=1).copy()
            x[5] = -x[5]
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=1).copy(); y = np.flip(y, axis=0).copy()
            x[6] = -x[6]
        return x, y


def compute_gamma_mixture_climatology_season(train_dataset, n_sample=1000):
    """Same EM-fit climatology as v2's compute_gamma_mixture_climatology,
    adapted for GRAFSeasonDataset's 3-tuple (x, y, cond) items."""
    print('Computing 2-component Gamma mixture climatology (season-pooled)...')
    n = len(train_dataset)
    sample_indices = np.random.choice(n, size=min(n_sample, n), replace=False)

    all_values = []
    for idx in sample_indices:
        _, y, _ = train_dataset[idx]
        all_values.append(y[y >= 0].numpy())
    all_values = np.concatenate(all_values)

    fraction_zero = (all_values == 0).sum() / len(all_values)
    wet_values = all_values[all_values > 0]

    if len(wet_values) < 100:
        print('WARNING: too few wet pixels for reliable Gamma mixture fitting')
        return {'fraction_zero': fraction_zero, 'weight': 0.5, 'shape1': 0.8,
                'scale1': 1.0, 'shape2': 2.0, 'scale2': 3.0,
                'shape_min': 0.3, 'scale_min': 0.01}

    max_wet = 50000
    if len(wet_values) > max_wet:
        wet_values = wet_values[np.random.choice(len(wet_values), size=max_wet, replace=False)]

    try:
        weights, shapes, scales, em_model = fit_gamma_mixture(
            wet_values, n_components=2, init_method='moments', verbose=False, max_iter=500)
        sort_idx = np.argsort(shapes)
        weights, shapes, scales = weights[sort_idx], shapes[sort_idx], scales[sort_idx]
        weight1, shape1, scale1 = float(weights[0]), float(shapes[0]), float(scales[0])
        shape2, scale2 = float(shapes[1]), float(scales[1])
        print(f'  EM converged after {em_model.n_iter_} iterations, '
              f'log-likelihood {em_model.loglik_:.2f}')
    except Exception as e:
        print(f'  WARNING: EM algorithm failed ({e}); falling back to percentile init')
        weight1, shape1, scale1 = 0.5, 0.8, float(np.percentile(wet_values, 25))
        shape2, scale2 = 2.0, float(np.percentile(wet_values, 75))

    print(f'  fraction_zero={fraction_zero:.3f}  w={weight1:.3f}  '
          f'shape1/scale1={shape1:.3f}/{scale1:.3f}  shape2/scale2={shape2:.3f}/{scale2:.3f}')

    return {'fraction_zero': fraction_zero, 'weight': weight1,
            'shape1': shape1, 'scale1': scale1, 'shape2': shape2, 'scale2': scale2,
            'shape_min': 0.3, 'scale_min': 0.01}


def train_season(season, max_epochs=None, patience=None, init_from=None,
                 patches_dir=None, year=None):
    patches_dir = patches_dir or GRAF_SEASON_POOLS_DIR
    train_idx = build_seasonal_index(patches_dir, season, holdout=False, year=year)
    val_idx   = build_seasonal_index(patches_dir, season, holdout=True,  year=year)
    print(f'Season {season}: {len(train_idx)} train patches, {len(val_idx)} holdout patches '
          f'(pools: {patches_dir})')
    if not train_idx:
        raise RuntimeError(f'No training patches found for season {season} in {patches_dir}')

    train_dataset = GRAFSeasonDataset(train_idx, train=True)
    val_dataset   = GRAFSeasonDataset(val_idx,   train=False)

    climatology = compute_gamma_mixture_climatology_season(train_dataset)

    pin = (DEVICE.type != 'cpu')
    persist = NUM_WORKERS > 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=persist)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin,
                            persistent_workers=persist)

    model = AttnResUNetFiLM(in_channels=7, num_outputs=6, cond_dim=5).to(DEVICE)
    initialize_output_layer(model, climatology)

    checkpoint_path = f'{TRAIN_DIR}/resunet_gamma_mixture_season_{season}_best.pth'
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    ckpt = None

    if init_from and os.path.exists(init_from):
        print(f'Warm-starting from {init_from} (fresh optimizer/scheduler)')
        ckpt = torch.load(init_from, map_location=DEVICE, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f'  strict=False load: missing={list(missing)}, unexpected={list(unexpected)}')
    elif os.path.exists(checkpoint_path):
        print(f'Found existing season checkpoint: {checkpoint_path}')
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch']
        best_val_loss = ckpt['loss']
        print(f'  Resuming from epoch {start_epoch}, best val loss {best_val_loss:.4f}')

    resuming = (not init_from) and (ckpt is not None) and os.path.exists(checkpoint_path)

    criterion = GammaMixtureNLLLoss(
        shape_min=climatology['shape_min'], scale_min=climatology['scale_min'],
        ignore_index=-1, epsilon=STABILITY_PARAMS['epsilon'],
        shape_max=STABILITY_PARAMS['shape_max'], scale_max=STABILITY_PARAMS['scale_max'],
        nll_max=STABILITY_PARAMS['nll_max'], min_separation=0.5,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=BASE_LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2)
    if resuming:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        except Exception as e:
            print(f'  WARNING: could not resume optimizer/scheduler state: {e}')

    epochs = max_epochs or NUM_EPOCHS
    pat = patience or EARLY_STOPPING_PATIENCE

    print(f'Training batches/epoch: {len(train_loader)}  Val batches/epoch: {len(val_loader)}')

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for x, y, cond in train_loader:
            x, y, cond = x.to(DEVICE), y.to(DEVICE), cond.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                output = model(x, cond)
            loss = criterion(output.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=STABILITY_PARAMS['grad_clip'])
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, cond in val_loader:
                x, y, cond = x.to(DEVICE), y.to(DEVICE), cond.to(DEVICE)
                with torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                    output = model(x, cond)
                loss = criterion(output.float(), y)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  '
              f'val_loss={val_loss:.4f}  lr={current_lr:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
                'climatology': climatology,
                'power_transform': POWER_TRANSFORM,
                'normalization_bounds': NORM_BOUNDS,
                'season': season,
                'lead_max': LEAD_MAX,
                'cond_dim': 5,
                'architecture': 'season_film_v1',
            }, checkpoint_path)
            print(f'  -> saved best model: {checkpoint_path}')
        else:
            epochs_no_improve += 1
            print(f'  (no improvement for {epochs_no_improve} epochs)')

        if epochs_no_improve >= pat:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    print(f'Training complete. Best val loss: {best_val_loss:.4f}')
    return checkpoint_path, best_val_loss


def main():
    ap = argparse.ArgumentParser(description='Season-pooled, FiLM lead-pooled GRAF ResUNet training')
    ap.add_argument('--season', required=True, choices=list(SEASON_MONTHS.keys()))
    ap.add_argument('--max-epochs', type=int, default=None)
    ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--init-from', default=None, dest='init_from',
                    help='warm-start: load backbone weights from this checkpoint '
                         '(fresh optimizer/scheduler); e.g. another season\'s model')
    ap.add_argument('--patches-dir', default=None, dest='patches_dir')
    ap.add_argument('--year', type=int, default=None,
                    help='restrict to one calendar year (debugging/pilot use)')
    args = ap.parse_args()

    print(f'Device: {DEVICE}  Batch size: {BATCH_SIZE}  Season: {args.season}  '
          f'Lead range: 3-{LEAD_MAX}h')
    train_season(args.season, max_epochs=args.max_epochs, patience=args.patience,
                init_from=args.init_from, patches_dir=args.patches_dir, year=args.year)


if __name__ == '__main__':
    main()
