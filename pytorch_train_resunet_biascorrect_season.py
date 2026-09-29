"""pytorch_train_resunet_biascorrect_season.py

Season-pooled, FiLM lead-pooled training for the bias-correction
(single point-estimate) ResUNet, adapted from the GRAF gamma-mixture
recipe (pytorch_train_resunet_gamma_mixture_season.py):

  1. Train one model per calendar season (DJF/MAM/JJA/SON), pooling patches
     from every available year of that season rather than one per-lead
     recency window. Leakage-safe 7-day-block holdout split
     (graf_season_index.build_seasonal_index).
  2. FiLM-condition the ResUNet on [sin_doy, cos_doy, lead/lead_max]
     (resunet_film_biascorrect.AttnResUNetFiLM) so one season model
     covers all lead times instead of needing one model per lead.

Solar-hour is deliberately NOT part of the FiLM conditioning (unlike the
GRAF gamma-mixture season recipe this was adapted from). It varies
spatially (by longitude), so as a global per-image scalar it would force
full-domain inference to either freeze it at one arbitrary longitude or
fall back to patch tiling. Instead it's promoted to two per-pixel INPUT
channels - sin(local_solar_hour), cos(local_solar_hour), computed from
each pixel's own longitude, same treatment as the terrain-gradient
channels - so the model sees the correct value everywhere and a single
whole-domain forward pass is exact, not approximate. Day-of-year and
lead genuinely don't vary spatially, so they stay as global FiLM scalars.

Backbone architecture and loss (CensoredHuberLoss) are unchanged from
pytorch_train_resunet_biascorrect.py, reused via import.

Data source: the same GRAF season/lead zarr patch pools built by
build_patch_pools_graf.py and already used by the gamma-mixture season
trainer (see graf_season_index.py for the pool contract) - no new data
engineering needed, since the raw fields (GRAF, MRMS, MRMS_qual,
terrain_diff, dt_dlon, dt_dlat, GFS_r) are identical.

Usage:
    python pytorch_train_resunet_biascorrect_season.py --season JJA
    python pytorch_train_resunet_biascorrect_season.py --season DJF \\
        --year 2025 --max-epochs 1 --max-patches 500   # smoke test

Checkpoints: {TRAIN_DIR}/resunet_biascorrect_season_{season}_best.pth
"""

import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from graf_season_index import (
    LEAD_MAX, SEASON_MONTHS, ChunkShuffleSampler, build_seasonal_index, make_cond,
    read_patch_cached,
)
from pytorch_train_resunet_biascorrect import (
    AMP_DTYPE, BASE_DIR, BASE_LEARNING_RATE, BATCH_SIZE, DEVICE,
    EARLY_STOPPING_PATIENCE, GRAD_CLIP_NORM, HUBER_DELTA, NUM_EPOCHS,
    NUM_WORKERS, TRAIN_DIR, USE_AMP, CensoredHuberLoss,
)
from resunet_film_biascorrect import AttnResUNetFiLM

GRAF_SEASON_POOLS_DIR = os.path.join(BASE_DIR, 'graf_season_pools')

# Channel order for the 9-channel input stack (dataset stacking order here
# must match the inference script's rebuild order exactly).
CH_ORDER = ['graf', 'terrain_diff', 'gfs_r', 'terdiff_graf', 'graf_rh',
            'dlon', 'dlat', 'sin_sh', 'cos_sh']

# Fixed normalization bounds (the season pool is far too large to scan for
# exact min/max) - same physical-floor values the per-lead trainer's
# data-derived bounds converge to in practice. No power transform here
# (unlike the gamma-mixture recipe), so these are the raw physical bounds.
# sin_sh/cos_sh are already in [-1, 1] by construction.
NORM_BOUNDS = {
    'graf':         (0.0, 75.0),
    'terrain_diff': (-2500.0, 2500.0),
    'gfs_r':        (0.0, 100.0),
    'terdiff_graf': (-75.0 * 2500.0, 75.0 * 2500.0),
    'graf_rh':      (0.0, 75.0 * 100.0),
    'dlon':         (-0.02, 0.02),
    'dlat':         (-0.02, 0.02),
    'sin_sh':       (-1.0, 1.0),
    'cos_sh':       (-1.0, 1.0),
}


def local_solar_hour_sincos(cycle, lead, lon):
    """sin/cos of local solar hour at longitude `lon` (deg, -180..180),
    valid at init-cycle `cycle` + lead `lead`. `lon` may be a scalar
    (training - patch-center lon, ~4 deg wide patch so effectively
    constant) or a full (ny, nx) array (inference - true per-pixel).
    Matches graf_season_index.make_cond's solar-hour term exactly."""
    utc_valid = (cycle + lead) % 24
    solar_hour = (utc_valid + lon / 15.0) % 24
    sin_sh = np.sin(2 * np.pi * solar_hour / 24.0)
    cos_sh = np.cos(2 * np.pi * solar_hour / 24.0)
    return sin_sh, cos_sh


_POOL_VARS = ['GRAF', 'MRMS', 'MRMS_qual', 'terrain_diff', 'dt_dlon', 'dt_dlat', 'GFS_r']


class GRAFSeasonDataset(Dataset):
    """Lazily reads GRAF season/lead-pooled zarr patches (see
    graf_season_index.py for the pool contract), via a small per-worker
    LRU cache of whole decompressed chunks (graf_season_index.
    read_patch_cached) rather than one zarr read per patch - see that
    function's docstring for why (a single-index zarr read decompresses
    and discards the whole chunk regardless of locality)."""

    def __init__(self, index, train=False):
        self.index = index  # list of (zarr_path, patch_idx, day, cycle, lead, lat, lon)
        self.train = train
        self._zstore = {}          # per-worker zarr group cache, opened lazily
        self._chunk_cache = OrderedDict()  # per-worker LRU cache of decompressed chunks

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _normalize(x, key):
        lo, hi = NORM_BOUNDS[key]
        denom = hi - lo if (hi - lo) > 1e-6 else 1.0
        return ((x - lo) / denom).astype(np.float32)

    def __getitem__(self, i):
        zarr_path, pidx, day, cycle, lead, lat, lon = self.index[i]
        fields = read_patch_cached(self._zstore, self._chunk_cache, zarr_path, pidx, _POOL_VARS)

        graf  = fields['GRAF'].astype(np.float32)
        mrms  = fields['MRMS'].astype(np.float32)
        qual  = fields['MRMS_qual'].astype(np.float32)
        diff  = fields['terrain_diff'].astype(np.float32)
        dlon  = fields['dt_dlon'].astype(np.float32)
        dlat  = fields['dt_dlat'].astype(np.float32)
        gfs_r = fields['GFS_r'].astype(np.float32)

        terdiff_graf = graf * diff
        graf_rh = graf * gfs_r

        # Local-solar-hour channels: patch-center lon broadcast across the
        # patch (~4 deg wide, so this is a good approximation to per-pixel
        # even here; inference uses the true per-pixel lon instead).
        sin_sh_val, cos_sh_val = local_solar_hour_sincos(cycle, lead, lon)
        sin_sh = np.full_like(graf, sin_sh_val, dtype=np.float32)
        cos_sh = np.full_like(graf, cos_sh_val, dtype=np.float32)

        x = np.stack([
            self._normalize(graf,         'graf'),
            self._normalize(diff,         'terrain_diff'),
            self._normalize(gfs_r,        'gfs_r'),
            self._normalize(terdiff_graf, 'terdiff_graf'),
            self._normalize(graf_rh,      'graf_rh'),
            self._normalize(dlon,         'dlon'),
            self._normalize(dlat,         'dlat'),
            self._normalize(sin_sh,       'sin_sh'),
            self._normalize(cos_sh,       'cos_sh'),
        ], axis=0).astype(np.float32)

        y = mrms.copy()
        y[qual <= 0.01] = -1.0

        # Keep only [sin_doy, cos_doy, lead_norm] for FiLM - solar-hour is
        # now an input channel above, not a conditioning scalar.
        cond = make_cond(day, cycle, lead, lon, lead_max=LEAD_MAX)[[0, 1, 4]]

        if self.train:
            x, y = self._augment(x, y)

        return torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(cond).float()

    @staticmethod
    def _augment(x, y):
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=2).copy(); y = np.flip(y, axis=1).copy()
            x[5] = -x[5]
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=1).copy(); y = np.flip(y, axis=0).copy()
            x[6] = -x[6]
        return x, y


def train_season(season, max_epochs=None, patience=None, init_from=None,
                 patches_dir=None, year=None, max_patches=None):
    patches_dir = patches_dir or GRAF_SEASON_POOLS_DIR
    train_idx = build_seasonal_index(patches_dir, season, holdout=False, year=year)
    val_idx   = build_seasonal_index(patches_dir, season, holdout=True,  year=year)

    if max_patches is not None:
        train_idx = train_idx[:max_patches]
        val_idx = val_idx[:max(1, max_patches // 5)]

    print(f'Season {season}: {len(train_idx)} train patches, {len(val_idx)} holdout patches '
          f'(pools: {patches_dir})')
    if not train_idx:
        raise RuntimeError(f'No training patches found for season {season} in {patches_dir}')

    train_dataset = GRAFSeasonDataset(train_idx, train=True)
    val_dataset   = GRAFSeasonDataset(val_idx,   train=False)

    pin = (DEVICE.type != 'cpu')
    persist = NUM_WORKERS > 0
    train_sampler = ChunkShuffleSampler(train_idx)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=persist)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin,
                            persistent_workers=persist)

    model = AttnResUNetFiLM(in_channels=9, num_outputs=1, cond_dim=3).to(DEVICE)

    # max_patches means this is a smoke test - use a distinct filename so
    # it can never collide with (or accidentally overwrite) the real
    # per-season production checkpoint.
    tag = '_smoketest' if max_patches is not None else ''
    checkpoint_path = f'{TRAIN_DIR}/resunet_biascorrect_season_{season}{tag}_best.pth'
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

    criterion = CensoredHuberLoss(ignore_index=-1, delta=HUBER_DELTA).to(DEVICE)

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
            with torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP and DEVICE.type == 'cuda'):
                output = model(x, cond)
            loss = criterion(output.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, cond in val_loader:
                x, y, cond = x.to(DEVICE), y.to(DEVICE), cond.to(DEVICE)
                with torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP and DEVICE.type == 'cuda'):
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
                'huber_delta': HUBER_DELTA,
                'normalization_bounds': NORM_BOUNDS,
                'channel_order': CH_ORDER,
                'season': season,
                'lead_max': LEAD_MAX,
                'in_channels': 9,
                'cond_dim': 3,
                'architecture': 'season_film_v2_local_hour_channels',
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
    ap = argparse.ArgumentParser(description='Season-pooled, FiLM lead-pooled bias-correction ResUNet training')
    ap.add_argument('--season', required=True, choices=list(SEASON_MONTHS.keys()))
    ap.add_argument('--max-epochs', type=int, default=None)
    ap.add_argument('--patience', type=int, default=None)
    ap.add_argument('--init-from', default=None, dest='init_from',
                    help='warm-start: load backbone weights from this checkpoint '
                         '(fresh optimizer/scheduler); e.g. another season\'s model')
    ap.add_argument('--patches-dir', default=None, dest='patches_dir')
    ap.add_argument('--year', type=int, default=None,
                    help='restrict to one calendar year (debugging/pilot use)')
    ap.add_argument('--max-patches', type=int, default=None, dest='max_patches',
                    help='truncate train/val index to this many patches (smoke testing)')
    args = ap.parse_args()

    print(f'Device: {DEVICE}  Batch size: {BATCH_SIZE}  Season: {args.season}  '
          f'Lead range: 3-{LEAD_MAX}h')
    train_season(args.season, max_epochs=args.max_epochs, patience=args.patience,
                init_from=args.init_from, patches_dir=args.patches_dir, year=args.year,
                max_patches=args.max_patches)


if __name__ == '__main__':
    main()
