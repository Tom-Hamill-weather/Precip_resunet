"""
diagnostic_shape_comparison.py -- cheap CLT diagnostic: compare the 6h MLP's
fitted shape1/shape2 (the 6-hour AGGREGATE gamma-mixture shape parameters)
against the 1h ResUNet's own fitted shape1/shape2 (the per-hour INPUT values
already sitting in the training-sample features) for the same underlying
samples.

Motivation (see project_6h_mlp_architecture_options.md / conversation
2026-08-09): if 6h aggregation drives the distribution toward more
Gaussian-like bulk shape via CLT, and the Gamma family can already represent
that via larger shape parameters (Gamma(k) -> Normal as k grows, since a
Gamma(k) is itself a sum of k exponentials), the 6h MLP's fitted shapes
should come out systematically larger than the hourly ResUNet's. If they
come out roughly the same, that's a concrete sign the model is NOT
capturing the CLT-driven bulk-shape effect and a family change might
actually help -- rather than guessing, this checks it directly against
data already on disk, no new data generation needed.

Runs on CPU deliberately, to avoid contending with the lead24h/48h GPU eval
running concurrently.

Tom Hamill / Claude, Aug 2026
"""

import numpy as np
import torch
import train_6hourly_mlp as t6

CLEAD = 12
VARIANT_SUFFIX = 'texture_gru_seed999'   # the nominal-winner plain-NLL+GRU checkpoint

print(f'Loading lead{CLEAD}h training samples...')
features, targets, dates = t6.load_data(CLEAD, use_texture=True)
n = features.shape[0]
print(f'{n} samples')

# FEATURE_VARS = ['fraction_zero','mixture_weight','gamma_shape1','gamma_scale1',
#                 'gamma_shape2','gamma_scale2'], each (N,6), concatenated in that
# order along axis=1 -- see load_data()'s hourly_feats construction.
shape1_1h = features[:, 12:18].ravel()   # 1h ResUNet's own fitted shape1, all 6 hours, RAW (pre-normalization)
shape2_1h = features[:, 24:30].ravel()   # 1h ResUNet's own fitted shape2, all 6 hours

ckpt_path = f'mlp_trainings/6h_mlp_lead{CLEAD}h_{VARIANT_SUFFIX}.pth'
print(f'Loading checkpoint: {ckpt_path}')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
feat_mean = ckpt['feature_mean']
feat_std  = ckpt['feature_std']
hidden_sizes = ckpt.get('hidden_sizes', t6.HIDDEN_SIZES_TEXTURE)
n_input      = ckpt.get('n_input', t6.N_INPUT_TEXTURE)
architecture = ckpt.get('architecture', 'concat')
dedicated_fz_head = ckpt.get('dedicated_fz_head', False)
assert architecture == 'gru', f'expected gru architecture, got {architecture!r}'

model = t6.GammaMixtureGRU(hidden_sizes=hidden_sizes, n_input=n_input,
                           dedicated_fz_head=dedicated_fz_head)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

feats_norm = (features - feat_mean) / feat_std

BATCH = 50000
shape1_6h_list = []
shape2_6h_list = []
with torch.no_grad():
    for i in range(0, n, BATCH):
        X = torch.tensor(feats_norm[i:i+BATCH], dtype=torch.float32)
        fz, mw, s1, sc1, s2, sc2 = model(X)
        shape1_6h_list.append(s1.numpy())
        shape2_6h_list.append(s2.numpy())
        if (i // BATCH) % 5 == 0:
            print(f'  {min(i+BATCH, n)}/{n}')

shape1_6h = np.concatenate(shape1_6h_list)
shape2_6h = np.concatenate(shape2_6h_list)

def summarize(name, arr):
    print(f'{name:24s}  mean={np.mean(arr):7.3f}  median={np.median(arr):7.3f}  '
         f'p10={np.percentile(arr,10):7.3f}  p90={np.percentile(arr,90):7.3f}')

print()
print('=== Shape parameter comparison: 1h ResUNet (input) vs 6h MLP aggregate (output) ===')
summarize('1h shape1 (all 6 hrs)', shape1_1h)
summarize('6h MLP shape1',         shape1_6h)
print()
summarize('1h shape2 (all 6 hrs)', shape2_1h)
summarize('6h MLP shape2',         shape2_6h)

print()
ratio1 = np.median(shape1_6h) / np.median(shape1_1h)
ratio2 = np.median(shape2_6h) / np.median(shape2_1h)
print(f'Median ratio (6h/1h): shape1={ratio1:.2f}x   shape2={ratio2:.2f}x')
print('(CLT-consistent aggregation of ~6 comparable draws would suggest something')
print(' in the neighborhood of Nx if hours contributed roughly equally and independently;')
print(' zero-inflation/dependence/regime-mixing all pull this below that naive ceiling,')
print(' so treat "clearly > 1x" vs "roughly 1x" as the meaningful distinction, not the')
print(' exact ratio value.)')
