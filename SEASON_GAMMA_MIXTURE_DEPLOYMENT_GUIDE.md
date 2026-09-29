# Deploying the season-pooled, FiLM-conditioned Gamma-mixture ResUNet

This guide is for adapting **probviewer** and the **operational inference**
pipeline to the current production precipitation-postprocessing model,
replacing the old per-(month, lead) checkpoint scheme. It assumes the
reader has the `Precip_resunet` git repo checked out and is copying the
4 season checkpoints, this file, and the supporting data files from
`s3://twc-nvidia/resunet-precip/gamma_mixture_season_v3_precip_climo/`:

```
gamma_mixture_season_v3_precip_climo/
├── resunet_gamma_mixture_season_{DJF,MAM,JJA,SON}_best.pth   (checkpoints)
├── SEASON_GAMMA_MIXTURE_DEPLOYMENT_GUIDE.md                  (this file)
└── support_data/
    ├── precip_climo_graf.nc            (inference: precip-climatology input channel)
    ├── terrain_roughness_mask_graf.nc  (verification: top-10%/bottom-90% terrain masks)
    └── stage4_climo_reference.nc       (verification: BSS climatological reference, 1.8 GiB)
```

## 1. What changed, in one paragraph

The old system trained one checkpoint per (calendar month x 3-h lead
step) -- up to 192 checkpoints -- and ran inference by tiling the CONUS
domain into overlapping patches and Manhattan-blending them back
together, because solar-hour dependence was approximated as a single
scalar per patch. The new system trains one checkpoint per **calendar
season** (DJF/MAM/JJA/SON, 4 total), covers the full **+3h to +72h**
lead range continuously via FiLM conditioning, and runs inference as a
**single whole-domain forward pass** (no tiling, no blending), because
local solar hour is now a true per-pixel input channel rather than a
per-patch approximation. The output netCDF schema is **unchanged** --
same variable names, same file layout -- so anything that only reads
the probability netCDFs (this may include probviewer) needs no changes
at all. The burden is entirely on whatever code actually *runs*
inference to produce those netCDFs.

**This is a breaking checkpoint-format change.** Old-format checkpoints
(7 input channels, `cond_dim=5`) cannot be loaded by the new inference
code, and the new checkpoints (10 channels, `cond_dim=3`) cannot be
loaded by the old inference code. Do not mix old inference code with
new checkpoints or vice versa.

## 2. Code you need from the repo

Pull the latest `main` (or whichever branch this was pushed to) of
<https://github.com/Tom-Hamill-weather/Precip_resunet>. The files that
matter for this deployment:

| File | Role |
|---|---|
| `resunet_inference_gamma_mixture_season.py` | Entry point: `python resunet_inference_gamma_mixture_season.py <cyyyymmddhh> <clead>`. Replaces the old per-lead inference script for this model family. |
| `resunet_film.py` | `AttnResUNetFiLM` model class (already in the repo, unchanged). |
| `graf_season_index.py` | `SEASON_MONTHS` (calendar-month -> season lookup) and `make_cond()` (FiLM conditioning-vector construction). |
| `graf_precip_climo.py` | Loader for the static precipitation-climatology input channel (`fulldomain_climo(month)`). Needs the data file described in \S3. |
| `pytorch_train_resunet_biascorrect_season.py` | Provides `local_solar_hour_sincos()`, used to build the two per-pixel solar-hour input channels at inference time. |
| `resunet_inference_gamma_mixture_optimized.py` | Shared helpers reused unchanged: GRAF/GFS reading, raw-probability calc, `write_probabilities_to_netcdf()` (output schema). |
| `GRAF_CONUS_terrain_info.nc` | Terrain data (already tracked in the repo). |

If your operational pipeline vendors/copies individual files instead of
running from a full checkout, copy all of the above together -- they
import from each other.

## 3. Data dependencies NOT in git: climatology and terrain-mask files

None of these are checked into git (data files are gitignored). All
three are included under `support_data/` in the S3 upload (see the
tree above) -- copy them to the target system(s) and either place
each at the exact path its consuming script hard-codes, or edit that
path to point at wherever you put it.

**Required for inference:**

- `precip_climo_graf.nc` -- static (12, ny, nx) mm/month
  PRISM+WorldClim+ERA5 blend on the GRAF grid, built by
  `build_precip_climo_graf.py`. Read by `graf_precip_climo.py`, which
  hard-codes:
  ```
  PRECIP_CLIMO_NC = '/data/resnet_data/static/precip_climo_graf.nc'
  ```

**Required for verification (BSS/reliability scoring), not for inference itself:**

- `stage4_climo_reference.nc` (1.8 GiB) -- the canonical NCEP Stage IV
  2020--2024 climatological reference used by `reliability_resunet_mixture.py`
  to compute Brier Skill Score against climatology. Hard-coded in that
  script as:
  ```
  climo_graf_file = os.path.join(AWS_BASE_PATH, 'stage4_climo_reference.nc')  # AWS
  climo_graf_file = os.path.expanduser('~/python/resnet_data/stage4_climo_reference.nc')  # laptop
  ```
  Several older/superseded files with similar names exist on the
  training system (`stage4_climo_2020_2024.nc`, `stage4_climo_on_graf.nc`,
  `stage4_climo_on_graf_PREBUGFIX_20260505.nc`) -- do **not** copy
  those; `stage4_climo_reference.nc` is the current canonical one and
  the only one actually read by the verification code.
- `terrain_roughness_mask_graf.nc` (10 MB) -- boolean top-10%/bottom-90%
  terrain-roughness masks, also read unconditionally by
  `reliability_resunet_mixture.py` at import time (it will fail to even
  start without this file, regardless of whether you care about the
  terrain-stratified breakdown). Expected in the same directory as that
  script unless you edit its `_mask_nc_path`.

## 4. Checkpoint files

Four checkpoints, one per season, named `resunet_gamma_mixture_season_{SEASON}_best.pth`
for `SEASON` in `DJF`, `MAM`, `JJA`, `SON`. Place them wherever
`TRAIN_DIR` resolves to in your config (`resunet_inference_gamma_mixture_optimized.TRAIN_DIR`,
currently `../resnet_data/trainings` relative to the code directory --
adjust your config/path if the target system's layout differs).

Each checkpoint is a plain `torch.save`'d dict. The inference code
reads its architecture and normalization metadata **from the
checkpoint itself** rather than hard-coding it -- this is deliberate,
so a future retrain with a different channel count or bound doesn't
require an inference-code change. Do not hard-code `in_channels=10` or
similar in any new consuming code; read it from the checkpoint the way
`resunet_inference_gamma_mixture_season.read_pytorch_season()` does:

```python
checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
in_channels = checkpoint.get('in_channels', 10)
cond_dim = checkpoint.get('cond_dim', 3)
bounds = checkpoint['normalization_bounds']       # dict: channel name -> (lo, hi)
ch_order = checkpoint.get('channel_order', [...]) # list of 10 channel names, in stacking order
climatology = checkpoint.get('climatology')        # dict: EM bias-init Gamma-mixture params
power_transform = checkpoint.get('power_transform', 1.0)
lead_max = checkpoint.get('lead_max', 72)
```

Current metadata (for reference/validation -- read from the DJF
checkpoint at the time of writing; confirm this matches what you
actually load, since it's read dynamically):

- `architecture`: `season_film_v3_precip_climo`
- `in_channels`: 10
- `cond_dim`: 3
- `lead_max`: 72
- `channel_order`: `['graf', 'terrain_diff', 'gfs_r', 'terdiff_graf', 'graf_rh', 'dlon', 'dlat', 'sin_sh', 'cos_sh', 'precip_climo']`

## 5. Checkpoint selection logic

One checkpoint covers all leads (+3h to +72h) for its season. Select by
the **initialization date's calendar month**, not the valid/verification
date:

```python
from graf_season_index import SEASON_MONTHS
_MONTH_TO_SEASON = {m: season for season, months in SEASON_MONTHS.items() for m in months}
season = _MONTH_TO_SEASON[int(cyyyymmddhh[4:6])]
ckpt_path = f'.../resunet_gamma_mixture_season_{season}_best.pth'
```

## 6. Input-feature construction (whole-domain, no tiling)

Ten channels, normalized to `[0,1]` per-channel using the `bounds` dict
from the checkpoint, stacked in `ch_order`:

1. `graf` -- GRAF 1-h precip forecast (optionally power-transformed by
   `power_transform` before normalization, matching training)
2. `terrain_diff` -- local terrain height difference
3. `gfs_r` -- GFS relative humidity
4. `terdiff_graf` -- `graf * terrain_diff`
5. `graf_rh` -- `graf * gfs_r`
6. `dlon`, `dlat` -- terrain gradient (zonal, meridional)
7. `sin_sh`, `cos_sh` -- **per-pixel** local solar hour, sine/cosine
   encoded. This is the key architectural change from the old model:
   `local_solar_hour_sincos(cycle, lead, lons)` where `lons` is the
   **full (ny, nx) longitude array**, not a scalar. This is what makes
   single-pass whole-domain inference exact instead of an approximation.
8. `precip_climo` -- `graf_precip_climo.fulldomain_climo(month)`, a
   direct month-indexed slice (no interpolation needed at inference,
   since the climatology and inference grids are identical).

Use `generate_features_fulldomain()` in `resunet_inference_gamma_mixture_season.py`
as the reference implementation rather than re-deriving this -- the
normalization formula, power-transform handling, and channel-order
indirection all matter for numerical match with training.

## 7. FiLM conditioning vector

Three scalars, constant over the whole domain (not per-pixel):

```python
from graf_season_index import make_cond
cond_full = make_cond(day, cycle, lead, 0.0, lead_max=lead_max)  # 5-vector
cond = cond_full[[0, 1, 4]]  # [sin(doy), cos(doy), lead/lead_max]
```

Note `make_cond` returns a 5-vector (it's shared with an older
5-scalar-conditioning model family); only indices 0, 1, 4 are used
here since indices 2-3 (solar-hour sin/cos) moved to the per-pixel
input channels described above. Passing the full 5-vector to this
model will fail with a shape mismatch (`cond_dim=3`).

## 8. Inference call and output

```python
logits = run_gamma_mixture_season_fulldomain(model, Xpredict_tensor, cond, ny, nx)
gamma_probs, fraction_zero, weight, shape1, scale1, shape2, scale2 = \
    calc_gamma_probabilities_fulldomain(logits, climatology['shape_min'], climatology['scale_min'])
write_probabilities_to_netcdf(nc_filename, lats, lons, raw_probs, gamma_probs, ...)
```

Output filename convention: `{cyyyymmddhh}_{clead}_probs_gamma_mixture_season.nc`
-- note the `_season` suffix, distinct from the old model's
`_probs_gamma_mixture.nc`. **If probviewer (or any downstream consumer)
hard-codes the old filename pattern, that's the one place it likely
needs a change** -- the netCDF *contents* (variable names, dimensions,
scale factors) are identical to the old format, only the filename
differs.

## 9. Smoke-testing before switching production over

1. Run `python resunet_inference_gamma_mixture_season.py <a recent cyyyymmddhh> <a lead in 3..72>` on the target system and confirm it produces a netCDF without errors.
2. Compare a few grid-point probability values against a known-good run of the same case from the training system, to catch environment-specific (e.g. package-version, path-config) numerical drift before trusting a full operational cutover.
3. Confirm the `_season` output lands somewhere probviewer (or whatever reads these files) actually looks -- check the configured probs directory matches.
