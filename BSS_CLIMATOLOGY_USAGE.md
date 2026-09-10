# Using the Stage IV BSS-reference climatology on another computer

There is **one canonical climatology**, built once on the GRAF grid from Stage IV
2020-2024, that both the GRAF (this repo) and HRRR (`/home/thamill/HRRRcal`)
applications use as their Brier-Skill-Score reference forecast. HRRR's copy is a
direct resample of the GRAF-grid file onto the HRRR 3-km grid — not an
independent re-fit — so the two are guaranteed consistent. This doc covers how to
fetch and use both on a machine that doesn't have `/data/resnet_data` or
`/data/hrrr_cal` locally.

## 1. Fetching the files

Both files live in S3 (`s3://twc-nvidia`), uploaded as `tar` bundles (not
gzipped — the `.nc` files are already zlib-compressed internally).

```bash
# GRAF-grid canonical reference (needed for GRAF verification)
aws s3 cp s3://twc-nvidia/resnet/bss_climatology/resnet_bss_climatology.tar .
tar xf resnet_bss_climatology.tar        # -> stage4_climo_reference.nc  (~1.9 GB)

# HRRR 3-km grid, resampled from the GRAF-grid file (needed for HRRR verification)
aws s3 cp s3://twc-nvidia/hrrrcal/bss_climatology/hrrrcal_bss_climatology.tar .
tar xf hrrrcal_bss_climatology.tar       # -> stage4_climo_on_hrrr3km.nc (~4.3 GB)
```

You only need the file matching the model grid you're verifying — GRAF work
needs `stage4_climo_reference.nc`, HRRR work needs `stage4_climo_on_hrrr3km.nc`.
Byte counts to sanity-check a complete download: GRAF-grid file is
1,909,862,400 bytes; HRRR-grid file is 4,294,123,520 bytes (uncompressed inside
the tar; the tar itself is the same size since there's no gzip layer).

**Do not regenerate either file independently on the new machine.** Both are
already the shared, canonical artifact — a fresh independent interpolation
would silently reintroduce the two-different-climatologies problem this setup
was built to eliminate. If you suspect either file is stale, regenerate on
*this* machine (see `build_stage4_climatology_reference.py` in this repo, or
`stage4_climo_to_hrrr3km.py` in HRRRcal) and re-upload — don't re-derive on the
consuming machine.

## 2. File structure (identical layout, different grid)

Both files share one variable layout:

| Variable | Dims | Meaning |
|---|---|---|
| `climo_prob` | `(threshold, month, hour, y, x)` | `P(Stage IV 1-h QPE >= threshold(t) \| month=m, UTC-hour=h)`, float32, `NaN` outside Stage IV's native coverage |
| `threshold` | `(threshold,)` | mm: `[0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0]` |
| `month` | `(month,)` | `1..12` (Jan..Dec) |
| `hour` | `(hour,)` | `0..23` (UTC hour of the 1-h accumulation's *valid* time) |
| `lat`, `lon` | `(y, x)` | degrees_north / degrees_east, same grid as `climo_prob`'s last two dims |

- GRAF-grid file (`stage4_climo_reference.nc`): `y,x = 1308, 1524`, lat range ~7-63°N (GRAF's full Lambert-conformal domain, which extends into Canada/Mexico/ocean well beyond Stage IV's coverage).
- HRRR-grid file (`stage4_climo_on_hrrr3km.nc`): `y,x = 1059, 1799`, lat range ~21-53°N.

**`NaN` means "no Stage IV data here"** — most of Canada, Mexico, and ocean
points on the GRAF grid, and a small fraction (~9%) of the HRRR grid too. Any
BSS computation must mask on `np.isfinite(climo_2d)` in addition to your normal
quality/RQI mask, or you'll be comparing against garbage at those points.

## 3. Looking up the right 2-D slice

For a given forecast valid at `(month, utc_hour)`, scored against threshold
`thr` (mm):

```python
import numpy as np
from netCDF4 import Dataset

nc = Dataset('stage4_climo_reference.nc')   # or stage4_climo_on_hrrr3km.nc
climo_prob_arr = nc.variables['climo_prob']          # keep as a Variable, don't
                                                      # nc.variables['climo_prob'][:] the
                                                      # whole thing (GRAF-grid file is
                                                      # ~16 GB fully materialized)
climo_thresholds = nc.variables['threshold'][:]

def threshold_index(thr_mm, tol=0.01):
    idx = int(np.argmin(np.abs(climo_thresholds - thr_mm)))
    assert abs(float(climo_thresholds[idx]) - thr_mm) <= tol, f'no climo threshold near {thr_mm}mm'
    return idx

tidx = threshold_index(2.5)          # e.g. 2.5 mm
climo_2d = np.asarray(climo_prob_arr[tidx, valid_month - 1, valid_utc_hour])  # (y, x)
climo_valid = np.isfinite(climo_2d)
```

`valid_month` is 1-12; `valid_utc_hour` is 0-23, both taken from the forecast's
**valid time** (init time + lead), not the init time. All 24 UTC hours are
populated (not just synoptic 00/06/12/18Z) — GRAF's gamma-mixture models run at
1h and 3h lead steps, so every UTC hour needs a real entry; this was a deliberate
design decision, not an oversight.

## 4. Computing Brier Skill Score

Standard pattern, mirroring `reliability_resunet_mixture.py` (GRAF) and
`validate_independent_2026.py` (HRRR):

```python
obs = (truth_precip_mm >= thr_mm).astype(np.float64)          # (y, x) binary event
climo_prob_clipped = np.clip(climo_2d, 0.0, 1.0)

# Combine your existing quality/RQI mask with climo coverage
valid = quality_mask & climo_valid                             # quality_mask: e.g. RQI/quality > 0.5

bs_forecast = np.mean((forecast_prob[valid] - obs[valid]) ** 2)
bs_climo    = np.mean((climo_prob_clipped[valid] - obs[valid]) ** 2)

bss = 1.0 - bs_forecast / bs_climo if bs_climo > 0 else np.nan
```

Accumulate `sum((p - obs)**2)` and a sample count separately per threshold
across all your verification cases before dividing, rather than averaging
per-case BSS values — that's what both `reliability_resunet_mixture.py` and
`validate_independent_2026.py` do, and it's the statistically correct way to
pool Brier scores across unequal sample sizes.

## 5. Common pitfalls

- **Wrong grid file for the model.** The GRAF-grid and HRRR-grid files are *not*
  interchangeable — indexing `stage4_climo_on_hrrr3km.nc` with GRAF-grid pixel
  coordinates (or vice versa) will silently give you climatology from the wrong
  physical location. Match the file to the model's native grid.
- **Forgetting the `climo_valid` mask.** Points outside Stage IV coverage are
  `NaN`, not zero. Skipping the `isfinite` check turns those into `NaN`
  contamination in your Brier-score sums (or, if you `nan_to_num` carelessly,
  silently wrong scores from treating "no data" as "zero climatological risk").
- **Loading `climo_prob` fully into memory.** The GRAF-grid file's `climo_prob`
  is `(7, 12, 24, 1308, 1524)` float32 — about 16 GB uncompressed. Keep it as a
  live `netCDF4.Variable` and slice `[tidx, month-1, hour]` per lookup (as shown
  above) rather than doing `[:]` on the whole array.
- **Threshold mismatch.** Your model's own threshold list may be a subset of
  the climatology's 7 stored thresholds (GRAF/HRRR verification code typically
  scores `[0.25, 1.0, 2.5, 5.0, 10.0]` mm, skipping the climatology's 0.5 and
  25 mm entries) — always look up by nearest match, not by raw index, in case
  the subsets differ between scripts.

## 6. Provenance

- Source: Stage IV 1-h QPE, Iowa Environmental Mesonet archive, 2020-01-01 to
  2024-12-31 (`compute_stage4_climatology.py` in this repo).
- GRAF-grid reference built by `build_stage4_climatology_reference.py` (this
  repo) — Delaunay triangulation + `LinearNDInterpolator` from Stage IV's
  native polar-stereographic grid onto the GRAF 4-km Lambert-conformal grid.
- HRRR-grid file built by `stage4_climo_to_hrrr3km.py` (`/home/thamill/HRRRcal`)
  — resampled from the GRAF-grid reference above (a second, smaller
  interpolation step), not re-fit from raw Stage IV.
- Consolidated to this single-canonical-file setup 2026-09-10, after
  discovering GRAF and HRRR had each been maintaining independent
  interpolations of the same source data (which had let a since-fixed
  axis-transpose bug diverge between the two for a time).
