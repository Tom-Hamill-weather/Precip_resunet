# Adapting the season Gamma-mixture ResUNet to the European GRAF domain

Companion to `SEASON_GAMMA_MIXTURE_DEPLOYMENT_GUIDE.md` (the CONUS
deployment guide) -- read that one first for the checkpoint format,
FiLM conditioning, and general architecture background. This guide
covers only what's *different* for Europe.

**Checkpoint generation: this guide tracks whatever the CONUS guide's
top-of-file currency note says** (no separate Europe checkpoints
exist -- \S1). The `v3`/`gamma_mixture_season_v3_precip_climo` string
below is repeated for convenience; if it's gone stale relative to the
CONUS guide, trust that one and update both the S3 paths here (\S1,
\S3) to match.

## 1. What this is

A transfer-learning port: the **same** CONUS-trained
`resunet_gamma_mixture_season_{DJF,MAM,JJA,SON}_best.pth` checkpoints
(from `s3://twc-nvidia/resunet-precip/gamma_mixture_season_v3_precip_climo/`)
run over the European GRAF grid instead of CONUS. No European training
data is involved, no separate Europe checkpoints exist. The model is
fully convolutional and takes no absolute lat/lon input, so nothing
about the network itself needs to change for a different domain --
only the input-feature *sourcing* (reading Europe's own GRAF/GFS/terrain
files instead of CONUS's) and one climatology file are different.

**Verification is intentionally out of scope here** -- MRMS doesn't
cover Europe, so there's no BSS/reliability scoring path for this
domain, same as before this port existed. This is inference/qualitative
use only.

## 2. Code (now in the `Precip_resunet` repo, `Precip_resunet_AWS` branch)

| File | Role |
|---|---|
| `resunet_inference_gamma_mixture_season_europe.py` | Entry point: `python resunet_inference_gamma_mixture_season_europe.py <cyyyymmddhh> <clead>`. Europe counterpart of the CONUS season inference script. |
| `graf_precip_climo_europe.py` | Loader for the Europe precip-climatology channel (see \S3). |
| `build_precip_climo_europe.py` | One-time builder for that climatology file; you should not need to re-run this unless you want to improve the ocean-fill approximation (see \S4). |
| `resunet_inference_gamma_mixture_optimized_europe.py` | Already-existing (pre-dates this port) shared I/O helpers: Europe GRAF/GFS readers, Europe terrain reader, raw-probability calc, netCDF writer. Unchanged by this work. |
| `make_plots_gamma_mixture2_season_europe.py`, `make_plots_gamma_mixture2_3panel_season_europe.py` | Plotting, pointed at the new `_season` probs suffix. |
| `resunet_infer_plot_europe_season.sh` | Driver for a lead sweep (infer + plot). |

Everything else this script needs (`resunet_film.py`, `graf_season_index.py`,
`pytorch_train_resunet_biascorrect_season.py`) is shared, unchanged code
already covered by the CONUS deployment guide.

## 3. Data dependencies (S3: `.../gamma_mixture_season_v3_precip_climo/europe/`)

Two files, neither in git (data files are gitignored), both uploaded:

- **`GRAF_Europe_terrain_info.nc`** (13.8 MiB) -- terrain height/gradient
  fields on the European GRAF grid (723 x 666, ~4 km LCC). Expected at
  `/data/resnet_data/terrain/GRAF_Europe_terrain_info.nc` (AWS) --
  matches the path convention `resunet_inference_gamma_mixture_season_europe.py`
  builds from `AWS_BASE_PATH`.
- **`precip_climo_europe.nc`** (2.3 MiB) -- the `precip_climo` input
  channel, (12, 723, 666) mm/month, on the same Europe grid. Expected
  at `/data/resnet_data/static/precip_climo_europe.nc` (hard-coded in
  `graf_precip_climo_europe.py`'s `PRECIP_CLIMO_NC`). **This file
  wasn't explicitly asked for this round, but it's a hard runtime
  dependency** -- `fulldomain_climo()` will raise if it's missing, the
  same way the CONUS `precip_climo_graf.nc` is required there. Included
  since it's small and otherwise inference simply can't run; say so if
  you'd rather it not have been added.

Both are read from disk paths, not S3, at run time -- after downloading
from S3 once, place them at the exact paths above (or edit the
hard-coded path constants if your target system's layout differs).

## 4. Climatology quality caveat (not fixed, just documented)

Unlike the CONUS blend (PRISM core + WorldClim for non-CONUS land +
ERA5 for open water), Europe's climatology is **WorldClim-only**:
neither PRISM (US-only) nor the ERA5 extract cached on this box
(20-60N / -140 to -50E, doesn't reach Europe's -20 to 29E) cover this
domain. About half the domain (North Sea, Atlantic margin,
Mediterranean/Baltic interior -- open water with no WorldClim land
value) is filled by nearest-valid-neighbor from the nearest coastline,
which is a cruder approximation than CONUS gets for its own offshore
areas. This is the most likely source of any Europe-specific forecast
degradation relative to CONUS. Not addressed in this round; the fix,
if wanted later, is finding an ERA5 (or equivalent) extract that
actually spans Europe's bounding box and re-running
`build_precip_climo_europe.py` with a real offshore term.

## 5. GRAF/GFS input data -- assumed available on the target system

Per your note, don't assume the archive staleness seen on this box
(local GRAF-Europe mirror here stops mid-Feb 2026) applies elsewhere --
this guide assumes the target AWS instance has its own current
European GRAF forecast archive at whatever path its own config points
`GRAFdatadir_europe` at (`config_aws.ini`'s `[DIRECTORIES]` section
falls back to `/data/resnet_data/GRAF/hdo-graf_europe/` if that key
isn't set explicitly -- set it if the target layout differs).

GFS relative humidity for Europe is fetched **live** over HTTPS from
the public `s3://noaa-gfs-bdp-pds` bucket inside
`read_gfs_data_europe()` -- no local archive, no AWS credentials
needed, nothing to pre-stage for this one.

## 6. Output

`<probs_dir>/<cyyyymmddhh>_<clead>_probs_europe_gamma_mixture_season.nc`
-- same variable-name schema as the CONUS output (see the CONUS
deployment guide \S8), distinct filename suffix from the older,
pre-season Europe script's `_probs_europe_gamma_mixture.nc` output, so
the two don't overwrite each other if both are ever run against the
same IC/lead.

## 7. Smoke-testing

1. Confirm both S3 files landed at the expected local paths (\S3).
2. `python resunet_inference_gamma_mixture_season_europe.py <a recent cyyyymmddhh> <a lead 3..72>` and confirm a netCDF is written without error.
3. Run the corresponding plot script and visually sanity-check: probability fields should look terrain-aware (e.g. sharper features over the Alps/Pyrenees/Scandinavian mountains) rather than uniformly smooth -- this is what past test runs (IC2025120812, IC2025120900) looked like.
