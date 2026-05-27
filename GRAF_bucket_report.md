# GRAF S3 Bucket Survey Report

**Bucket:** `s3://twc-cf-model-graf-us-east-1/`

**Date surveyed:** 2026-03-13

---

## Top-Level Folder Structure

The bucket contains **11 top-level folders**, each representing a distinct product suite:

| Folder | Domain | Resolution | Time Step | Format |
|--------|--------|-----------|-----------|--------|
| GRAF-CPG | Global | 13 km | Hourly | GRIB2 |
| GRAF-ENERGY | Global/CONUS | 4 km | 30-min | NetCDF |
| GRAF-HOURLY | Global | 4 km | Hourly | GRIB2 |
| GRAF-MEDIA5M | CONUS | 4 km | 5-min | GRIB2 |
| GRAF-MEDIA15M | CONUS | 4 km | 15-min | GRIB2 |
| GRAF-MEDIA30M | CONUS | 4 km | 30-min | GRIB2 |
| GRAF-MEDIA60M | CONUS | 4 km | Hourly | GRIB2 |
| GRAF-SSDS60M | Global | 4 km | Hourly | NetCDF |
| GRAF-STFOD | Global | 4 km | 5-min | GRIB2 |
| GRAF-WXMIX | CONUS | 4 km | Hourly | GRIB2 |
| VERIF | — | — | — | .command |

---

## Directory Structure Pattern

All product folders share the same sub-tree layout:

```
s3://twc-cf-model-graf-us-east-1/
└── <PRODUCT>/
    └── deterministic/
        └── <domain>/       (global | conus)
            └── <res>/      (4km | 13km)
                └── <YYYY>/
                    └── <MM>/
                        └── <DD>/
                            └── <HH>/
                                └── <files>.grb2 | .nc
```

Example path:
```
GRAF-CPG/deterministic/global/13km/2026/03/07/22/
  GRAF-CPG_deterministic_global_13km_20260307T220000Z_20260310T110000Z_rh_p850mbar.grb2
```

File naming: `<PRODUCT>_deterministic_<domain>_<res>_<ref_time>_<valid_time>_<parameter>_<vert_level>.<ext>`

---

## Forecast Lead Times & Time Steps

| Folder | Time Step | Max Lead Time (observed) |
|--------|-----------|--------------------------|
| GRAF-CPG | 1 hour | ~60+ hours |
| GRAF-HOURLY | 1 hour | ~60+ hours |
| GRAF-MEDIA5M | 5 minutes | ~62 hours |
| GRAF-MEDIA15M | 15 minutes | ~62 hours |
| GRAF-MEDIA30M | 30 minutes | ~62 hours |
| GRAF-MEDIA60M | 1 hour | ~62 hours |
| GRAF-ENERGY | 30 minutes | ~60+ hours |
| GRAF-SSDS60M | 1 hour | ~60+ hours |
| GRAF-STFOD | 5 minutes | ~62 hours |
| GRAF-WXMIX | 1 hour | ~60+ hours |

---

## Forecast Parameters by Folder

### GRAF-CPG (Global 13 km — general-purpose climate/weather)
bli, ceiling, cmltice, dpt, flike, gustspd, hifrel, hterrain, landcov, pmsl, prate, prec, pres, ptype, rh, snod, t, tcdc, u, ugrd, v, vgrd, vis, vsmoist

### GRAF-HOURLY (Global 4 km — hourly high-res)
cape, ceiling, cmltice, dist, prate, prec, snod, snowratio, t, u, ugrd, v, vgrd, vis

### GRAF-MEDIA5M (CONUS 4 km — 5-minute media products)
cmltice, dbz, pbits, prate, prec, satir, snod, tcdc, winstorm

### GRAF-MEDIA15M (CONUS 4 km — 15-minute media products)
dpt, gustspd, olr, pmsl, prate, prec, pwat, t, ugrd, vgrd, vis, wind

### GRAF-MEDIA30M (CONUS 4 km — 30-minute media products)
cape, firewx, flike, gustspd, mixr, pwrout, rh, u, v, vis

### GRAF-MEDIA60M (CONUS 4 km — hourly media products)
ehlx, mxuphl, vwsh

### GRAF-ENERGY (Global/CONUS — energy sector)
Multi-variable composite files (NetCDF)

### GRAF-STFOD (Global 4 km — short-term, 5-min)
cpoice, cporain, cposnow, dist, hifrel, hterrain, mixr, pot, poth, prate, prec@pt1h, prec@pt5m, pres, retop@18dbz, rh, snowratio, t, ugrd, vgrd

---

## GRAF-SSDS60M — Detailed Variable Inventory

**Product:** Global Severe Storm Data Server, 4 km, hourly
**Format:** NetCDF (one composite file per valid time)
**File size:** ~4.7 GB per file
**Grid:** MPAS unstructured mesh — 4,794,413 cells covering the globe
**Model:** MPAS-A (atmosphere), run with WSM6 microphysics, RRTMG radiation, YSU PBL

Each file contains a single valid time (`Time=1`) and the following variables:

#### Coordinate / Metadata

| Variable | Description |
|----------|-------------|
| `xtime` | Model valid time string (YYYY-MM-DD_hh:mm:ss) |

#### 3-D Flight-Level Variables (50 flight levels, global)

These variables span 50 flight levels and are intended for aviation applications. The WINC
names are embedded in each variable's `long_name` attribute.

| Variable | Units | Description |
|----------|-------|-------------|
| `temperature_flightlevels` | K | Temperature at 50 flight levels |
| `relhum_flightlevels` | % | Relative humidity at 50 flight levels |
| `uzonal_flightlevels` | m/s | Zonal (east–west) wind at 50 flight levels |
| `umeridional_flightlevels` | m/s | Meridional (north–south) wind at 50 flight levels |

#### AGL Variables at 6 Fixed Heights (500 ft increments to 3000 ft AGL)

The same four quantities are provided at each of the following AGL heights:
500 ft, 1000 ft, 1500 ft, 2000 ft, 2500 ft, 3000 ft.

| Variable pattern | Units | Description |
|-----------------|-------|-------------|
| `temperature_<H>ft_agl` | K | Temperature at height H ft AGL |
| `relhum_<H>ft_agl` | % | Relative humidity at height H ft AGL |
| `uzonal_<H>ft_agl` | m/s | Zonal wind at height H ft AGL |
| `umeridional_<H>ft_agl` | m/s | Meridional wind at height H ft AGL |

#### Surface and Near-Surface Variables

| Variable | Units | WINC name | Description |
|----------|-------|-----------|-------------|
| `u10` | m/s | U.ZSFC@10m | 10-meter zonal wind |
| `v10` | m/s | V.ZSFC@10m | 10-meter meridional wind |
| `windspeed10m` | m/s | WSPD.ZSFC@10m | Wind speed at 10 m AGL |
| `windgust10m` | m/s | GUSTSPD.ZSFC@10m | Wind gust at 10 m AGL |
| `dewpoint_2m` | K | DPT.ZSFC@2m | Dewpoint temperature at 2 m AGL |
| `rh2m` | % | — | Relative humidity (liquid or ice) at 2 m AGL |
| `t2m` | K | T.ZSFC@2m | 2-meter temperature |
| `t2m_fslk` | K | — | Feels-like temperature at 2 m AGL |
| `t2m_wb` | K | — | Wet-bulb temperature at 2 m AGL |
| `mslp` | Pa | PMSL.MSL | Mean sea-level pressure |
| `visibility` | km | — | Visibility at surface |
| `snowh` | m | — | Physical snow depth |
| `frzlev` | m | — | Freezing level height |
| `ceiling_agl` | m | — | Ceiling height above ground level |
| `cape` | J/kg | CAPE.SFC | Convective available potential energy |
| `echotop` | ft/1000 | — | Aviation echo top |
| `swdnb` | W/m² | — | All-sky downward surface shortwave radiation flux |
| `total_cloud_cover` | % | — | Total cloud cover (derived from prate, hydrometeors, RH) |

#### Accumulated Precipitation Variables

| Variable | Units | WINC name | Description |
|----------|-------|-----------|-------------|
| `apcp` | mm | — | Accumulated total precipitation |
| `rain_total` | mm | CMLTRAIN.SFC | Accumulated liquid precipitation |
| `ice_total` | mm | CMLTICE.SFC | Accumulated freezing precipitation |
| `snow_total` | m | CMLTSNOW.SFC | Accumulated snow total (Kuchera/Cobb/Gottlieb or static ratio scheme) |

---

## GRAF-WXMIX — Detailed Variable Inventory

**Product:** CONUS mixed weather suite, 4 km, hourly
**Format:** GRIB2 (one file per variable/level combination per valid time)
**Grid:** Regular lat/lon, CONUS domain

Each valid time produces a separate GRIB2 file for each parameter–level combination listed
below. Variable names in GRIB2 (as decoded by wgrib2) are shown in the GRIB2 Name column.

#### Single-Level Surface and Near-Surface Variables

| Filename token | GRIB2 Name | Level | Description |
|----------------|------------|-------|-------------|
| `ceiling_sfc` | CEIL | Surface | Cloud ceiling height above ground |
| `ceiling_msl` | CEIL | Mean sea level | Cloud ceiling height above MSL |
| `dswrf@pt1h_sfc` | — | Surface | Downward shortwave radiation flux (1-hour accumulation) |
| `gustfctr_zsfc10m` | local param (cat=2, parm=254) | 10 m AGL | Wind gust factor (ratio of gust to mean wind) |
| `gustspd_zsfc10m` | GUST | 10 m AGL | Wind gust speed |
| `hterrain_sfc` | MTERH | Surface | Model terrain height |
| `landcov_sfc` | — | Surface | Land cover type |
| `lapr_sfc` | LAPR | Surface | Lapse rate |
| `lhflux@pt1h_sfc` | — | Surface | Latent heat flux (1-hour accumulation) |
| `pmsl_msl` | PRMSL | Mean sea level | Mean sea-level pressure |
| `pots_sfc` | TSTM | Surface | Potential temperature surplus at surface |
| `prate_sfc` | — | Surface | Instantaneous precipitation rate |
| `prec@pt1h_sfc` | — | Surface | Total precipitation (1-hour accumulation) |
| `pres_sfc` | PRES | Surface | Surface pressure |
| `qnh_msl` | local param (cat=193, parm=234) | Mean sea level | QNH altimeter setting |
| `rh_zsfc2m` | — | 2 m AGL | Relative humidity |
| `skint_sfc` | SKINT | Surface | Skin (radiative surface) temperature |
| `snowaccwe@pt1h_sfc` | — | Surface | Snow water equivalent accumulation (1-hour) |
| `swddni@pt1h_sfc` | — | Surface | Direct normal irradiance (1-hour accumulation) |
| `t_p850mbar` | TMP | 850 hPa | Temperature at 850 mb pressure level |
| `t_zmsl2m` | TMP | 2 m above MSL | Temperature 2 m above mean sea level |
| `t_zsfc2m` | TMP | 2 m AGL | Temperature 2 m above ground |
| `tcdc_cb` | TCDC | Cloud base | Total cloud cover at cloud base |
| `ugrd_zsfc10m` | — | 10 m AGL | U-component (zonal) wind |
| `vgrd_zsfc10m` | — | 10 m AGL | V-component (meridional) wind |
| `vis_sfc` | — | Surface | Visibility |

#### Multi-Level Variables (15 Eta levels, ~32–5135 m AGL)

The `_all` suffix denotes files containing data at 15 model-native Eta levels. The approximate
height of each level in meters AGL is: 32, 106, 197, 314, 462, 649, 880, 1163, 1504, 1909,
2386, 2940, 3579, 4308, 5135 m.

| Filename token | GRIB2 Name | Description |
|----------------|------------|-------------|
| `density_all` | DEN | Air density at 15 Eta levels |
| `dist_all` | — | Geometric height (distance) at 15 Eta levels |
| `t_all` | TMP | Temperature at 15 Eta levels |
| `u_all` | — | U-component wind at 15 Eta levels |
| `v_all` | — | V-component wind at 15 Eta levels |

#### Soil-Layer Variables

| Filename token | GRIB2 Name | Soil Layers | Description |
|----------------|------------|-------------|-------------|
| `t_dsfc` | UPLST | 0–0.07 m, 0.07–0.28 m, 0.28–1 m below ground | Upper-layer soil temperature (3 layers) |
| `vsmoist_dsfc` | local param (disc=2, cat=3, parm=199) | 0–0.07 m, 0.07–0.28 m, 0.28–1 m, 1–2.89 m below ground | Volumetric soil moisture (4 layers) |

---

## Vertical Levels Used (bucket-wide)

**Pressure levels:** p100mbar, p150mbar, p200mbar, p250mbar, p300mbar, p400mbar, p500mbar,
p600mbar, p700mbar, p850mbar, p925mbar, p1000mbar

**Height/surface levels:** sfc, msl, dsfc, zsfc2m, zsfc10m, zsfc1000m, zsfc3000m, zsfc5000m,
zsfc6000m, cb (cloud base), atmos, toa

**SSDS60M flight levels:** 50 levels (aviation standard flight levels, global)

**SSDS60M AGL levels:** 500, 1000, 1500, 2000, 2500, 3000 ft AGL

**WXMIX Eta levels:** 15 levels at ~32, 106, 197, 314, 462, 649, 880, 1163, 1504, 1909, 2386,
2940, 3579, 4308, 5135 m AGL

**WXMIX soil layers:** 0–0.07, 0.07–0.28, 0.28–1, 1–2.89 m below ground

---

## File Sizes (typical)

| Folder | Typical File Size |
|--------|------------------|
| GRAF-CPG (single var) | 0.05 – 3.4 MB |
| GRAF-HOURLY (single var) | 22 – 102 MB |
| GRAF-MEDIA5M | 0.04 – 1.3 MB |
| GRAF-MEDIA15M | 0.6 – 2.3 MB |
| GRAF-MEDIA30M | 1.8 – 3.5 MB |
| GRAF-MEDIA60M | 40 KB – 1.5 MB |
| GRAF-ENERGY | ~575 MB (composite NetCDF) |
| GRAF-SSDS60M | ~4.7 GB (composite NetCDF) |
| GRAF-STFOD | 0.2 MB – 1.6 GB (highly variable) |
| GRAF-WXMIX | 0.02 – 18 MB per variable file |

---

## Notes on File Naming vs. WINC

The actual filenames in this bucket **do not use WINC dot-separated notation** for individual
files. Instead they use underscore-separated fields in this order:
```
<PRODUCT>_deterministic_<domain>_<res>_<ref_time>_<valid_time>_<parameter>_<vert_level>.<ext>
```
The WINC reference describes a broader WSI naming standard that the GRAF product names are
loosely aligned with (same semantic fields: dataset, ref_time, valid_time, fcst_period encoded
as diff of those two times, location/resolution, parameter, vert_level, format). Many
SSDS60M variables carry explicit WINC names in their NetCDF `long_name` attributes.

---

## Next Steps: Archive Design

The next task is to design and implement a daily archive system that:

1. Identifies which product folders to archive (and which to skip)
2. Determines a local directory structure
3. Writes a script (likely Python + boto3 or AWS CLI) to sync selected folders daily
4. Handles deduplication and manages local disk usage
