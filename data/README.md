# Data for: Slope Activity Identification and Segment Risk Assessment at Mountainous Great Wall Sites — Xinguangwu Ming Great Wall, Shanxi, China

This repository contains the derived datasets used in the paper. Raw satellite imagery (Sentinel-1A SLC, GF-7 stereo pairs, Sentinel-2) is **not** redistributed here due to data volume and licensing; see *External data sources* below for access.

## Overview

| Folder | Content | Related paper section |
|---|---|---|
| `01_study_area/` | Great Wall centreline, 50 m wall segments, landslide zone outlines | Study area, Methods |
| `02_terrain/` | 10 m GF-7 derived DEM, hillshade, TWI | Terrain factors |
| `03_insar_deformation/` | LOS velocity measurement points within the 1 km buffer (T113, T40) and 8-epoch cumulative LOS displacement grids (T40, IDW interpolated to 10 m) | InSAR results |
| `04_c_index_landslide/` | C-index rasters (T113, T40) and connection zones of the two identified landslides | Landslide identification, downslope inversion |
| `05_slope_activity/` | Slope units within the 1 km buffer, activity comparison, active sample points, prediction report | Slope activity prediction |
| `06_structural_instability/` | Damage points (70 objects) interpreted from UAV imagery and field-checked, with per-segment assignment | Damage records |
| `07_segment_risk/` | Segment risk results (vector + 10 m rasters for the protection zone and construction control zone), risk values by segment and zone, damage–risk association statistics | Segment risk assessment |
| `08_rainfall_displacement/` | Monthly rainfall (37 months), displacement time series (81 epochs), joint time series, and the three representative monitoring points | Rainfall correspondence |

## File details

### 01_study_area
- `great_wall_centreline.shp` — Wall centreline (WGS84). Source: Great Wall Heritage website.
- `wall_segments_50m.gpkg` — Wall divided into 50 m segments (90 segments) used as risk assessment units.
- `landslide_zone2.kml`, `landslide_zone3.kml` — Outlines of the two identified potentially active landslide zones.

### 02_terrain
- `dem_10m.tif` — 10 m DEM generated from GF-7 stereo image pairs.
- `hillshade_10m.tif` — Hillshade derived from the DEM.
- `twi_10m.tif` — Topographic wetness index derived from the DEM.

### 03_insar_deformation
- `los_velocity_points_t113.csv`, `los_velocity_points_t40.csv` — LOS velocity of all valid measurement points within the 1 km wall buffer for ascending tracks T113 (286,936 points, 963 PS and 285,973 DS) and T40 (207,043 points, 392 PS and 206,651 DS). Columns: `lon`, `lat`, `z_m` (elevation), `velocity_mm_yr`, `coherence`, `scatterer_type` (PS/DS). Coordinates are WGS84. Per-epoch displacement time series of each point are held by the authors and available on request.
- `T40_CumLOS_YYYYMMDD_IDW20m_10m.tif` — Cumulative LOS displacement of track T40 on 8 dates (20221119 to 20251103), interpolated from measurement points (IDW, 20 m search, 10 m grid).

### 04_c_index_landslide
- `c_index_t113_clipped.tif`, `c_index_t40_clipped.tif` — InSAR sensitivity (C-index) rasters for both tracks.
- `zone2_connection_3d.gpkg`, `zone3_connection_3d.gpkg` — Connected zones used for landslide identification and downslope inversion in Zones 2 and 3.

### 05_slope_activity
- `slope_units_1km_buffer.gpkg` — Complete slope units within the 1 km buffer of the wall (including unevaluated units).
- `slope_activity_comparison.gpkg` — Predicted activity of slope units versus InSAR observed activity.
- `active_slope_sample_points.gpkg` — Sample points of InSAR active slope units.
- `slope_activity_prediction_report.md` — Model configuration and prediction report (information value + Firth penalized logistic regression, spatial cross-validation).

### 06_structural_instability
- `damage_points.gpkg` — 70 structural instability objects interpreted from UAV imagery and checked in the field.
- `damage_points_by_segment.csv` — Damage objects assigned to wall segments.
- `damage_point_coordinates.csv` — Coordinates of damage objects from point cloud surveys.

### 07_segment_risk
- `segment_risk_results.gpkg` — Segment risk index results for the 50 m segments.
- `protection_risk_10m.tif`, `control_risk_10m.tif` — Pixel level combined risk rasters for the protection zone and the construction control zone.
- `protection_zone_0_50m.gpkg` — Protection zone extent (0–50 m from the wall).
- `segment_risk_values.csv` — Risk index of each segment.
- `segment_risk_values_by_zone.csv` — Segment risk values by management zone.
- `damage_risk_association.csv` — Statistics linking damage density to risk classes.
- `verification.json` — Verification outputs of the risk ranking.

### 08_rainfall_displacement
- `monthly_rainfall_37months.csv` — Monthly rainfall aggregated from NASA POWER daily records.
- `displacement_timeseries_81epochs.csv` — Displacement time series at 81 acquisition epochs.
- `joint_displacement_timeseries.csv` — Joint time series used for plotting.
- `monitoring_point_zone1/2/3.gpkg` — The three representative monitoring points (P1, P2, P3).

## External data sources (not redistributed)

| Dataset | Source | Notes |
|---|---|---|
| Sentinel-1A SLC (146 scenes, tracks 113 and 40, Nov 2022–Nov 2025) | Copernicus Open Access Hub / ASF | Not included; VV polarization, IW mode |
| Copernicus DEM 30 m | Copernicus DEM – ESA | Auxiliary data for InSAR processing |
| GF-7 stereo imagery | Licensed; not redistributable | The 10 m DEM in `02_terrain/` is derived by the authors |
| Sentinel-2 L2A (NDVI, land cover) | ESA Copernicus Data Space | 10 m products |
| SoilGrids topsoil clay content | ISRIC SoilGrids (soilgrids.org) | 250 m |
| Global fault dataset | Public geological fault database | Distance to faults (DFA) |
| Daily rainfall | NASA POWER project | Aggregated to monthly in `08_.../monthly_rainfall_37months.csv` |
| Roads | OpenStreetMap (ODbL) | Distance to roads (DRO) requires attribution if reused |
| Great Wall centreline | Great Wall Heritage website | Vectorized boundary |
| UAV imagery and field surveys | Authors | Damage records |

## Coordinate reference and units

- Vector files are in WGS84 geographic coordinates unless otherwise noted; see individual file headers.
- Rasters are 10 m grids. LOS velocity and displacement units are mm/yr and mm, respectively.
- Sign convention: negative LOS values indicate movement towards the satellite (ascending tracks).

## Usage

These datasets are provided to support reproducibility of the paper. If you use them, please cite the paper:

> Zhang, Y., Yang, Y., Chen, S., Chen, S., Hu, Q., Ren, C., Guo, Y., Elbeltagi, A. M. E. (2026). Slope activity identification and segment risk assessment at mountainous Great Wall sites: a case study of the Xinguangwu Ming Great Wall in Shanxi. (to be updated upon publication)

OpenStreetMap derived data (roads) is licensed under ODbL; other public-source derived products follow their respective licenses.
