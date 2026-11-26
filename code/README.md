# Code for: Slope Activity Identification and Segment Risk Assessment at Mountainous Great Wall Sites — Xinguangwu Ming Great Wall, Shanxi, China

Python scripts for the analysis pipeline of the paper. Companion datasets are provided in the `data` repository.

## Pipeline overview

Scripts are listed in the order of the paper's Methods. All scripts originally lived in a single `scripts/` directory (imports between scripts rely on this flat layout).

### Stage 1 — Terrain and environmental factors

| Script | Function |
|---|---|
| `prepare_dem_inputs.py` | Prepare GF-7 DEM tiles for analysis |
| `derive_dem_factors.py` | Derive elevation, slope gradient, slope aspect, topographic relief from the 10 m DEM |
| `create_dem_hillshade.py` | Generate hillshade rasters |
| `extract_twi_dem_coverage.py` | Compute the topographic wetness index (TWI) |
| `extract_drainage_network.py` | Extract the drainage network and distance to channels (DCH) |
| `build_expanded_static_factors.py` | Assemble static environmental factors per slope unit |
| `build_full_slope_static_factors.py` | Full-domain slope unit factor table (includes C-index geometry terms) |
| `prepare_full_slope_factor_domain.py` | Define the analysis domain for factor aggregation |
| `attach_soilgrids_wrb.py` | Attach SoilGrids topsoil clay content (CLAY) |
| `build_sentinel2_monthly_ndvi.py` | Build monthly NDVI from Sentinel-2 |
| `download_annual_landcover.py` | Download annual land cover for LCC/BLP |

### Stage 2 — Slope units

| Script | Function |
|---|---|
| `delineate_slope_units.py` | Delineate complete slope units from topographic and hydrological boundaries |
| `clip_paper_slope_units_to_1km_20260917.py` | Clip slope units to the 1 km wall buffer used in the paper |

### Stage 3 — InSAR aggregation

| Script | Function |
|---|---|
| `aggregate_insar_to_slope_units.py` | Aggregate PS/DS measurement points of both tracks to slope units |
| `prepare_slope_timeseries.py` | Prepare per-slope-unit displacement time series |

### Stage 4 — Slope activity prediction (information value + Firth penalized logistic regression)

| Script | Function |
|---|---|
| `run_susceptibility_models.py` | Model library: information value model, `firth_fit` (Firth penalized logistic regression), combined model |
| `run_insar_labeled_activity_susceptibility.py` | Activity prediction with InSAR-derived labels, fixed spatial cross-validation, ROC-AUC evaluation |
| `optimize_oof_ensembles.py` | Out-of-fold ensemble optimization |
| `audit_prediction_inputs.py` | Input auditing for prediction reproducibility |
| `export_susceptibility_outputs.py` | Export prediction results (maps and tables) |

### Stage 5 — Rainfall and deformation correspondence

| Script | Function |
|---|---|
| `prepare_nasa_power_climate.py` | Download and aggregate NASA POWER daily rainfall to monthly |
| `plot_three_zone_representative_points_20260916.py` | Representative monitoring point selection (P1, P2, P3) |
| `plot_three_zones_monthly_rain_los_20260916.py` | Cumulative LOS displacement versus monthly rainfall |
| `plot_displacement_precip_20260916.py` | Displacement–precipitation correspondence analysis |

### Stage 6 — Landslides and the wall

| Script | Function |
|---|---|
| `analyze_landslide_wall_impact.py` | Spatial relationship between identified landslides and wall segments |
| `export_landslide_wall_profiles.py` | Export landslide–wall profiles |

### Stage 7 — Segment risk assessment

| Script | Function |
|---|---|
| `create_wall_risk_actual_buffer.py` | Build wall risk assessment buffers |
| `evaluate_wall_pixel_risk_100m.py` | Pixel level combined risk (InSAR deformation, landslide impact, environmental conditions) |
| `aggregate_pixel_risk_to_wall_segments.py` | Aggregate pixel risk to 50 m wall segments |
| `evaluate_protection_control_risk.py` | Segment risk index for the protection zone and construction control zone; Poisson regression with segment length offset |
| `search_risk_weights.py` | Weight grid search (0.05 step, weights ≥ 0.10, sum = 1) with retention rate and Spearman correlation stability |
| `risk_weight_sensitivity_20260918.py` | Weight sensitivity analysis |
| `build_precomputed_10m_risk_display.py` | Precompute 10 m risk rasters for display |

### Stage 8 — Damage association

| Script | Function |
|---|---|
| `plot_risk_level_instability_density.py` | Damage density by risk class |
| `plot_risk_instability_integrated_figure.py` | Integrated damage–risk figure (circular shift test along the wall) |
| `analyze_protection_control_zones.py` | Comparison of the two management zones |

## Requirements

```
numpy
pandas
scipy
scikit-learn
geopandas
shapely
fiona
pyproj
rasterio
matplotlib
lxml
GDAL
```

GDAL Python bindings (`osgeo`) are required by several scripts; installing `gdal` via conda is the most reliable route.

## Notes on reproducibility

1. **Paths**: scripts reference the original working environment through a `ROOT` variable (absolute local path). To rerun, set `ROOT` to your local clone and place the data from the companion data repository under the expected subfolders, or adjust the input paths at the top of each script.
2. **InSAR time series processing** (PS/DS InSAR, orbit combination, unwrapping) was performed in commercial software and is not part of this repository; the measurement point velocities and cumulative displacement grids are provided in the data repository.
3. **C-index rasters** were computed from the GF-7 DEM and satellite viewing geometry (see paper Eq. 7); the raster products are included in the data repository.
4. Python 3.9+ is assumed.

## Citation

If you use this code, please cite the paper (citation to be updated upon publication).
