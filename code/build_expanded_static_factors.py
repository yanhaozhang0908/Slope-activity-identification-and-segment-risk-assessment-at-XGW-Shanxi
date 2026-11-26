from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import gdal
from scipy.ndimage import distance_transform_edt


ROOT = Path(r"D:\project\shuozhouinflucefactor")
BASE_GPKG = (
    ROOT / "process" / "static_factors" / "slope_units_static_factors_full.gpkg"
)
LANDCOVER_DIR = (
    ROOT / "data" / "landcover" / "annual_10m_full_slope_units"
)
LANDCOVER_MODE = LANDCOVER_DIR / "landcover_mode_2022_2025_10m.tif"
LANDCOVER_CHANGE = LANDCOVER_DIR / "landcover_change_2022_2025.tif"
SOIL_DIR = ROOT / "data" / "soilgrids_properties"
OUTPUT_DIR = ROOT / "process" / "expanded_factors"
OUTPUT_GPKG = OUTPUT_DIR / "slope_units_expanded_factors.gpkg"
OUTPUT_CSV = OUTPUT_DIR / "slope_units_expanded_factors.csv"
SUMMARY_CSV = OUTPUT_DIR / "expanded_factor_summary.csv"
METADATA_JSON = OUTPUT_DIR / "expanded_factor_metadata.json"

LANDCOVER_CLASSES = {
    1: "water",
    2: "forest",
    4: "flooded_veg",
    5: "cropland",
    7: "built",
    8: "bare",
    9: "snow",
    11: "grass",
}

SOIL_LAYERS = {
    "SG_clay_top_pct": ("clay_0-5cm_mean_utm49n_250m.tif", 0.1),
    "SG_sand_top_pct": ("sand_0-5cm_mean_utm49n_250m.tif", 0.1),
    "SG_silt_top_pct": ("silt_0-5cm_mean_utm49n_250m.tif", 0.1),
    "SG_bdod_top_gcm3": ("bdod_0-5cm_mean_utm49n_250m.tif", 0.01),
    "SG_soc_top_gkg": ("soc_0-5cm_mean_utm49n_250m.tif", 0.1),
    "SG_cfvo_top_pct": ("cfvo_0-5cm_mean_utm49n_250m.tif", 0.1),
    "SG_cec_top_cmolkg": ("cec_0-5cm_mean_utm49n_250m.tif", 0.1),
    "SG_ph_top": ("phh2o_0-5cm_mean_utm49n_250m.tif", 0.1),
    "SG_clay_sub_pct": ("clay_15-30cm_mean_utm49n_250m.tif", 0.1),
    "SG_sand_sub_pct": ("sand_15-30cm_mean_utm49n_250m.tif", 0.1),
    "SG_silt_sub_pct": ("silt_15-30cm_mean_utm49n_250m.tif", 0.1),
    "SG_bdod_sub_gcm3": ("bdod_15-30cm_mean_utm49n_250m.tif", 0.01),
    "SG_cfvo_sub_pct": ("cfvo_15-30cm_mean_utm49n_250m.tif", 0.1),
}


def unit_id_raster(reference: Path) -> tuple[np.ndarray, tuple, str]:
    source = gdal.Open(str(reference))
    if source is None:
        raise FileNotFoundError(reference)
    target = gdal.GetDriverByName("MEM").Create(
        "", source.RasterXSize, source.RasterYSize, 1, gdal.GDT_Int32
    )
    target.SetGeoTransform(source.GetGeoTransform())
    target.SetProjection(source.GetProjection())
    target.GetRasterBand(1).Fill(0)
    vector = gdal.OpenEx(str(BASE_GPKG), gdal.OF_VECTOR)
    layer = vector.GetLayerByName("slope_units")
    result = gdal.RasterizeLayer(
        target,
        [1],
        layer,
        options=["ATTRIBUTE=SU_ID", "ALL_TOUCHED=FALSE"],
    )
    if result != 0:
        raise RuntimeError(f"Rasterizing slope units failed with code {result}")
    identifiers = target.GetRasterBand(1).ReadAsArray().astype("int32")
    transform = source.GetGeoTransform()
    projection = source.GetProjection()
    vector = None
    target = None
    source = None
    return identifiers, transform, projection


def aggregate_landcover(units: gpd.GeoDataFrame) -> pd.DataFrame:
    identifiers, transform, _ = unit_id_raster(LANDCOVER_MODE)
    source = gdal.Open(str(LANDCOVER_MODE))
    landcover = source.GetRasterBand(1).ReadAsArray()
    source = None
    source = gdal.Open(str(LANDCOVER_CHANGE))
    change = source.GetRasterBand(1).ReadAsArray()
    source = None

    valid = (identifiers > 0) & (landcover > 0) & (landcover != 10)
    edge = np.zeros(landcover.shape, dtype=bool)
    horizontal = (
        valid[:, 1:]
        & valid[:, :-1]
        & (landcover[:, 1:] != landcover[:, :-1])
    )
    vertical = (
        valid[1:, :]
        & valid[:-1, :]
        & (landcover[1:, :] != landcover[:-1, :])
    )
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:, :] |= vertical
    edge[:-1, :] |= vertical

    built = landcover == 7
    pixel_size = abs(float(transform[1]))
    built_distance = distance_transform_edt(~built) * pixel_size

    rows = []
    for unit_id in units["SU_ID"].astype(int):
        inside = (identifiers == unit_id) & valid
        values = landcover[inside]
        row: dict[str, float | int] = {"SU_ID": unit_id}
        if values.size == 0:
            for name in LANDCOVER_CLASSES.values():
                row[f"LCfrac_{name}"] = np.nan
            row.update(
                {
                    "LC_entropy": np.nan,
                    "LC_edge_fraction": np.nan,
                    "LC_change_fraction": np.nan,
                    "BUILTdist_mean_m": np.nan,
                }
            )
        else:
            proportions = []
            for code, name in LANDCOVER_CLASSES.items():
                fraction = float(np.mean(values == code))
                row[f"LCfrac_{name}"] = fraction
                if fraction > 0:
                    proportions.append(fraction)
            entropy = -sum(value * np.log(value) for value in proportions)
            denominator = np.log(max(2, len(proportions)))
            row["LC_entropy"] = float(entropy / denominator)
            row["LC_edge_fraction"] = float(np.mean(edge[inside]))
            row["LC_change_fraction"] = float(np.mean(change[inside] > 0))
            row["BUILTdist_mean_m"] = float(np.mean(built_distance[inside]))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_soil(
    units: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    result = pd.DataFrame({"SU_ID": units["SU_ID"].astype(int)})
    coverage: dict[str, dict[str, float]] = {}
    for field, (filename, scale) in SOIL_LAYERS.items():
        path = SOIL_DIR / filename
        if not path.exists():
            result[field] = np.nan
            coverage[field] = {"missing_file": 1.0}
            continue
        identifiers, _, _ = unit_id_raster(path)
        source = gdal.Open(str(path))
        band = source.GetRasterBand(1)
        array = band.ReadAsArray()
        nodata = band.GetNoDataValue()
        geotransform = source.GetGeoTransform()
        source = None
        values = np.asarray(array, dtype="float64") * scale
        if nodata is not None:
            values[array == nodata] = np.nan
        means = {}
        unit_lookup = units.set_index("SU_ID")
        for unit_id in units["SU_ID"].astype(int):
            selected = values[identifiers == unit_id]
            selected = selected[np.isfinite(selected)]
            if not selected.size:
                centroid = unit_lookup.loc[unit_id].geometry.centroid
                column = int((centroid.x - geotransform[0]) / geotransform[1])
                row = int((centroid.y - geotransform[3]) / geotransform[5])
                row = int(np.clip(row, 0, values.shape[0] - 1))
                column = int(np.clip(column, 0, values.shape[1] - 1))
                sampled = values[row, column]
                if np.isfinite(sampled):
                    selected = np.array([sampled])
            means[unit_id] = float(selected.mean()) if selected.size else np.nan
        result[field] = result["SU_ID"].map(means)
        coverage[field] = {
            "valid_unit_fraction": float(result[field].notna().mean()),
            "minimum": float(result[field].min()),
            "maximum": float(result[field].max()),
            "standard_deviation": float(result[field].std()),
        }

    result["SG_clay_vertical_diff"] = (
        result["SG_clay_sub_pct"] - result["SG_clay_top_pct"]
    )
    result["SG_sand_vertical_diff"] = (
        result["SG_sand_sub_pct"] - result["SG_sand_top_pct"]
    )
    result["SG_bdod_vertical_diff"] = (
        result["SG_bdod_sub_gcm3"] - result["SG_bdod_top_gcm3"]
    )
    result["SG_cfvo_vertical_diff"] = (
        result["SG_cfvo_sub_pct"] - result["SG_cfvo_top_pct"]
    )
    return result, coverage


def add_terrain_derivatives(units: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = units.copy()
    result["elev_range"] = result["elev_max"] - result["elev_min"]
    result["slope_cv"] = result["slope_std"] / result["slope_mean"].clip(lower=0.1)
    radians = np.deg2rad(result["aspect_mean"])
    result["northness"] = np.cos(radians) * result["aspect_R"]
    result["eastness"] = np.sin(radians) * result["aspect_R"]
    result["relief_elev_ratio"] = result["relief_mean"] / result["elev_mean"]
    result["LC_instability"] = 1.0 - result["LCstable4"]
    return result


def factor_summary(data: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows = []
    for field in fields:
        values = data[field]
        rows.append(
            {
                "factor": field,
                "dtype": str(values.dtype),
                "valid_count": int(values.notna().sum()),
                "missing_fraction": float(values.isna().mean()),
                "unique_count": int(values.nunique(dropna=True)),
                "minimum": (
                    float(values.min())
                    if pd.api.types.is_numeric_dtype(values)
                    else np.nan
                ),
                "maximum": (
                    float(values.max())
                    if pd.api.types.is_numeric_dtype(values)
                    else np.nan
                ),
                "standard_deviation": (
                    float(values.std())
                    if pd.api.types.is_numeric_dtype(values)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    units = gpd.read_file(BASE_GPKG, layer="slope_units")
    units = add_terrain_derivatives(units)
    landcover = aggregate_landcover(units)
    soil, soil_coverage = aggregate_soil(units)
    units = units.merge(landcover, on="SU_ID", how="left")
    units = units.merge(soil, on="SU_ID", how="left")

    new_fields = [
        "elev_range",
        "slope_cv",
        "northness",
        "eastness",
        "relief_elev_ratio",
        "LC_instability",
        *[field for field in landcover.columns if field != "SU_ID"],
        *[field for field in soil.columns if field != "SU_ID"],
    ]
    summary = factor_summary(units.drop(columns="geometry"), new_fields)
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()
    units.to_file(OUTPUT_GPKG, layer="slope_units", driver="GPKG")
    units.drop(columns="geometry").to_csv(
        OUTPUT_CSV, index=False, encoding="utf-8-sig"
    )

    metadata = {
        "unit_count": int(len(units)),
        "analysis_domain": (
            "270 complete slope units intersecting the 1 km core corridor"
        ),
        "new_factor_count": len(new_fields),
        "new_fields": new_fields,
        "sources": {
            "terrain": (
                "GF-7 10 m DEM derivatives already generated in process/dem"
            ),
            "landcover": (
                "Esri/Impact Observatory/Microsoft Sentinel-2 annual 10 m "
                "land cover, 2022-2025"
            ),
            "soil": (
                "ISRIC SoilGrids 2.0 continuous properties, 250 m, mean "
                "predictions for 0-5 cm and selected 15-30 cm layers"
            ),
            "soil_wcs": "https://maps.isric.org/mapserv",
        },
        "soil_units": {
            "clay_sand_silt": "percent by mass (source g/kg divided by 10)",
            "bulk_density": "g/cm3 (source cg/cm3 divided by 100)",
            "soil_organic_carbon": "g/kg (source dg/kg divided by 10)",
            "coarse_fragments": "volume percent (source cm3/dm3 divided by 10)",
            "cec": "cmol(c)/kg (source mmol(c)/kg divided by 10)",
            "ph": "pH (source value divided by 10)",
        },
        "soil_coverage": soil_coverage,
        "modelling_note": (
            "These are candidate factors. Low-variance fields and one member "
            "of strongly correlated groups must be removed before modelling."
        ),
    }
    METADATA_JSON.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
