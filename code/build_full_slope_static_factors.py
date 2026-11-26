from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import gdal


ROOT = Path(r"D:\project\shuozhouinflucefactor")
UNITS_GPKG = (
    ROOT
    / "process"
    / "factor_domain"
    / "slope_units_full_with_core_relation.gpkg"
)
LANDCOVER_DIR = (
    ROOT / "data" / "landcover" / "annual_10m_full_slope_units"
)
LANDCOVER_MODE = LANDCOVER_DIR / "landcover_mode_2022_2025_10m.tif"
LANDCOVER_AGREEMENT = (
    LANDCOVER_DIR / "landcover_agreement_count_2022_2025.tif"
)
FAULT_SHP = Path(r"D:\shuozhou\data\山西省断层\山西省.shp")
ROAD_SHP = Path(r"D:\shuozhou\data\shapefile\国道\国道.shp")
HIGHWAY_KML = Path(r"D:\shuozhou\data\shapefile\国道\G55高速.kml")
GEOLOGY_SHP = Path(r"D:\shuozhou\data\山西省地质图\山西省.shp")
OUTPUT_DIR = ROOT / "process" / "static_factors"
OUTPUT_GPKG = OUTPUT_DIR / "slope_units_static_factors_full.gpkg"
OUTPUT_CSV = OUTPUT_DIR / "slope_units_static_factors_full.csv"
OUTPUT_META = OUTPUT_DIR / "static_factor_metadata.json"
LANDSLIDE_FILES = [
    ROOT / "data" / "landslide" / "滑坡区域2.kml",
    ROOT / "data" / "landslide" / "滑坡区域3.kml",
]
UNSTABLE_FILE = ROOT / "data" / "landslide" / "非滑坡区域1.kml"

LANDCOVER_NAMES = {
    1: "水体",
    2: "林地",
    4: "淹水植被",
    5: "耕地",
    7: "建设用地",
    8: "裸地",
    9: "冰雪",
    10: "云",
    11: "草地/灌草地",
}


def raster_unit_statistics(
    units: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, dict]:
    source = gdal.Open(str(LANDCOVER_MODE))
    agreement_source = gdal.Open(str(LANDCOVER_AGREEMENT))
    if source is None or agreement_source is None:
        raise FileNotFoundError("Full-slope-unit land-cover rasters are missing")
    landcover = source.GetRasterBand(1).ReadAsArray().astype("uint8")
    agreement = agreement_source.GetRasterBand(1).ReadAsArray().astype("uint8")

    memory_vector = gdal.OpenEx(str(UNITS_GPKG), gdal.OF_VECTOR)
    layer = memory_vector.GetLayerByName("slope_units")
    unit_raster = gdal.GetDriverByName("MEM").Create(
        "", source.RasterXSize, source.RasterYSize, 1, gdal.GDT_Int32
    )
    unit_raster.SetGeoTransform(source.GetGeoTransform())
    unit_raster.SetProjection(source.GetProjection())
    unit_raster.GetRasterBand(1).Fill(0)
    result = gdal.RasterizeLayer(
        unit_raster,
        [1],
        layer,
        options=["ATTRIBUTE=SU_ID", "ALL_TOUCHED=FALSE"],
    )
    if result != 0:
        raise RuntimeError(f"Rasterizing slope units failed with code {result}")
    unit_ids = unit_raster.GetRasterBand(1).ReadAsArray().astype("int32")

    rows = []
    missing_units = []
    for _, row in units.iterrows():
        unit_id = int(row["SU_ID"])
        inside = unit_ids == unit_id
        valid = inside & (landcover != 0)
        values = landcover[valid]
        if values.size == 0:
            missing_units.append(unit_id)
            rows.append(
                {
                    "SU_ID": unit_id,
                    "LC_code": 0,
                    "LC_type": "无有效像元",
                    "LC_ratio": np.nan,
                    "LC_px_n": 0,
                    "LCagree_m": np.nan,
                    "LCstable4": np.nan,
                }
            )
            continue
        classes, counts = np.unique(values, return_counts=True)
        max_count = counts.max()
        candidates = classes[counts == max_count]
        if len(candidates) > 1:
            latest = gdal.Open(
                str(LANDCOVER_DIR / "landcover_2025_10m.tif")
            ).GetRasterBand(1).ReadAsArray()
            latest_values = latest[valid]
            latest_classes, latest_counts = np.unique(
                latest_values, return_counts=True
            )
            latest_mode = int(latest_classes[np.argmax(latest_counts)])
            mode = latest_mode if latest_mode in candidates else int(candidates[-1])
        else:
            mode = int(candidates[0])
        agreement_values = agreement[valid]
        rows.append(
            {
                "SU_ID": unit_id,
                "LC_code": mode,
                "LC_type": LANDCOVER_NAMES.get(mode, "未知"),
                "LC_ratio": float((values == mode).mean()),
                "LC_px_n": int(values.size),
                "LCagree_m": float(agreement_values.mean()),
                "LCstable4": float((agreement_values == 4).mean()),
            }
        )

    source = None
    agreement_source = None
    memory_vector = None
    unit_raster = None
    return pd.DataFrame(rows), {"landcover_missing_unit_ids": missing_units}


def nearest_distance(
    units: gpd.GeoDataFrame, lines: gpd.GeoDataFrame, field_name: str
) -> pd.Series:
    lines = lines[lines.geometry.notna() & ~lines.geometry.is_empty].copy()
    if lines.crs != units.crs:
        lines = lines.to_crs(units.crs)
    nearest = gpd.sjoin_nearest(
        units[["SU_ID", "geometry"]],
        lines[["geometry"]],
        how="left",
        distance_col=field_name,
    )
    return nearest.groupby("SU_ID")[field_name].min()


def road_distances(units: gpd.GeoDataFrame) -> tuple[pd.Series, dict]:
    road_frames = []
    sources = []
    for path in (ROAD_SHP, HIGHWAY_KML):
        if not path.exists():
            continue
        frame = gpd.read_file(path)
        frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
        if frame.empty:
            continue
        road_frames.append(frame.to_crs(units.crs))
        sources.append(str(path))
    if not road_frames:
        return pd.Series(dtype="float64"), {"road_sources": [], "road_count": 0}
    roads = gpd.GeoDataFrame(
        pd.concat(road_frames, ignore_index=True),
        geometry="geometry",
        crs=units.crs,
    )
    return nearest_distance(units, roads, "ROADdist_m"), {
        "road_sources": sources,
        "road_count": len(roads),
    }


def fault_distances(units: gpd.GeoDataFrame) -> tuple[pd.Series, dict]:
    faults = gpd.read_file(FAULT_SHP)
    distances = nearest_distance(units, faults, "FAULTdist_m")
    return distances, {
        "fault_source": str(FAULT_SHP),
        "fault_count": len(faults),
    }


def dominant_geology(
    units: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame, dict]:
    geology = gpd.read_file(GEOLOGY_SHP)
    if geology.crs != units.crs:
        geology = geology.to_crs(units.crs)
    minx, miny, maxx, maxy = units.total_bounds
    geology = geology.cx[minx:maxx, miny:maxy].copy()
    geology = geology[geology.geometry.notna() & ~geology.geometry.is_empty]
    spatial_index = geology.sindex
    rows = []
    for _, unit in units.iterrows():
        geometry = unit.geometry
        indices = spatial_index.query(geometry, predicate="intersects")
        candidates = geology.iloc[indices]
        if candidates.empty:
            rows.append(
                {
                    "SU_ID": int(unit["SU_ID"]),
                    "GEO_code": "",
                    "GEO_unit": "",
                    "GEO_ratio": np.nan,
                }
            )
            continue
        intersections = candidates.geometry.intersection(geometry)
        candidate = candidates.copy()
        candidate["_area"] = intersections.area
        grouped = (
            candidate.groupby(["O_CODE1", "UNITNAME"], dropna=False)["_area"]
            .sum()
            .sort_values(ascending=False)
        )
        code, name = grouped.index[0]
        rows.append(
            {
                "SU_ID": int(unit["SU_ID"]),
                "GEO_code": "" if pd.isna(code) else str(code),
                "GEO_unit": "" if pd.isna(name) else str(name),
                "GEO_ratio": float(grouped.iloc[0] / geometry.area),
            }
        )
    return pd.DataFrame(rows), {
        "geology_source": str(GEOLOGY_SHP),
        "geology_polygons_near_domain": len(geology),
        "geology_warning": (
            "仅保留原始地质图UNITNAME和O_CODE1；用户自编地质年代表不使用。"
            "正式入模前仍需核验原始发布单位、比例尺和类别含义。"
        ),
    }


def attach_inventory_labels(
    units: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, dict]:
    units = units.copy()
    units["LS_count"] = 0
    inventory_sources = []
    for index, path in enumerate(LANDSLIDE_FILES, start=2):
        field_name = f"LS{index}"
        units[field_name] = 0
        if not path.exists():
            continue
        inventory = gpd.read_file(path)
        if inventory.crs != units.crs:
            inventory = inventory.to_crs(units.crs)
        geometry = inventory.geometry.union_all()
        intersects = units.geometry.intersects(geometry).astype(int)
        units[field_name] = intersects
        units["LS_count"] += intersects
        inventory_sources.append(str(path))
    units["LS_label"] = (units["LS_count"] > 0).astype(int)

    units["UNSTABLE"] = 0
    if UNSTABLE_FILE.exists():
        unstable = gpd.read_file(UNSTABLE_FILE)
        if unstable.crs != units.crs:
            unstable = unstable.to_crs(units.crs)
        units["UNSTABLE"] = units.geometry.intersects(
            unstable.geometry.union_all()
        ).astype(int)
    units["keep_scope"] = (
        (units["retain_rule"] == 1)
        | (units["LS_label"] == 1)
        | (units["UNSTABLE"] == 1)
    ).astype(int)
    units["model_keep"] = (
        (units["keep_scope"] == 1) & (units["use_model"] == 1)
    ).astype(int)
    return units, {
        "landslide_inventory_sources": inventory_sources,
        "unstable_inventory_source": (
            str(UNSTABLE_FILE) if UNSTABLE_FILE.exists() else ""
        ),
        "landslide_positive_unit_count": int(units["LS_label"].sum()),
        "unstable_unit_count": int(units["UNSTABLE"].sum()),
        "keep_scope_unit_count": int(units["keep_scope"].sum()),
        "model_keep_unit_count": int(units["model_keep"].sum()),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    units = gpd.read_file(UNITS_GPKG, layer="slope_units")
    centroids = units.geometry.centroid
    units["Xc"] = centroids.x
    units["Yc"] = centroids.y

    landcover_stats, landcover_meta = raster_unit_statistics(units)
    units = units.merge(landcover_stats, on="SU_ID", how="left")

    road_series, road_meta = road_distances(units)
    units["ROADdist_m"] = units["SU_ID"].map(road_series)

    fault_series, fault_meta = fault_distances(units)
    units["FAULTdist_m"] = units["SU_ID"].map(fault_series)

    geology_stats, geology_meta = dominant_geology(units)
    units = units.merge(geology_stats, on="SU_ID", how="left")
    units["GEO_use"] = 0
    units, inventory_meta = attach_inventory_labels(units)

    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()
    units.to_file(OUTPUT_GPKG, layer="slope_units", driver="GPKG")
    units.drop(columns="geometry").to_csv(
        OUTPUT_CSV, index=False, encoding="utf-8-sig"
    )

    metadata = {
        "unit_count": len(units),
        "domain": "270 complete slope units intersecting the 1 km core area",
        "crs": str(units.crs),
        "continuous_terrain_fields_already_available": [
            column
            for column in [
                "elev_mean",
                "slope_mean",
                "slope_std",
                "aspect_mean",
                "aspect_R",
                "pcurv_mean",
                "plcurv_mean",
                "twi_mean",
                "relief_mean",
                "tpi_mean",
                "tri_mean",
                "drain_dist_mean",
            ]
            if column in units.columns
        ],
        "new_fields": {
            "LC_code": "2022-2025 modal land-cover class within the complete unit",
            "LC_type": "Chinese land-cover class name",
            "LC_ratio": "share of valid pixels occupied by LC_code",
            "LCagree_m": "mean number of annual maps agreeing with pixel modal class, 1-4",
            "LCstable4": "share of pixels whose class is identical in all four years",
            "ROADdist_m": "minimum planar distance from complete unit polygon to G208/G55 roads",
            "FAULTdist_m": "minimum planar distance from complete unit polygon to mapped faults",
            "GEO_code": "dominant raw O_CODE1 by intersection area",
            "GEO_unit": "dominant raw UNITNAME by intersection area",
            "GEO_ratio": "area share of dominant mapped geology unit",
            "GEO_use": "0: do not model until geology source and legend are verified",
            "LS_label": "1 if the complete slope unit intersects either mapped landslide",
            "LS2": "1 if the unit intersects 滑坡区域2",
            "LS3": "1 if the unit intersects 滑坡区域3",
            "UNSTABLE": "1 if the unit intersects the deformation area rejected by C-index",
            "keep_scope": "1 if retain_rule, LS_label or UNSTABLE is 1",
            "model_keep": "1 if keep_scope=1 and the existing use_model flag is 1",
            "Xc": "complete slope-unit centroid easting in EPSG:32649",
            "Yc": "complete slope-unit centroid northing in EPSG:32649",
        },
        **landcover_meta,
        **road_meta,
        **fault_meta,
        **geology_meta,
        **inventory_meta,
    }
    OUTPUT_META.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
