"""Recalculate geohazard risk for every 10 m pixel in the 100 m wall buffer.

Structural-instability observations are intentionally not read or used here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr

from analyze_landslide_wall_impact import (
    DEM_PATH,
    FACTOR_PATHS,
    LANDSLIDES,
    WALL_PATH,
    aligned_array,
    load_union,
    raster_grid,
)
from evaluate_wall_100m_segment_risk import (
    INSAR_TRACKS,
    load_insar_track,
    make_target_srs,
)


ROOT = Path(r"D:\project\shuozhouinflucefactor")
OUT_DIR = ROOT / "process" / "wall_pixel_risk_100m"

BUFFER_DISTANCE_M = 100.0
PIXEL_SIZE_M = 10.0
NODATA_FLOAT = -9999.0
NODATA_CLASS = 0

COMPONENT_WEIGHTS = {
    "environment": 0.50,
    "insar": 0.25,
    "landslide": 0.25,
}
ENVIRONMENT_WEIGHTS = {
    "slope": 0.30,
    "relief": 0.25,
    "twi": 0.20,
    "drainage": 0.15,
    "ndvi": 0.10,
}
LANDSLIDE_CONFIG = {
    "滑坡区域2": {"overall": 0.52, "nearest": 231.79445779263304},
    "滑坡区域3": {"overall": 0.43, "nearest": 335.9936470933815},
}
CLASS_DEFINITIONS = [
    (1, "Low", "低风险", 0.0, 20.0, (91, 181, 82, 255)),
    (2, "Moderate-low", "较低风险", 20.0, 40.0, (177, 215, 91, 255)),
    (3, "Moderate", "中风险", 40.0, 60.0, (250, 199, 79, 255)),
    (4, "High", "高风险", 60.0, 80.0, (238, 124, 46, 255)),
    (5, "Very high", "极高风险", 80.0, 100.000001, (202, 59, 51, 255)),
]


def rasterize_geometry(
    geometry: ogr.Geometry, grid: dict, burn_value: int = 1
) -> np.ndarray:
    memory_vector = ogr.GetDriverByName("Memory").CreateDataSource("")
    layer = memory_vector.CreateLayer(
        "mask", srs=make_target_srs(grid["projection"]), geom_type=ogr.wkbPolygon
    )
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(geometry)
    layer.CreateFeature(feature)

    memory_raster = gdal.GetDriverByName("MEM").Create(
        "", grid["width"], grid["height"], 1, gdal.GDT_Byte
    )
    memory_raster.SetGeoTransform(grid["gt"])
    memory_raster.SetProjection(grid["projection"])
    memory_raster.GetRasterBand(1).Fill(0)
    gdal.RasterizeLayer(memory_raster, [1], layer, burn_values=[burn_value])
    return memory_raster.GetRasterBand(1).ReadAsArray().astype(bool)


def robust_scale_in_mask(
    array: np.ndarray, mask: np.ndarray, invert: bool = False
) -> tuple[np.ndarray, tuple[float, float]]:
    values = array[mask & np.isfinite(array)]
    if values.size == 0:
        raise RuntimeError("No finite factor values in the wall buffer")
    low, high = np.nanpercentile(values, [5, 95])
    if high <= low:
        scaled = np.zeros(array.shape, dtype=np.float32)
    else:
        scaled = np.clip((array - low) / (high - low), 0.0, 1.0).astype(
            np.float32
        )
        scaled[~np.isfinite(scaled)] = 0.0
    if invert:
        scaled = 1.0 - scaled
    scaled[~mask] = np.nan
    return scaled, (float(low), float(high))


def pixel_centres(mask: np.ndarray, gt: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.where(mask)
    xs = gt[0] + (cols + 0.5) * gt[1] + (rows + 0.5) * gt[2]
    ys = gt[3] + (cols + 0.5) * gt[4] + (rows + 0.5) * gt[5]
    return rows, cols, np.column_stack([xs, ys])


def local_insar_score(
    tracks: dict, coordinates: np.ndarray, shape: tuple[int, int], rows: np.ndarray, cols: np.ndarray
) -> tuple[np.ndarray, dict]:
    track_p90 = []
    track_stats = {}
    for name, track in tracks.items():
        neighbour_lists = track["tree"].query_ball_point(
            coordinates, r=BUFFER_DISTANCE_M, workers=-1
        )
        values = np.full(coordinates.shape[0], np.nan, dtype=float)
        counts = np.zeros(coordinates.shape[0], dtype=np.int32)
        for index, neighbours in enumerate(neighbour_lists):
            if not neighbours:
                continue
            local_values = track["centered_abs"][np.asarray(neighbours, dtype=int)]
            local_values = local_values[np.isfinite(local_values)]
            if local_values.size:
                values[index] = np.nanpercentile(local_values, 90)
                counts[index] = local_values.size
        track_p90.append(values)
        track_stats[name] = {
            "pixels_with_points": int(np.count_nonzero(counts)),
            "maximum_point_count": int(np.max(counts)) if counts.size else 0,
        }

    stacked = np.column_stack(track_p90)
    finite_any = np.any(np.isfinite(stacked), axis=1)
    maximum = np.full(coordinates.shape[0], np.nan, dtype=float)
    maximum[finite_any] = np.nanmax(stacked[finite_any], axis=1)
    score_values = np.clip((maximum - 3.0) / 3.0, 0.0, 1.0)
    score_values[~np.isfinite(score_values)] = 0.0

    score = np.full(shape, np.nan, dtype=np.float32)
    score[rows, cols] = score_values.astype(np.float32)
    track_stats["pixels_without_insar_points"] = int(np.count_nonzero(~finite_any))
    return score, track_stats


def landslide_score(
    slides: dict, coordinates: np.ndarray, shape: tuple[int, int], rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    combined = np.ones(coordinates.shape[0], dtype=float)
    for name, geometry in slides.items():
        configuration = LANDSLIDE_CONFIG[name]
        contributions = np.zeros(coordinates.shape[0], dtype=float)
        point = ogr.Geometry(ogr.wkbPoint)
        for index, (x, y) in enumerate(coordinates):
            point.Empty()
            point.AddPoint(float(x), float(y))
            distance = float(point.Distance(geometry))
            if distance < 500.0:
                denominator = max(1.0, 500.0 - configuration["nearest"])
                decay = np.clip((500.0 - distance) / denominator, 0.0, 1.0)
                contributions[index] = configuration["overall"] * decay
        combined *= 1.0 - contributions

    score = np.full(shape, np.nan, dtype=np.float32)
    score[rows, cols] = (1.0 - combined).astype(np.float32)
    return score


def classify(score: np.ndarray, mask: np.ndarray) -> np.ndarray:
    classes = np.zeros(score.shape, dtype=np.uint8)
    for code, _, _, lower, upper, _ in CLASS_DEFINITIONS:
        selection = mask & np.isfinite(score) & (score >= lower) & (score < upper)
        classes[selection] = code
    return classes


def write_float_raster(path: Path, array: np.ndarray, reference: gdal.Dataset, description: str) -> None:
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(
        str(path),
        reference.RasterXSize,
        reference.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=3", "BIGTIFF=IF_SAFER"],
    )
    dataset.SetGeoTransform(reference.GetGeoTransform())
    dataset.SetProjection(reference.GetProjection())
    band = dataset.GetRasterBand(1)
    band.SetDescription(description)
    band.SetNoDataValue(NODATA_FLOAT)
    band.WriteArray(np.where(np.isfinite(array), array, NODATA_FLOAT).astype(np.float32))
    band.ComputeStatistics(False)
    dataset.BuildOverviews("AVERAGE", [2, 4, 8, 16])
    dataset.FlushCache()
    dataset = None


def write_component_raster(
    path: Path, arrays: list[np.ndarray], names: list[str], reference: gdal.Dataset
) -> None:
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path),
        reference.RasterXSize,
        reference.RasterYSize,
        len(arrays),
        gdal.GDT_Float32,
        options=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=3", "BIGTIFF=IF_SAFER"],
    )
    dataset.SetGeoTransform(reference.GetGeoTransform())
    dataset.SetProjection(reference.GetProjection())
    for index, (array, name) in enumerate(zip(arrays, names), start=1):
        band = dataset.GetRasterBand(index)
        band.SetDescription(name)
        band.SetNoDataValue(NODATA_FLOAT)
        band.WriteArray(np.where(np.isfinite(array), array, NODATA_FLOAT).astype(np.float32))
        band.ComputeStatistics(False)
    dataset.BuildOverviews("AVERAGE", [2, 4, 8, 16])
    dataset.FlushCache()
    dataset = None


def write_class_raster(path: Path, classes: np.ndarray, reference: gdal.Dataset) -> None:
    dataset = gdal.GetDriverByName("GTiff").Create(
        str(path),
        reference.RasterXSize,
        reference.RasterYSize,
        1,
        gdal.GDT_Byte,
        options=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )
    dataset.SetGeoTransform(reference.GetGeoTransform())
    dataset.SetProjection(reference.GetProjection())
    band = dataset.GetRasterBand(1)
    band.SetDescription("Risk class: 1 Low, 2 Moderate-low, 3 Moderate, 4 High, 5 Very high")
    band.SetNoDataValue(NODATA_CLASS)
    color_table = gdal.ColorTable()
    color_table.SetColorEntry(0, (0, 0, 0, 0))
    for code, _, _, _, _, rgba in CLASS_DEFINITIONS:
        color_table.SetColorEntry(code, rgba)
    band.SetRasterColorTable(color_table)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    band.SetCategoryNames(["NoData"] + [item[2] for item in CLASS_DEFINITIONS])
    band.WriteArray(classes)
    band.ComputeStatistics(False)
    dataset.BuildOverviews("NEAREST", [2, 4, 8, 16])
    dataset.FlushCache()
    dataset = None


def write_buffer_vector(path: Path, geometry: ogr.Geometry, projection: str) -> None:
    if path.exists():
        ogr.GetDriverByName("GPKG").DeleteDataSource(str(path))
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    srs = make_target_srs(projection)
    layer = dataset.CreateLayer(
        "wall_buffer_100m", srs=srs, geom_type=ogr.wkbPolygon,
        options=["GEOMETRY_NAME=geom", "SPATIAL_INDEX=YES"]
    )
    layer.CreateField(ogr.FieldDefn("buffer_m", ogr.OFTReal))
    layer.CreateField(ogr.FieldDefn("description", ogr.OFTString))
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetField("buffer_m", BUFFER_DISTANCE_M)
    feature.SetField("description", "Wall centreline buffer: 100 m on each side")
    feature.SetGeometry(geometry)
    layer.CreateFeature(feature)
    dataset = None


def main() -> None:
    gdal.UseExceptions()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dem_ds = gdal.Open(str(DEM_PATH))
    if dem_ds is None:
        raise FileNotFoundError(DEM_PATH)
    grid = raster_grid(dem_ds)
    target_srs = make_target_srs(grid["projection"])
    dem = dem_ds.GetRasterBand(1).ReadAsArray().astype(float)
    dem_nodata = dem_ds.GetRasterBand(1).GetNoDataValue()
    if dem_nodata is not None:
        dem[np.isclose(dem, dem_nodata)] = np.nan

    wall = load_union(WALL_PATH, target_srs)
    wall_buffer = wall.Buffer(BUFFER_DISTANCE_M, 32)
    buffer_mask = rasterize_geometry(wall_buffer, grid)
    valid_mask = buffer_mask & np.isfinite(dem)
    rows, cols, coordinates = pixel_centres(valid_mask, grid["gt"])

    factors = {
        "slope": aligned_array(FACTOR_PATHS["slope_deg"], dem_ds)[0],
        "relief": aligned_array(FACTOR_PATHS["relief_300m"], dem_ds)[0],
        "twi": aligned_array(FACTOR_PATHS["twi"], dem_ds)[0],
        "drainage": aligned_array(FACTOR_PATHS["distance_to_drainage_m"], dem_ds)[0],
    }
    ndvi_cube = aligned_array(FACTOR_PATHS["ndvi"], dem_ds)
    ndvi_ds = gdal.Open(str(FACTOR_PATHS["ndvi"]))
    descriptions = [
        ndvi_ds.GetRasterBand(index + 1).GetDescription()
        for index in range(ndvi_ds.RasterCount)
    ]
    growing_indices = [
        index
        for index, description in enumerate(descriptions)
        if any(f"{month:02d}" in description[-2:] for month in (5, 6, 7, 8, 9))
    ]
    if not growing_indices:
        growing_indices = list(range(ndvi_cube.shape[0]))
    with np.errstate(all="ignore"):
        factors["ndvi"] = np.nanmedian(ndvi_cube[growing_indices], axis=0)

    scaling = {}
    slope_score, scaling["slope"] = robust_scale_in_mask(factors["slope"], valid_mask)
    relief_score, scaling["relief"] = robust_scale_in_mask(factors["relief"], valid_mask)
    twi_score, scaling["twi"] = robust_scale_in_mask(factors["twi"], valid_mask)
    drainage_score, scaling["drainage"] = robust_scale_in_mask(
        factors["drainage"], valid_mask, invert=True
    )
    ndvi_score, scaling["ndvi"] = robust_scale_in_mask(
        factors["ndvi"], valid_mask, invert=True
    )
    environment = (
        ENVIRONMENT_WEIGHTS["slope"] * slope_score
        + ENVIRONMENT_WEIGHTS["relief"] * relief_score
        + ENVIRONMENT_WEIGHTS["twi"] * twi_score
        + ENVIRONMENT_WEIGHTS["drainage"] * drainage_score
        + ENVIRONMENT_WEIGHTS["ndvi"] * ndvi_score
    ).astype(np.float32)
    environment[~valid_mask] = np.nan

    tracks = {
        name: load_insar_track(path, target_srs) for name, path in INSAR_TRACKS.items()
    }
    insar, insar_stats = local_insar_score(
        tracks, coordinates, dem.shape, rows, cols
    )
    slides = {name: load_union(path, target_srs) for name, path in LANDSLIDES.items()}
    landslide = landslide_score(slides, coordinates, dem.shape, rows, cols)

    risk_score = 100.0 * (
        COMPONENT_WEIGHTS["environment"] * environment
        + COMPONENT_WEIGHTS["insar"] * insar
        + COMPONENT_WEIGHTS["landslide"] * landslide
    )
    risk_score = risk_score.astype(np.float32)
    risk_score[~valid_mask] = np.nan
    risk_class = classify(risk_score, valid_mask)

    score_path = OUT_DIR / "guangwu_wall_100m_pixel_risk_score_10m.tif"
    class_path = OUT_DIR / "guangwu_wall_100m_pixel_risk_class_10m.tif"
    component_path = OUT_DIR / "guangwu_wall_100m_pixel_risk_components_10m.tif"
    buffer_path = OUT_DIR / "guangwu_wall_actual_100m_buffer.gpkg"
    summary_path = OUT_DIR / "guangwu_wall_100m_pixel_risk_class_summary.csv"
    metadata_path = OUT_DIR / "guangwu_wall_100m_pixel_risk_metadata.json"

    write_float_raster(score_path, risk_score, dem_ds, "Geohazard risk score (0-100)")
    write_class_raster(class_path, risk_class, dem_ds)
    write_component_raster(
        component_path,
        [environment, insar, landslide],
        ["Environment score (0-1)", "InSAR score (0-1)", "Landslide score (0-1)"],
        dem_ds,
    )
    write_buffer_vector(buffer_path, wall_buffer, grid["projection"])

    total_pixels = int(np.count_nonzero(valid_mask))
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "class_code", "class_name", "class_name_cn", "score_range",
                "pixel_count", "area_m2", "area_km2", "percent"
            ],
        )
        writer.writeheader()
        for code, name, name_cn, lower, upper, _ in CLASS_DEFINITIONS:
            count = int(np.count_nonzero(risk_class == code))
            writer.writerow(
                {
                    "class_code": code,
                    "class_name": name,
                    "class_name_cn": name_cn,
                    "score_range": f"[{lower:g}, {upper if upper <= 100 else 100:g})",
                    "pixel_count": count,
                    "area_m2": count * PIXEL_SIZE_M * PIXEL_SIZE_M,
                    "area_km2": count * PIXEL_SIZE_M * PIXEL_SIZE_M / 1_000_000.0,
                    "percent": 100.0 * count / total_pixels if total_pixels else 0.0,
                }
            )

    finite_scores = risk_score[valid_mask & np.isfinite(risk_score)]
    metadata = {
        "method": "10 m pixel risk evaluation inside a 100 m buffer on each side of the wall",
        "structural_instability_used": False,
        "crs": "EPSG:32649",
        "pixel_size_m": PIXEL_SIZE_M,
        "buffer_distance_each_side_m": BUFFER_DISTANCE_M,
        "component_weights": COMPONENT_WEIGHTS,
        "environment_weights": ENVIRONMENT_WEIGHTS,
        "robust_scaling_percentiles_5_95": scaling,
        "insar_method": "90th percentile of median-centred absolute velocity within 100 m; <=3 mm/a score 0, >=6 mm/a score 1",
        "insar_statistics": insar_stats,
        "landslide_method": "distance-decay within 500 m, matching the segment-risk model",
        "valid_pixel_count": total_pixels,
        "evaluated_area_km2": total_pixels * PIXEL_SIZE_M * PIXEL_SIZE_M / 1_000_000.0,
        "risk_score_statistics": {
            "minimum": float(np.nanmin(finite_scores)),
            "maximum": float(np.nanmax(finite_scores)),
            "mean": float(np.nanmean(finite_scores)),
            "median": float(np.nanmedian(finite_scores)),
            "p05": float(np.nanpercentile(finite_scores, 5)),
            "p95": float(np.nanpercentile(finite_scores, 95)),
        },
        "outputs": {
            "risk_score": str(score_path),
            "risk_class": str(class_path),
            "components": str(component_path),
            "buffer": str(buffer_path),
            "class_summary": str(summary_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("class_counts", {item[2]: int(np.count_nonzero(risk_class == item[0])) for item in CLASS_DEFINITIONS})
    dem_ds = None


if __name__ == "__main__":
    main()
