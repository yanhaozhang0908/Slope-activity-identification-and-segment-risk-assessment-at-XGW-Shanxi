from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from osgeo import gdal, ogr, osr
from scipy.spatial import cKDTree


ROOT = Path(r"D:\project\shuozhouinflucefactor")
WALL_PATH = Path(r"D:\shuozhou\data\shapefile\guangwu\长城连续\长城连续.shp")
LANDSLIDES = {
    "滑坡区域2": ROOT / "data" / "landslide" / "滑坡区域2.kml",
    "滑坡区域3": ROOT / "data" / "landslide" / "滑坡区域3.kml",
}

DEM_PATH = ROOT / "process" / "dem" / "dem_10m_analysis_extent.tif"
FACTOR_PATHS = {
    "slope_deg": ROOT / "process" / "dem" / "slope_deg.tif",
    "aspect_deg": ROOT / "process" / "dem" / "aspect_deg.tif",
    "relief_300m": ROOT / "process" / "dem" / "local_relief_300m.tif",
    "twi": ROOT / "process" / "dem" / "twi.tif",
    "distance_to_drainage_m": ROOT / "process" / "dem" / "distance_to_drainage_m.tif",
    "landcover": ROOT / "data" / "landcover" / "annual_10m" / "landcover_mode_2022_2025_10m.tif",
    "landcover_agreement": ROOT / "data" / "landcover" / "annual_10m" / "landcover_agreement_count_2022_2025.tif",
    "landcover_change": ROOT / "data" / "landcover" / "annual_10m" / "landcover_change_2022_2025.tif",
    "ndvi": ROOT / "data" / "ndvi" / "sentinel2_monthly" / "sentinel2_ndvi_monthly_202211_202512.tif",
}

OUT_DIR = ROOT / "process" / "landslide_wall_impact"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def load_union(path: Path, target_srs: osr.SpatialReference) -> ogr.Geometry:
    dataset = ogr.Open(str(path))
    if dataset is None:
        raise RuntimeError(f"Cannot open vector: {path}")
    layer = dataset.GetLayer(0)
    source_srs = layer.GetSpatialRef()
    transform = None
    if source_srs is not None and not source_srs.IsSame(target_srs):
        transform = osr.CoordinateTransformation(source_srs, target_srs)
    union = None
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            continue
        geom = geom.Clone()
        if transform is not None:
            geom.Transform(transform)
        union = geom if union is None else union.Union(geom)
    if union is None:
        raise RuntimeError(f"No geometry in {path}")
    return union


def geometry_vertices(geom: ogr.Geometry) -> list[tuple[float, float]]:
    flat = ogr.GT_Flatten(geom.GetGeometryType())
    if flat in (ogr.wkbPoint, ogr.wkbLineString, ogr.wkbLinearRing):
        return [(geom.GetX(i), geom.GetY(i)) for i in range(geom.GetPointCount())]
    points: list[tuple[float, float]] = []
    for i in range(geom.GetGeometryCount()):
        points.extend(geometry_vertices(geom.GetGeometryRef(i)))
    return points


def densify_geometry(geom: ogr.Geometry, spacing: float = 2.0) -> np.ndarray:
    flat = ogr.GT_Flatten(geom.GetGeometryType())
    sequences: list[list[tuple[float, float]]] = []

    def collect(item: ogr.Geometry) -> None:
        item_flat = ogr.GT_Flatten(item.GetGeometryType())
        if item_flat in (ogr.wkbLineString, ogr.wkbLinearRing):
            seq = [(item.GetX(i), item.GetY(i)) for i in range(item.GetPointCount())]
            if seq:
                sequences.append(seq)
            return
        if item_flat == ogr.wkbPolygon:
            for j in range(item.GetGeometryCount()):
                collect(item.GetGeometryRef(j))
            return
        for j in range(item.GetGeometryCount()):
            collect(item.GetGeometryRef(j))

    collect(geom)
    output: list[tuple[float, float]] = []
    for seq in sequences:
        if len(seq) == 1:
            output.append(seq[0])
            continue
        for a, b in zip(seq[:-1], seq[1:]):
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            count = max(1, int(math.ceil(length / spacing)))
            for t in np.linspace(0.0, 1.0, count, endpoint=False):
                output.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
        output.append(seq[-1])
    return np.asarray(output, dtype=float)


def raster_grid(dataset: gdal.Dataset) -> dict:
    return {
        "width": dataset.RasterXSize,
        "height": dataset.RasterYSize,
        "gt": dataset.GetGeoTransform(),
        "projection": dataset.GetProjection(),
    }


def aligned_array(path: Path, reference: gdal.Dataset, resample: str = "bilinear") -> np.ndarray:
    source = gdal.Open(str(path))
    if source is None:
        raise RuntimeError(f"Cannot open raster: {path}")
    ref_gt = reference.GetGeoTransform()
    ref_extent = (
        ref_gt[0],
        ref_gt[3] + reference.RasterYSize * ref_gt[5],
        ref_gt[0] + reference.RasterXSize * ref_gt[1],
        ref_gt[3],
    )
    method = gdal.GRA_NearestNeighbour if resample == "near" else gdal.GRA_Bilinear
    warped = gdal.Warp(
        "",
        source,
        format="MEM",
        dstSRS=reference.GetProjection(),
        outputBounds=ref_extent,
        width=reference.RasterXSize,
        height=reference.RasterYSize,
        resampleAlg=method,
    )
    array = warped.ReadAsArray()
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    output = array.astype("float64")
    for index in range(warped.RasterCount):
        band = warped.GetRasterBand(index + 1)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            output[index][np.isclose(output[index], nodata)] = np.nan
    return output


def rasterize_geometry(geom: ogr.Geometry, reference: gdal.Dataset) -> np.ndarray:
    memory_driver = ogr.GetDriverByName("Memory")
    vector = memory_driver.CreateDataSource("")
    srs = osr.SpatialReference()
    srs.ImportFromWkt(reference.GetProjection())
    layer = vector.CreateLayer("mask", srs=srs, geom_type=geom.GetGeometryType())
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(geom)
    layer.CreateFeature(feature)
    raster = gdal.GetDriverByName("MEM").Create(
        "", reference.RasterXSize, reference.RasterYSize, 1, gdal.GDT_Byte
    )
    raster.SetGeoTransform(reference.GetGeoTransform())
    raster.SetProjection(reference.GetProjection())
    gdal.RasterizeLayer(raster, [1], layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"])
    return raster.GetRasterBand(1).ReadAsArray().astype(bool)


def xy_from_rc(rows: np.ndarray, cols: np.ndarray, gt: tuple) -> tuple[np.ndarray, np.ndarray]:
    x = gt[0] + (cols + 0.5) * gt[1] + (rows + 0.5) * gt[2]
    y = gt[3] + (cols + 0.5) * gt[4] + (rows + 0.5) * gt[5]
    return x, y


def sample_array(array: np.ndarray, x: float, y: float, gt: tuple) -> float:
    inverse = gdal.InvGeoTransform(gt)
    if inverse is None:
        return float("nan")
    if len(inverse) == 2 and isinstance(inverse[0], (bool, int)):
        success, inv_gt = inverse
        if not success:
            return float("nan")
    else:
        inv_gt = inverse
    col_float, row_float = gdal.ApplyGeoTransform(inv_gt, x, y)
    row, col = int(math.floor(row_float)), int(math.floor(col_float))
    if row < 0 or col < 0 or row >= array.shape[0] or col >= array.shape[1]:
        return float("nan")
    return float(array[row, col])


def bearing_degrees(x1: float, y1: float, x2: float, y2: float) -> float:
    return (math.degrees(math.atan2(x2 - x1, y2 - y1)) + 360.0) % 360.0


def angular_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def circular_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    radians = np.deg2rad(values)
    return float((math.degrees(math.atan2(np.sin(radians).mean(), np.cos(radians).mean())) + 360.0) % 360.0)


def numeric_stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "p90": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p90": float(np.percentile(values, 90)),
    }


def class_proportions(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)].astype(int)
    values = values[(values > 0) & (values != 10)]
    if values.size == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    total = counts.sum()
    return {
        LANDCOVER_NAMES.get(int(code), f"类别{int(code)}"): float(count / total)
        for code, count in zip(unique, counts)
    }


def straight_profile(array: np.ndarray, gt: tuple, start: np.ndarray, end: np.ndarray, spacing: float = 5.0):
    distance = float(np.linalg.norm(end - start))
    count = max(2, int(math.ceil(distance / spacing)) + 1)
    fractions = np.linspace(0.0, 1.0, count)
    points = start[None, :] + fractions[:, None] * (end - start)[None, :]
    elevations = np.asarray([sample_array(array, x, y, gt) for x, y in points])
    return fractions * distance, elevations


def prepare_wall_chainage(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orient the wall from its southwestern end and return cumulative chainage."""
    if (points[0, 0] + points[0, 1]) > (points[-1, 0] + points[-1, 1]):
        points = points[::-1].copy()
    step_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(step_lengths)))
    return points, cumulative


def point_at_chainage(points: np.ndarray, cumulative: np.ndarray, chainage: float) -> np.ndarray:
    chainage = float(np.clip(chainage, 0.0, cumulative[-1]))
    index = int(np.searchsorted(cumulative, chainage, side="right") - 1)
    index = min(index, len(points) - 2)
    length = cumulative[index + 1] - cumulative[index]
    fraction = 0.0 if length <= 0 else (chainage - cumulative[index]) / length
    return points[index] + fraction * (points[index + 1] - points[index])


def main() -> None:
    gdal.UseExceptions()
    dem_ds = gdal.Open(str(DEM_PATH))
    if dem_ds is None:
        raise RuntimeError(f"Cannot open DEM: {DEM_PATH}")
    grid = raster_grid(dem_ds)
    dem = dem_ds.GetRasterBand(1).ReadAsArray().astype("float64")
    dem_nodata = dem_ds.GetRasterBand(1).GetNoDataValue()
    if dem_nodata is not None:
        dem[np.isclose(dem, dem_nodata)] = np.nan

    target_srs = osr.SpatialReference()
    target_srs.ImportFromWkt(grid["projection"])
    wall = load_union(WALL_PATH, target_srs)
    slides = {name: load_union(path, target_srs) for name, path in LANDSLIDES.items()}

    factors = {
        "slope_deg": aligned_array(FACTOR_PATHS["slope_deg"], dem_ds)[0],
        "aspect_deg": aligned_array(FACTOR_PATHS["aspect_deg"], dem_ds)[0],
        "relief_300m": aligned_array(FACTOR_PATHS["relief_300m"], dem_ds)[0],
        "twi": aligned_array(FACTOR_PATHS["twi"], dem_ds)[0],
        "distance_to_drainage_m": aligned_array(FACTOR_PATHS["distance_to_drainage_m"], dem_ds)[0],
        "landcover": aligned_array(FACTOR_PATHS["landcover"], dem_ds, "near")[0],
        "landcover_agreement": aligned_array(FACTOR_PATHS["landcover_agreement"], dem_ds, "near")[0],
        "landcover_change": aligned_array(FACTOR_PATHS["landcover_change"], dem_ds, "near")[0],
    }
    ndvi_cube = aligned_array(FACTOR_PATHS["ndvi"], dem_ds)
    ndvi_ds = gdal.Open(str(FACTOR_PATHS["ndvi"]))
    descriptions = [ndvi_ds.GetRasterBand(i + 1).GetDescription() for i in range(ndvi_ds.RasterCount)]
    growing_indices = [
        i for i, text in enumerate(descriptions)
        if any(f"{month:02d}" in text[-2:] for month in (5, 6, 7, 8, 9))
    ]
    if not growing_indices:
        growing_indices = list(range(ndvi_cube.shape[0]))
    with np.errstate(all="ignore"):
        growing_ndvi = np.nanmedian(ndvi_cube[growing_indices], axis=0)

    wall_points = densify_geometry(wall, 2.0)
    wall_tree = cKDTree(wall_points)
    results: list[dict] = []
    profile_data = {}

    for name, geom in slides.items():
        mask = rasterize_geometry(geom, dem_ds)
        rows, cols = np.where(mask & np.isfinite(dem))
        if rows.size == 0:
            raise RuntimeError(f"No DEM cells overlap {name}")
        xs, ys = xy_from_rc(rows, cols, grid["gt"])
        elevations = dem[rows, cols]

        boundary_points = densify_geometry(geom.Boundary(), 2.0)
        distances, wall_indices = wall_tree.query(boundary_points, k=1)
        nearest_slide_index = int(np.argmin(distances))
        slide_near = boundary_points[nearest_slide_index]
        wall_near = wall_points[int(wall_indices[nearest_slide_index])]
        min_distance = float(distances[nearest_slide_index])

        centroid = geom.Centroid()
        centroid_xy = np.asarray([centroid.GetX(), centroid.GetY()])
        wall_bearing = bearing_degrees(centroid_xy[0], centroid_xy[1], wall_near[0], wall_near[1])

        lower_threshold = np.nanpercentile(elevations, 15)
        upper_threshold = np.nanpercentile(elevations, 85)
        low_xy = np.asarray([np.mean(xs[elevations <= lower_threshold]), np.mean(ys[elevations <= lower_threshold])])
        high_xy = np.asarray([np.mean(xs[elevations >= upper_threshold]), np.mean(ys[elevations >= upper_threshold])])
        morph_downslope_bearing = bearing_degrees(high_xy[0], high_xy[1], low_xy[0], low_xy[1])
        mean_aspect = circular_mean(factors["aspect_deg"][mask])

        connection = ogr.Geometry(ogr.wkbLineString)
        connection.AddPoint(float(slide_near[0]), float(slide_near[1]))
        connection.AddPoint(float(wall_near[0]), float(wall_near[1]))
        corridor = connection.Buffer(30.0).Union(geom.Buffer(10.0))
        corridor_mask = rasterize_geometry(corridor, dem_ds)

        wall_elevation = sample_array(dem, float(wall_near[0]), float(wall_near[1]), grid["gt"])
        slide_near_elevation = sample_array(dem, float(slide_near[0]), float(slide_near[1]), grid["gt"])
        profile_distance, profile_elevation = straight_profile(dem, grid["gt"], slide_near, wall_near)
        profile_data[name] = (profile_distance, profile_elevation)

        wall_lengths = {}
        for radius in (100.0, 200.0, 500.0):
            intersection = wall.Intersection(geom.Buffer(radius))
            wall_lengths[f"wall_length_within_{int(radius)}m_m"] = float(intersection.Length())

        slide_lc = class_proportions(factors["landcover"][mask])
        corridor_lc = class_proportions(factors["landcover"][corridor_mask])
        changed = factors["landcover_change"][mask]
        changed = changed[np.isfinite(changed)]
        change_ratio = float(np.mean(changed > 0)) if changed.size else None
        agreement = factors["landcover_agreement"][mask]
        agreement = agreement[np.isfinite(agreement)]

        result = {
            "name": name,
            "area_m2": float(geom.GetArea()),
            "perimeter_m": float(geom.Boundary().Length()),
            "min_distance_to_wall_m": min_distance,
            "wall_length_within_buffers": wall_lengths,
            "centroid_x": float(centroid_xy[0]),
            "centroid_y": float(centroid_xy[1]),
            "nearest_slide_x": float(slide_near[0]),
            "nearest_slide_y": float(slide_near[1]),
            "nearest_wall_x": float(wall_near[0]),
            "nearest_wall_y": float(wall_near[1]),
            "bearing_to_wall_deg": wall_bearing,
            "morphologic_downslope_bearing_deg": morph_downslope_bearing,
            "mean_raster_aspect_deg": mean_aspect,
            "angle_downslope_to_wall_deg": angular_difference(morph_downslope_bearing, wall_bearing),
            "angle_aspect_to_wall_deg": angular_difference(mean_aspect, wall_bearing),
            "elevation_m": numeric_stats(elevations),
            "nearest_slide_elevation_m": slide_near_elevation,
            "nearest_wall_elevation_m": wall_elevation,
            "slide_to_wall_elevation_difference_m": (
                float(slide_near_elevation - wall_elevation)
                if np.isfinite(slide_near_elevation) and np.isfinite(wall_elevation)
                else None
            ),
            "slope_deg": numeric_stats(factors["slope_deg"][mask]),
            "relief_300m_m": numeric_stats(factors["relief_300m"][mask]),
            "twi": numeric_stats(factors["twi"][mask]),
            "distance_to_drainage_m": numeric_stats(factors["distance_to_drainage_m"][mask]),
            "growing_season_ndvi": numeric_stats(growing_ndvi[mask]),
            "corridor_growing_season_ndvi": numeric_stats(growing_ndvi[corridor_mask]),
            "landcover_proportions": slide_lc,
            "corridor_landcover_proportions": corridor_lc,
            "landcover_change_ratio_2022_2025": change_ratio,
            "landcover_agreement_mean_years": float(np.mean(agreement)) if agreement.size else None,
        }
        results.append(result)

    # Divide the continuous Great Wall polyline into 50 m units. The risk
    # coefficients are relative indices: each landslide's overall index is
    # attenuated linearly from its nearest wall point to zero at 500 m.
    wall_chain_points, wall_cumulative = prepare_wall_chainage(wall_points)
    segment_length_target = 50.0
    wall_total_length = float(wall_cumulative[-1])
    segment_starts = np.arange(0.0, wall_total_length, segment_length_target)
    segment_records: list[dict] = []
    overall_indices = {"滑坡区域2": 52.0, "滑坡区域3": 43.0}
    nearest_distances = {item["name"]: item["min_distance_to_wall_m"] for item in results}

    for segment_id, start_chainage in enumerate(segment_starts, start=1):
        end_chainage = min(start_chainage + segment_length_target, wall_total_length)
        midpoint_chainage = (start_chainage + end_chainage) / 2.0
        start_xy = point_at_chainage(wall_chain_points, wall_cumulative, start_chainage)
        end_xy = point_at_chainage(wall_chain_points, wall_cumulative, end_chainage)
        midpoint_xy = point_at_chainage(wall_chain_points, wall_cumulative, midpoint_chainage)
        midpoint_geometry = ogr.Geometry(ogr.wkbPoint)
        midpoint_geometry.AddPoint(float(midpoint_xy[0]), float(midpoint_xy[1]))

        contributions = {}
        distances_to_slides = {}
        for name in ("滑坡区域2", "滑坡区域3"):
            distance = float(midpoint_geometry.Distance(slides[name]))
            distances_to_slides[name] = distance
            nearest = nearest_distances[name]
            if distance >= 500.0:
                contribution = 0.0
            else:
                denominator = max(1.0, 500.0 - nearest)
                spatial_decay = float(np.clip((500.0 - distance) / denominator, 0.0, 1.0))
                contribution = overall_indices[name] * spatial_decay
            contributions[name] = contribution

        risk_zone2 = contributions["滑坡区域2"]
        risk_zone3 = contributions["滑坡区域3"]
        combined_risk = 100.0 * (
            1.0 - (1.0 - risk_zone2 / 100.0) * (1.0 - risk_zone3 / 100.0)
        )
        influenced2 = distances_to_slides["滑坡区域2"] < 500.0
        influenced3 = distances_to_slides["滑坡区域3"] < 500.0
        if influenced2 and influenced3:
            influence_type = "common"
        elif influenced2:
            influence_type = "zone2_only"
        elif influenced3:
            influence_type = "zone3_only"
        else:
            influence_type = "outside"

        segment_records.append({
            "segment_id": segment_id,
            "start_chainage_m": float(start_chainage),
            "end_chainage_m": float(end_chainage),
            "mid_chainage_m": float(midpoint_chainage),
            "segment_length_m": float(end_chainage - start_chainage),
            "start_x": float(start_xy[0]),
            "start_y": float(start_xy[1]),
            "end_x": float(end_xy[0]),
            "end_y": float(end_xy[1]),
            "distance_zone2_m": distances_to_slides["滑坡区域2"],
            "distance_zone3_m": distances_to_slides["滑坡区域3"],
            "risk_zone2_percent": float(risk_zone2),
            "risk_zone3_percent": float(risk_zone3),
            "combined_risk_percent": float(combined_risk),
            "influence_type": influence_type,
        })

    segment_csv_path = OUT_DIR / "wall_segment_relative_risk_50m.csv"
    with segment_csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(segment_records[0].keys()))
        writer.writeheader()
        writer.writerows(segment_records)

    common_records = [record for record in segment_records if record["influence_type"] == "common"]
    affected_records = [record for record in segment_records if record["influence_type"] != "outside"]
    common_length = sum(record["segment_length_m"] for record in common_records)
    common_mean_risk = (
        float(np.average(
            [record["combined_risk_percent"] for record in common_records],
            weights=[record["segment_length_m"] for record in common_records],
        )) if common_records else 0.0
    )
    common_max_risk = max(
        (record["combined_risk_percent"] for record in common_records), default=0.0
    )
    segment_summary = {
        "segment_length_m": segment_length_target,
        "chainage_origin": "southwestern end of the continuous Great Wall polyline",
        "wall_total_length_m": wall_total_length,
        "zone2_only_length_m": sum(
            record["segment_length_m"] for record in segment_records
            if record["influence_type"] == "zone2_only"
        ),
        "zone3_only_length_m": sum(
            record["segment_length_m"] for record in segment_records
            if record["influence_type"] == "zone3_only"
        ),
        "common_influence_length_m": common_length,
        "common_influence_start_chainage_m": (
            min(record["start_chainage_m"] for record in common_records) if common_records else None
        ),
        "common_influence_end_chainage_m": (
            max(record["end_chainage_m"] for record in common_records) if common_records else None
        ),
        "common_mean_combined_risk_percent": common_mean_risk,
        "common_max_combined_risk_percent": common_max_risk,
        "affected_max_combined_risk_percent": max(
            (record["combined_risk_percent"] for record in affected_records), default=0.0
        ),
        "calculation_note": (
            "Relative indices for internal segment prioritization, not calibrated event probabilities. "
            "Combined index uses 100*[1-(1-R2/100)*(1-R3/100)]."
        ),
    }
    segment_summary_path = OUT_DIR / "wall_segment_relative_risk_summary.json"
    segment_summary_path.write_text(
        json.dumps(segment_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    json_path = OUT_DIR / "landslide_wall_impact_metrics.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = OUT_DIR / "landslide_wall_impact_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "滑坡", "面积_m2", "距长城_m", "平均高程_m", "平均坡度_deg", "主坡向_deg",
            "下坡方向_deg", "指向长城方位_deg", "下坡方向与长城夹角_deg", "滑坡邻近点高于长城_m",
            "生长季NDVI中位数", "主要土地覆盖", "100m内长城长度_m", "200m内长城长度_m",
        ])
        for item in results:
            main_lc = max(item["landcover_proportions"].items(), key=lambda pair: pair[1])[0] if item["landcover_proportions"] else "无数据"
            writer.writerow([
                item["name"], item["area_m2"], item["min_distance_to_wall_m"], item["elevation_m"]["mean"],
                item["slope_deg"]["mean"], item["mean_raster_aspect_deg"], item["morphologic_downslope_bearing_deg"],
                item["bearing_to_wall_deg"], item["angle_downslope_to_wall_deg"], item["slide_to_wall_elevation_difference_m"],
                item["growing_season_ndvi"]["median"], main_lc,
                item["wall_length_within_buffers"]["wall_length_within_100m_m"],
                item["wall_length_within_buffers"]["wall_length_within_200m_m"],
            ])

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False
    figure = plt.figure(figsize=(12.8, 10.4))
    grid_spec = figure.add_gridspec(
        3, 2, width_ratios=[1.08, 1.0], height_ratios=[1.0, 1.0, 0.82],
        wspace=0.08, hspace=0.20,
    )
    map_axis = figure.add_subplot(grid_spec[:, 0])

    all_points = np.vstack([wall_points] + [densify_geometry(g, 5.0) for g in slides.values()])
    point_min = np.min(all_points, axis=0)
    point_max = np.max(all_points, axis=0)
    xmin, ymin = point_min[0] - 450, point_min[1] - 180
    xmax, ymax = point_max[0] + 450, point_max[1] + 180
    gt = grid["gt"]
    col0 = max(0, int((xmin - gt[0]) / gt[1]))
    col1 = min(dem.shape[1], int(math.ceil((xmax - gt[0]) / gt[1])))
    row0 = max(0, int((gt[3] - ymax) / abs(gt[5])))
    row1 = min(dem.shape[0], int(math.ceil((gt[3] - ymin) / abs(gt[5]))))
    dem_subset = dem[row0:row1, col0:col1]
    subset_extent = [
        gt[0] + col0 * gt[1], gt[0] + col1 * gt[1],
        gt[3] + row1 * gt[5], gt[3] + row0 * gt[5],
    ]
    light = LightSource(azdeg=315, altdeg=45)
    finite = np.where(np.isfinite(dem_subset), dem_subset, np.nanmedian(dem_subset))
    hillshade = light.hillshade(finite, vert_exag=1.2, dx=abs(gt[1]), dy=abs(gt[5]))
    map_axis.imshow(hillshade, cmap="gray", extent=subset_extent, origin="upper", alpha=0.75)
    map_axis.contour(
        np.linspace(subset_extent[0], subset_extent[1], dem_subset.shape[1]),
        np.linspace(subset_extent[3], subset_extent[2], dem_subset.shape[0]),
        finite,
        levels=10,
        colors="#7b6d5c",
        linewidths=0.35,
        alpha=0.55,
    )
    map_axis.plot(wall_points[:, 0], wall_points[:, 1], color="#b2182b", linewidth=2.2)
    colors = {"滑坡区域2": "#f28e2b", "滑坡区域3": "#4e79a7"}
    english_names = {"滑坡区域2": "Landslide zone 2", "滑坡区域3": "Landslide zone 3"}
    for item in results:
        name = item["name"]
        geom_points = densify_geometry(slides[name].Boundary(), 2.0)
        map_axis.fill(geom_points[:, 0], geom_points[:, 1], color=colors[name], alpha=0.45)
        map_axis.plot(geom_points[:, 0], geom_points[:, 1], color=colors[name], linewidth=1.5)

        influence_boundary = densify_geometry(slides[name].Buffer(500.0).Boundary(), 5.0)
        map_axis.plot(
            influence_boundary[:, 0], influence_boundary[:, 1], color=colors[name],
            linewidth=1.0, linestyle=(0, (5, 4)), alpha=0.9,
        )
        map_axis.plot(
            [item["nearest_slide_x"], item["nearest_wall_x"]],
            [item["nearest_slide_y"], item["nearest_wall_y"]],
            linestyle="--", color=colors[name], linewidth=1.4,
        )
        mid_x = (item["nearest_slide_x"] + item["nearest_wall_x"]) / 2
        mid_y = (item["nearest_slide_y"] + item["nearest_wall_y"]) / 2
        map_axis.text(mid_x, mid_y, f"{item['min_distance_to_wall_m']:.0f} m", fontsize=9,
                      bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5})

    # Draw independently affected and common-influence wall units above the
    # base wall. Each line is one 50 m chainage unit.
    segment_map_colors = {
        "zone2_only": colors["滑坡区域2"],
        "zone3_only": colors["滑坡区域3"],
        "common": "#7b3294",
    }
    for record in segment_records:
        influence_type = record["influence_type"]
        if influence_type == "outside":
            continue
        map_axis.plot(
            [record["start_x"], record["end_x"]],
            [record["start_y"], record["end_y"]],
            color=segment_map_colors[influence_type], linewidth=4.2,
            solid_capstyle="round", alpha=0.98, zorder=6,
        )
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)
    wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    utm49n = osr.SpatialReference()
    utm49n.ImportFromWkt(grid["projection"])
    utm49n.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    to_wgs84 = osr.CoordinateTransformation(utm49n, wgs84)
    to_utm49n = osr.CoordinateTransformation(wgs84, utm49n)

    corner_lonlat = [
        to_wgs84.TransformPoint(x, y)[:2]
        for x, y in ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax))
    ]
    lon_min = min(point[0] for point in corner_lonlat)
    lon_max = max(point[0] for point in corner_lonlat)
    lat_min = min(point[1] for point in corner_lonlat)
    lat_max = max(point[1] for point in corner_lonlat)
    lon_mid = (lon_min + lon_max) / 2.0
    lat_mid = (lat_min + lat_max) / 2.0

    def degree_ticks(value_min: float, value_max: float, step_minutes: float) -> np.ndarray:
        step = step_minutes / 60.0
        start = math.ceil(value_min / step) * step
        stop = math.floor(value_max / step) * step
        return np.arange(start, stop + step * 0.1, step)

    def dms_label(value: float, positive_suffix: str, negative_suffix: str) -> str:
        absolute = abs(value)
        degrees = int(math.floor(absolute))
        minutes = (absolute - degrees) * 60.0
        suffix = positive_suffix if value >= 0 else negative_suffix
        minute_text = f"{minutes:.0f}" if abs(minutes - round(minutes)) < 1e-6 else f"{minutes:.1f}"
        return f"{degrees}°{minute_text}′{suffix}"

    longitude_ticks = np.asarray([112.0 + 47.0 / 60.0, 112.0 + 48.0 / 60.0])
    latitude_ticks = degree_ticks(lat_min, lat_max, 1.0)
    longitude_positions = [to_utm49n.TransformPoint(float(lon), lat_mid)[0] for lon in longitude_ticks]
    latitude_positions = [to_utm49n.TransformPoint(lon_mid, float(lat))[1] for lat in latitude_ticks]
    map_axis.set_xticks(longitude_positions, [dms_label(float(lon), "E", "W") for lon in longitude_ticks])
    map_axis.set_yticks(latitude_positions, [dms_label(float(lat), "N", "S") for lat in latitude_ticks])
    map_axis.xaxis.set_ticks_position("top")
    map_axis.tick_params(
        axis="x", which="both", top=True, labeltop=True, bottom=False, labelbottom=False,
        direction="out", length=4, width=1,
    )
    map_axis.yaxis.set_ticks_position("left")
    map_axis.tick_params(
        axis="y", which="both", left=True, labelleft=True, right=False, labelright=False,
        direction="out", length=4, width=1, labelrotation=90,
    )
    map_axis.set_xlabel("")
    map_axis.set_ylabel("")
    map_axis.set_aspect("equal")
    legend_handles = [
        Line2D([0], [0], color="#b2182b", linewidth=2.2, label="Guangwu Great Wall"),
        Patch(facecolor=colors["滑坡区域2"], edgecolor=colors["滑坡区域2"], alpha=0.45,
              label="Landslide zone 2"),
        Patch(facecolor=colors["滑坡区域3"], edgecolor=colors["滑坡区域3"], alpha=0.45,
              label="Landslide zone 3"),
        Line2D([0], [0], color="#555555", linewidth=1.2, linestyle="--",
               label="Shortest distance"),
        Line2D([0], [0], color="#555555", linewidth=1.2, linestyle=(0, (5, 4)),
               label="500 m influence boundary"),
        Line2D([0], [0], color=colors["滑坡区域2"], linewidth=4.0,
               label="Zone 2-only influence"),
        Line2D([0], [0], color=colors["滑坡区域3"], linewidth=4.0,
               label="Zone 3-only influence"),
        Line2D([0], [0], color="#7b3294", linewidth=4.0,
               label="Common influence"),
    ]
    map_axis.legend(
        handles=legend_handles, loc="upper right", frameon=True, framealpha=0.9,
        fontsize=7.7, borderpad=0.5, labelspacing=0.35, handlelength=2.4,
    )
    map_axis.text(
        0.025, 0.975, "(a)", transform=map_axis.transAxes,
        ha="left", va="top", fontsize=14, fontweight="bold",
    )

    # North arrow and a 500 m scale bar use the projected UTM coordinates.
    map_axis.annotate(
        "N", xy=(0.075, 0.12), xytext=(0.075, 0.055), xycoords="axes fraction",
        ha="center", va="center", fontsize=11, fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 1.2},
    )
    scale_x0 = xmin + 0.73 * (xmax - xmin)
    scale_y = ymin + 0.035 * (ymax - ymin)
    map_axis.plot([scale_x0, scale_x0 + 500.0], [scale_y, scale_y], color="black", linewidth=2.0)
    map_axis.plot([scale_x0, scale_x0], [scale_y - 12, scale_y + 12], color="black", linewidth=1.2)
    map_axis.plot([scale_x0 + 500.0, scale_x0 + 500.0], [scale_y - 12, scale_y + 12],
                  color="black", linewidth=1.2)
    map_axis.text(scale_x0 + 250.0, scale_y + 22, "500 m", ha="center", va="bottom", fontsize=8.5)

    for index, item in enumerate(results):
        axis = figure.add_subplot(grid_spec[index, 1])
        distance, elevation = profile_data[item["name"]]
        axis.plot(distance, elevation, color=colors[item["name"]], linewidth=2)
        axis.fill_between(distance, elevation, np.nanmin(elevation) - 5, color=colors[item["name"]], alpha=0.15)
        axis.scatter([0, distance[-1]], [elevation[0], elevation[-1]], c=[colors[item["name"]], "#b2182b"], s=45, zorder=3)
        axis.text(0, elevation[0], " Landslide boundary", va="bottom", fontsize=9)
        axis.text(distance[-1], elevation[-1], " Great Wall", va="bottom", ha="right", fontsize=9)
        panel_letter = "b" if index == 0 else "c"
        axis.text(
            0.025, 0.975, f"({panel_letter}) {english_names[item['name']]}",
            transform=axis.transAxes, ha="left", va="top", fontsize=13,
            fontweight="bold",
        )

        if index == 0:
            direct_index, indirect_index, overall_index = 16, 67, 52
            elevation_difference = 76
        else:
            direct_index, indirect_index, overall_index = 10, 57, 43
            elevation_difference = 34
        information = (
            f"Distance = {item['min_distance_to_wall_m']:.0f} m\n"
            f"Elevation difference = +{elevation_difference} m\n"
            f"Direct impact index = {direct_index}%\n"
            f"Indirect stability index = {indirect_index}%\n"
            f"Overall impact index = {overall_index}%"
        )
        axis.text(
            0.965, 0.055, information, transform=axis.transAxes,
            ha="right", va="bottom", fontsize=8.7, linespacing=1.25,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white",
                  "edgecolor": colors[item["name"]], "alpha": 0.9, "linewidth": 1.0},
        )
        axis.annotate(
            "", xy=(0.68, 0.83), xytext=(0.31, 0.83), xycoords="axes fraction",
            arrowprops={"arrowstyle": "-|>", "color": "#555555", "linewidth": 1.0,
                        "linestyle": "--"},
        )
        axis.text(
            0.495, 0.85, "Potential retrogressive influence",
            transform=axis.transAxes, ha="center", va="bottom", fontsize=8.0,
            color="#444444",
        )
        axis.set_xlabel("Distance from landslide boundary (m)")
        axis.set_ylabel("Elevation (m)")
        axis.grid(alpha=0.25)

    risk_axis = figure.add_subplot(grid_spec[2, 1])
    segment_midpoints = np.asarray([record["mid_chainage_m"] for record in segment_records])
    zone2_risk = np.asarray([record["risk_zone2_percent"] for record in segment_records])
    zone3_risk = np.asarray([record["risk_zone3_percent"] for record in segment_records])
    combined_risk = np.asarray([record["combined_risk_percent"] for record in segment_records])

    for record in common_records:
        risk_axis.axvspan(
            record["start_chainage_m"], record["end_chainage_m"],
            color="#7b3294", alpha=0.10, linewidth=0,
        )
    risk_axis.plot(
        segment_midpoints, zone2_risk, color=colors["滑坡区域2"],
        linewidth=1.7, label="Zone 2 contribution",
    )
    risk_axis.plot(
        segment_midpoints, zone3_risk, color=colors["滑坡区域3"],
        linewidth=1.7, label="Zone 3 contribution",
    )
    risk_axis.plot(
        segment_midpoints, combined_risk, color="#7b3294",
        linewidth=2.4, label="Combined coefficient",
    )
    if affected_records:
        affected_start = min(record["start_chainage_m"] for record in affected_records)
        affected_end = max(record["end_chainage_m"] for record in affected_records)
        risk_axis.set_xlim(max(0.0, affected_start - 100.0), min(wall_total_length, affected_end + 100.0))
    risk_axis.set_ylim(0.0, max(60.0, float(np.nanmax(combined_risk)) + 8.0))
    risk_axis.set_xlabel("Chainage from southwestern end (m)")
    risk_axis.set_ylabel("Relative risk coefficient (%)")
    risk_axis.grid(alpha=0.25)
    risk_axis.text(
        0.025, 0.975, "(d) Wall-segment risk coefficient",
        transform=risk_axis.transAxes, ha="left", va="top",
        fontsize=12.5, fontweight="bold",
    )
    risk_axis.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    if common_records:
        common_start = min(record["start_chainage_m"] for record in common_records)
        common_end = max(record["end_chainage_m"] for record in common_records)
        risk_axis.text(
            (common_start + common_end) / 2.0,
            max(2.0, common_mean_risk * 0.45),
            f"Common-influence section\n{common_start:.0f}-{common_end:.0f} m",
            ha="center", va="center", fontsize=8.0, color="#5b1a6e",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": "#7b3294", "alpha": 0.85},
        )

    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=0.95)

    figure_path = OUT_DIR / "landslide_wall_segment_risk_final.png"
    figure_pdf_path = OUT_DIR / "landslide_wall_segment_risk_final.pdf"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    figure.savefig(figure_pdf_path, bbox_inches="tight")
    plt.close(figure)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"JSON={json_path}")
    print(f"CSV={csv_path}")
    print(f"FIGURE={figure_path}")
    print(f"PDF={figure_pdf_path}")


if __name__ == "__main__":
    main()
