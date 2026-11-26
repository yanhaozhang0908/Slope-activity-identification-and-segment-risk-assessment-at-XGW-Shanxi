"""Aggregate the 10 m pixel-risk surface to 50 m Great Wall segments."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from osgeo import gdal, ogr


ROOT = Path(r"D:\project\shuozhouinflucefactor")
SOURCE_SEGMENTS = Path(
    r"D:\project\shuozhouinflucefactor\process\wall_100m_segment_risk_no_instability"
    r"\guangwu_wall_50m_segment_risk_100m_buffer_no_instability.gpkg"
)
PIXEL_RISK = (
    ROOT
    / "process"
    / "wall_pixel_risk_100m"
    / "guangwu_wall_100m_pixel_risk_score_10m.tif"
)
HILLSHADE = ROOT / "process" / "dem" / "dem_10m_hillshade_az315_alt42.tif"
OUT_DIR = ROOT / "process" / "wall_segment_risk_from_pixels"
OUTPUT_GPKG = OUT_DIR / "guangwu_wall_50m_segment_risk_from_10m_pixels.gpkg"
OUTPUT_CSV = OUT_DIR / "guangwu_wall_50m_segment_risk_from_10m_pixels.csv"
OUTPUT_SUMMARY = OUT_DIR / "guangwu_wall_50m_segment_risk_class_summary.csv"
OUTPUT_METADATA = OUT_DIR / "guangwu_wall_50m_segment_risk_metadata.json"
OUTPUT_FIGURE = OUT_DIR / "广武长城_像元汇总的50米分段风险评价图.png"
OUTPUT_FINE_FIGURE = OUT_DIR / "广武长城_50米分段风险得分八组精细显示图.png"

BUFFER_DISTANCE_M = 100.0
MEAN_WEIGHT = 0.70
P90_WEIGHT = 0.30
CLASS_DEFINITIONS = [
    ("Low", "低风险", 0.0, 20.0, "#5bb552"),
    ("Moderate-low", "较低风险", 20.0, 40.0, "#b1d75b"),
    ("Moderate", "中风险", 40.0, 60.0, "#fac74f"),
    ("High", "高风险", 60.0, 80.0, "#ee7c2e"),
    ("Very high", "极高风险", 80.0, 100.000001, "#ca3b33"),
]


FIELDS = [
    ("segment_id", ogr.OFTInteger),
    ("start_m", ogr.OFTReal),
    ("end_m", ogr.OFTReal),
    ("length_m", ogr.OFTReal),
    ("buffer_m", ogr.OFTReal),
    ("pix_count", ogr.OFTInteger),
    ("pix_mean", ogr.OFTReal),
    ("pix_p90", ogr.OFTReal),
    ("midhi_pct", ogr.OFTReal),
    ("high_pct", ogr.OFTReal),
    ("risk_score", ogr.OFTReal),
    ("risk_class", ogr.OFTString),
    ("score_grp8", ogr.OFTInteger),
    ("grp8_label", ogr.OFTString),
]


def classify(value: float) -> tuple[str, str]:
    for name, name_cn, lower, upper, _ in CLASS_DEFINITIONS:
        if lower <= value < upper:
            return name, name_cn
    return "Very high", "极高风险"


def pixel_values_in_polygon(
    array: np.ndarray, gt: tuple, nodata: float | None, polygon: ogr.Geometry
) -> np.ndarray:
    xmin, xmax, ymin, ymax = polygon.GetEnvelope()
    col0 = max(0, int(math.floor((xmin - gt[0]) / gt[1])))
    col1 = min(array.shape[1], int(math.ceil((xmax - gt[0]) / gt[1])))
    row0 = max(0, int(math.floor((gt[3] - ymax) / abs(gt[5]))))
    row1 = min(array.shape[0], int(math.ceil((gt[3] - ymin) / abs(gt[5]))))

    values = []
    point = ogr.Geometry(ogr.wkbPoint)
    for row in range(row0, row1):
        y = gt[3] + (row + 0.5) * gt[5]
        for col in range(col0, col1):
            value = float(array[row, col])
            if not np.isfinite(value) or (nodata is not None and np.isclose(value, nodata)):
                continue
            x = gt[0] + (col + 0.5) * gt[1]
            point.Empty()
            point.AddPoint(x, y)
            if polygon.Contains(point) or polygon.Touches(point):
                values.append(value)
    return np.asarray(values, dtype=float)


def set_fields(feature: ogr.Feature, record: dict) -> None:
    for name, _ in FIELDS:
        feature.SetField(name, record[name])


def geometry_parts(geometry: ogr.Geometry) -> list[np.ndarray]:
    flat = ogr.GT_Flatten(geometry.GetGeometryType())
    if flat in (ogr.wkbLineString, ogr.wkbLinearRing):
        return [
            np.asarray(
                [(geometry.GetX(i), geometry.GetY(i)) for i in range(geometry.GetPointCount())],
                dtype=float,
            )
        ]
    parts = []
    for index in range(geometry.GetGeometryCount()):
        parts.extend(geometry_parts(geometry.GetGeometryRef(index)))
    return [part for part in parts if len(part) >= 2]


def create_outputs(records: list[dict], srs) -> None:
    if OUTPUT_GPKG.exists():
        ogr.GetDriverByName("GPKG").DeleteDataSource(str(OUTPUT_GPKG))
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(str(OUTPUT_GPKG))
    line_layer = dataset.CreateLayer(
        "wall_segment_pixel_aggregated_risk",
        srs=srs,
        geom_type=ogr.wkbLineString,
        options=["GEOMETRY_NAME=geom", "SPATIAL_INDEX=YES"],
    )
    buffer_layer = dataset.CreateLayer(
        "wall_segment_100m_analysis_buffer",
        srs=srs,
        geom_type=ogr.wkbPolygon,
        options=["GEOMETRY_NAME=geom", "SPATIAL_INDEX=YES"],
    )
    for name, field_type in FIELDS:
        line_layer.CreateField(ogr.FieldDefn(name, field_type))
        buffer_layer.CreateField(ogr.FieldDefn(name, field_type))

    for record in records:
        line_feature = ogr.Feature(line_layer.GetLayerDefn())
        set_fields(line_feature, record)
        line_feature.SetGeometry(record["line"])
        line_layer.CreateFeature(line_feature)

        buffer_feature = ogr.Feature(buffer_layer.GetLayerDefn())
        set_fields(buffer_feature, record)
        buffer_feature.SetGeometry(record["buffer"])
        buffer_layer.CreateFeature(buffer_feature)
    dataset = None


def write_tables(records: list[dict]) -> None:
    fieldnames = [name for name, _ in FIELDS] + ["risk_class_cn"]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record[name] for name in fieldnames})

    total_length = sum(record["length_m"] for record in records)
    with OUTPUT_SUMMARY.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["risk_class", "risk_class_cn", "segment_count", "length_m", "length_km", "length_percent"],
        )
        writer.writeheader()
        for name, name_cn, _, _, _ in CLASS_DEFINITIONS:
            selected = [record for record in records if record["risk_class"] == name]
            length = sum(record["length_m"] for record in selected)
            writer.writerow(
                {
                    "risk_class": name,
                    "risk_class_cn": name_cn,
                    "segment_count": len(selected),
                    "length_m": length,
                    "length_km": length / 1000.0,
                    "length_percent": 100.0 * length / total_length,
                }
            )


def plot_result(records: list[dict], hillshade_path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    hillshade_ds = gdal.Open(str(hillshade_path))
    hillshade = hillshade_ds.GetRasterBand(1).ReadAsArray().astype(float)
    hillshade_nodata = hillshade_ds.GetRasterBand(1).GetNoDataValue()
    if hillshade_nodata is not None:
        hillshade[np.isclose(hillshade, hillshade_nodata)] = np.nan
    gt = hillshade_ds.GetGeoTransform()
    extent = [
        gt[0], gt[0] + hillshade_ds.RasterXSize * gt[1],
        gt[3] + hillshade_ds.RasterYSize * gt[5], gt[3]
    ]

    lines = []
    scores = []
    for record in records:
        for part in geometry_parts(record["line"]):
            lines.append(part)
            scores.append(record["risk_score"])

    xmin = min(part[:, 0].min() for part in lines) - 350
    xmax = max(part[:, 0].max() for part in lines) + 350
    ymin = min(part[:, 1].min() for part in lines) - 350
    ymax = max(part[:, 1].max() for part in lines) + 350

    colors = [item[4] for item in CLASS_DEFINITIONS]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([0, 20, 40, 60, 80, 100], cmap.N)

    figure, axis = plt.subplots(figsize=(8.2, 12.2), dpi=220)
    axis.imshow(hillshade, cmap="gray", extent=extent, origin="upper", vmin=20, vmax=245, alpha=0.76)
    collection = LineCollection(
        lines, cmap=cmap, norm=norm, linewidths=5.3, capstyle="round", joinstyle="round", zorder=4
    )
    collection.set_array(np.asarray(scores))
    axis.add_collection(collection)
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal")
    axis.set_xlabel("东坐标（m）")
    axis.set_ylabel("北坐标（m）")
    figure.suptitle(
        "广武长城50 m分段风险评价结果",
        fontsize=18,
        fontweight="bold",
        y=0.982,
    )
    axis.set_title(
        "由100 m缓冲区内10 m像元汇总：70%均值 + 30% P90；结构失稳数据未参与",
        fontsize=10.5,
        color="#555555",
        pad=12,
    )
    axis.ticklabel_format(style="plain", axis="both", useOffset=False)
    axis.grid(alpha=0.12, linewidth=0.5)

    colorbar = figure.colorbar(collection, ax=axis, orientation="horizontal", fraction=0.045, pad=0.075)
    colorbar.set_ticks([10, 30, 50, 70, 90])
    colorbar.set_ticklabels(["低风险", "较低风险", "中风险", "高风险", "极高风险"])
    colorbar.set_label("像元汇总后的分段风险等级（分段字段 risk_score 保留连续得分）", labelpad=8)

    figure.text(
        0.5, 0.015,
        "注：每个分段的100 m分析缓冲区允许与相邻分段重叠；地图仅以长城分段着色，避免缓冲面相互遮挡。",
        ha="center", va="bottom", fontsize=9.5, color="#555555"
    )
    figure.subplots_adjust(left=0.14, right=0.95, top=0.925, bottom=0.11)
    figure.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    hillshade_ds = None


def plot_fine_result(records: list[dict], hillshade_path: Path, group_edges: np.ndarray) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    hillshade_ds = gdal.Open(str(hillshade_path))
    hillshade = hillshade_ds.GetRasterBand(1).ReadAsArray().astype(float)
    hillshade_nodata = hillshade_ds.GetRasterBand(1).GetNoDataValue()
    if hillshade_nodata is not None:
        hillshade[np.isclose(hillshade, hillshade_nodata)] = np.nan
    gt = hillshade_ds.GetGeoTransform()
    extent = [
        gt[0], gt[0] + hillshade_ds.RasterXSize * gt[1],
        gt[3] + hillshade_ds.RasterYSize * gt[5], gt[3]
    ]

    lines = []
    groups = []
    endpoints = []
    for record in records:
        for part in geometry_parts(record["line"]):
            lines.append(part)
            groups.append(record["score_grp8"])
            endpoints.append(part[0])
    endpoints.append(lines[-1][-1])
    endpoints = np.asarray(endpoints)

    xmin = min(part[:, 0].min() for part in lines) - 350
    xmax = max(part[:, 0].max() for part in lines) + 350
    ymin = min(part[:, 1].min() for part in lines) - 350
    ymax = max(part[:, 1].max() for part in lines) + 350

    base_cmap = LinearSegmentedColormap.from_list(
        "fine_risk", ["#3f9f57", "#9dd35d", "#f3df63", "#f49a3f", "#c83b35"]
    )
    fine_colors = [base_cmap(value) for value in np.linspace(0.0, 1.0, 8)]
    cmap = ListedColormap(fine_colors)
    norm = BoundaryNorm(np.arange(0.5, 9.5, 1.0), cmap.N)

    figure, axis = plt.subplots(figsize=(8.5, 12.5), dpi=220)
    axis.imshow(hillshade, cmap="gray", extent=extent, origin="upper", vmin=20, vmax=245, alpha=0.74)
    collection = LineCollection(
        lines, cmap=cmap, norm=norm, linewidths=5.5, capstyle="butt", joinstyle="round", zorder=4
    )
    collection.set_array(np.asarray(groups, dtype=float))
    axis.add_collection(collection)
    axis.scatter(
        endpoints[:, 0], endpoints[:, 1], s=3.0, facecolors="white", edgecolors="#555555",
        linewidths=0.22, alpha=0.82, zorder=5
    )
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal")
    axis.set_xlabel("东坐标（m）")
    axis.set_ylabel("北坐标（m）")
    axis.ticklabel_format(style="plain", axis="both", useOffset=False)
    axis.grid(alpha=0.12, linewidth=0.5)

    figure.suptitle("广武长城50 m分段风险得分精细显示", fontsize=18, fontweight="bold", y=0.982)
    axis.set_title(
        "按分段综合得分由低到高等数量分为8组；白色细点表示50 m分段端点",
        fontsize=10.5, color="#555555", pad=12
    )

    colorbar = figure.colorbar(collection, ax=axis, orientation="horizontal", fraction=0.045, pad=0.075)
    colorbar.set_ticks(np.arange(1, 9))
    colorbar.set_ticklabels(
        [
            f"第{index + 1}组\n{group_edges[index]:.1f}–{group_edges[index + 1]:.1f}"
            for index in range(8)
        ]
    )
    colorbar.ax.tick_params(labelsize=8.2)
    colorbar.set_label("风险得分等数量分组（仅用于精细制图，不替代正式风险等级）", labelpad=9)
    figure.text(
        0.5, 0.014,
        "注：每组包含约19–20个分段；分段字段 risk_score 保留未经分组的连续得分。",
        ha="center", va="bottom", fontsize=9.5, color="#555555"
    )
    figure.subplots_adjust(left=0.14, right=0.95, top=0.925, bottom=0.12)
    figure.savefig(OUTPUT_FINE_FIGURE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    hillshade_ds = None


def main() -> None:
    gdal.UseExceptions()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raster_ds = gdal.Open(str(PIXEL_RISK))
    if raster_ds is None:
        raise FileNotFoundError(PIXEL_RISK)
    band = raster_ds.GetRasterBand(1)
    risk_array = band.ReadAsArray().astype(float)
    nodata = band.GetNoDataValue()
    gt = raster_ds.GetGeoTransform()

    source_ds = gdal.OpenEx(str(SOURCE_SEGMENTS), gdal.OF_VECTOR)
    source_layer = source_ds.GetLayerByName("wall_segment_risk")
    if source_layer is None:
        raise RuntimeError("Source segment layer not found")
    srs = source_layer.GetSpatialRef()

    records = []
    for feature in source_layer:
        line = feature.GetGeometryRef().Clone()
        buffer_geometry = line.Buffer(BUFFER_DISTANCE_M, 24)
        values = pixel_values_in_polygon(risk_array, gt, nodata, buffer_geometry)
        if values.size == 0:
            raise RuntimeError(f"No pixel values for segment {feature['segment_id']}")
        pixel_mean = float(np.mean(values))
        pixel_p90 = float(np.percentile(values, 90))
        score = MEAN_WEIGHT * pixel_mean + P90_WEIGHT * pixel_p90
        class_name, class_name_cn = classify(score)
        records.append(
            {
                "segment_id": int(feature["segment_id"]),
                "start_m": float(feature["start_m"]),
                "end_m": float(feature["end_m"]),
                "length_m": float(feature["end_m"] - feature["start_m"]),
                "buffer_m": BUFFER_DISTANCE_M,
                "pix_count": int(values.size),
                "pix_mean": pixel_mean,
                "pix_p90": pixel_p90,
                "midhi_pct": 100.0 * float(np.mean(values >= 40.0)),
                "high_pct": 100.0 * float(np.mean(values >= 60.0)),
                "risk_score": float(score),
                "risk_class": class_name,
                "risk_class_cn": class_name_cn,
                "line": line,
                "buffer": buffer_geometry,
            }
        )

    segment_scores = np.asarray([record["risk_score"] for record in records])
    group_edges = np.quantile(segment_scores, np.linspace(0.0, 1.0, 9))
    for record in records:
        group = int(np.searchsorted(group_edges[1:-1], record["risk_score"], side="right") + 1)
        record["score_grp8"] = group
        record["grp8_label"] = (
            f"第{group}组 {group_edges[group - 1]:.1f}–{group_edges[group]:.1f}"
        )

    create_outputs(records, srs)
    write_tables(records)
    plot_result(records, HILLSHADE)
    plot_fine_result(records, HILLSHADE, group_edges)

    scores = np.asarray([record["risk_score"] for record in records])
    metadata = {
        "source_pixel_risk": str(PIXEL_RISK),
        "segment_length_m": 50.0,
        "buffer_distance_each_side_m": BUFFER_DISTANCE_M,
        "aggregation": "0.70 * pixel mean + 0.30 * pixel P90",
        "structural_instability_used": False,
        "segment_count": len(records),
        "score_statistics": {
            "minimum": float(scores.min()),
            "maximum": float(scores.max()),
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "p90": float(np.percentile(scores, 90)),
        },
        "class_counts": {
            item[1]: sum(record["risk_class"] == item[0] for record in records)
            for item in CLASS_DEFINITIONS
        },
        "fine_display": {
            "method": "eight equal-count groups of segment risk_score; display only",
            "group_edges": [float(value) for value in group_edges],
            "group_counts": {
                str(group): sum(record["score_grp8"] == group for record in records)
                for group in range(1, 9)
            },
        },
        "outputs": {
            "geopackage": str(OUTPUT_GPKG),
            "segment_table": str(OUTPUT_CSV),
            "class_summary": str(OUTPUT_SUMMARY),
            "figure": str(OUTPUT_FIGURE),
            "fine_figure": str(OUTPUT_FINE_FIGURE),
        },
    }
    OUTPUT_METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    raster_ds = None
    source_ds = None


if __name__ == "__main__":
    main()
