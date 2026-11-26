"""Compare Guangwu Great Wall influence in legally motivated distance zones.

The available wall geometry is a line, so distances are measured from that line
as a proxy for the wall-body outer edge.  Results are analytical approximations,
not surveyed legal boundaries.
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, ogr

from analyze_landslide_wall_impact import DEM_PATH, FACTOR_PATHS, LANDSLIDES, aligned_array, load_union, raster_grid
from analyze_near_vs_outer_influence import (
    INDICATOR_LABELS,
    bootstrap_ratio_ci,
    finite_values,
    interpretation,
    mean_or_nan,
)
from evaluate_wall_100m_segment_risk import INSAR_TRACKS, load_insar_track, make_target_srs
from evaluate_wall_pixel_risk_100m import (
    COMPONENT_WEIGHTS,
    ENVIRONMENT_WEIGHTS,
    landslide_score,
    local_insar_score,
    pixel_centres,
    rasterize_geometry,
    robust_scale_in_mask,
    write_component_raster,
)


ROOT = Path(r"D:\project\shuozhouinflucefactor")
SEGMENTS = ROOT / "process" / "wall_100m_segment_risk_no_instability" / "guangwu_wall_50m_segment_risk_100m_buffer_no_instability.gpkg"
OUT_DIR = ROOT / "process" / "protection_control_zone_comparison"

PROTECTION_M = 50.0
RURAL_OUTER_M = 550.0
PIXEL_SIZE_M = 10.0
RANDOM_SEED = 20260904
SPATIAL_BLOCKS = 8

ZONE_LABELS = {
    "protection_0_50m": "保护范围 0–50 m",
    "rural_control_50_550m": "农村/郊野建控带 50–550 m",
}


def build_wall(layer) -> ogr.Geometry:
    wall = None
    layer.ResetReading()
    for feature in layer:
        geometry = feature.GetGeometryRef()
        if geometry is None:
            continue
        clone = geometry.Clone()
        wall = clone if wall is None else wall.Union(clone)
    if wall is None:
        raise RuntimeError("No wall geometry")
    return wall


def make_zone_geometries(wall: ogr.Geometry) -> dict[str, ogr.Geometry]:
    protection = wall.Buffer(PROTECTION_M, 32)
    rural_total = wall.Buffer(RURAL_OUTER_M, 32)
    return {
        "protection_0_50m": protection,
        "rural_control_50_550m": rural_total.Difference(protection),
    }


def write_zones(path: Path, geometries: dict[str, ogr.Geometry], srs) -> None:
    if path.exists():
        ogr.GetDriverByName("GPKG").DeleteDataSource(str(path))
    dataset = ogr.GetDriverByName("GPKG").CreateDataSource(str(path))
    layer = dataset.CreateLayer("protection_control_zones", srs=srs, geom_type=ogr.wkbMultiPolygon,
                                options=["GEOMETRY_NAME=geom", "SPATIAL_INDEX=YES"])
    for name, field_type in (("zone", ogr.OFTString), ("label_cn", ogr.OFTString),
                             ("min_m", ogr.OFTReal), ("max_m", ogr.OFTReal)):
        layer.CreateField(ogr.FieldDefn(name, field_type))
    distances = {
        "protection_0_50m": (0.0, 50.0),
        "rural_control_50_550m": (50.0, 550.0),
    }
    for name, geometry in geometries.items():
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("zone", name)
        feature.SetField("label_cn", ZONE_LABELS[name])
        feature.SetField("min_m", distances[name][0])
        feature.SetField("max_m", distances[name][1])
        geom = geometry.Clone()
        if ogr.GT_Flatten(geom.GetGeometryType()) == ogr.wkbPolygon:
            multi = ogr.Geometry(ogr.wkbMultiPolygon)
            multi.AddGeometry(geom)
            geom = multi
        feature.SetGeometry(geom)
        layer.CreateFeature(feature)
    layer.SyncToDisk()
    dataset = None


def plot_results(zone_rows: list[dict], segment_rows: list[dict], output: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    zones = ["protection_0_50m", "rural_control_50_550m"]
    indicators = ["drainage", "environment", "insar", "landslide", "composite"]
    colors = ["#c84b31", "#2f6fa3"]

    figure, (axis_bar, axis_line) = plt.subplots(2, 1, figsize=(12, 10.2), dpi=190,
                                                gridspec_kw={"height_ratios": [1.0, 1.15]})
    x = np.arange(len(indicators))
    width = 0.34
    for offset, (zone, color) in enumerate(zip(zones, colors)):
        values = []
        for indicator in indicators:
            row = next(item for item in zone_rows if item["zone"] == zone and item["indicator"] == indicator)
            values.append(row["mean_score"])
        axis_bar.bar(x + (offset - 0.5) * width, values, width=width, color=color, label=ZONE_LABELS[zone])
        for xpos, value in zip(x + (offset - 0.5) * width, values):
            axis_bar.text(xpos, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axis_bar.set_xticks(x)
    axis_bar.set_xticklabels([INDICATOR_LABELS[name] for name in indicators])
    axis_bar.set_ylabel("统一标定后的平均影响强度")
    axis_bar.set_title("A. 保护范围与农村/郊野建设控制地带的影响组成")
    axis_bar.legend(frameon=False, ncol=2, fontsize=9)
    axis_bar.grid(axis="y", alpha=0.2)

    ordered = sorted(segment_rows, key=lambda row: row["start_m"])
    chainage = np.asarray([(row["start_m"] + row["end_m"]) / 2000.0 for row in ordered])
    for zone, color in zip(zones, colors):
        values = np.asarray([row[f"{zone}_composite"] for row in ordered]) * 100.0
        axis_line.plot(chainage, values, color=color, linewidth=1.6, label=ZONE_LABELS[zone])
    axis_line.set_xlabel("沿长城里程（km）")
    axis_line.set_ylabel("综合影响得分")
    axis_line.set_title("B. 156个50米分段的综合影响剖面")
    axis_line.legend(frameon=False, ncol=2, fontsize=9)
    axis_line.grid(alpha=0.2)

    figure.suptitle("广武长城保护范围与建设控制地带综合影响比较", fontsize=17, fontweight="bold")
    figure.text(0.5, 0.012,
                "注：距离从现有长城线位起算，近似墙体外缘；法定边界仍以主管部门公布的文字与图纸为准。",
                ha="center", fontsize=9.5, color="#555555")
    figure.tight_layout(rect=[0.03, 0.04, 0.98, 0.95])
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    gdal.UseExceptions()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    dem_ds = gdal.Open(str(DEM_PATH))
    if dem_ds is None:
        raise FileNotFoundError(DEM_PATH)
    grid = raster_grid(dem_ds)
    target_srs = make_target_srs(grid["projection"])
    dem = dem_ds.GetRasterBand(1).ReadAsArray().astype(float)
    nodata = dem_ds.GetRasterBand(1).GetNoDataValue()
    if nodata is not None:
        dem[np.isclose(dem, nodata)] = np.nan

    source_ds = gdal.OpenEx(str(SEGMENTS), gdal.OF_VECTOR)
    source_layer = source_ds.GetLayerByName("wall_segment_risk")
    wall = build_wall(source_layer)
    zones = make_zone_geometries(wall)
    domain_geometry = wall.Buffer(RURAL_OUTER_M, 32)
    domain_mask = rasterize_geometry(domain_geometry, grid) & np.isfinite(dem)
    zone_masks = {name: rasterize_geometry(geometry, grid) & domain_mask for name, geometry in zones.items()}

    raw = {
        "slope": aligned_array(FACTOR_PATHS["slope_deg"], dem_ds)[0],
        "relief": aligned_array(FACTOR_PATHS["relief_300m"], dem_ds)[0],
        "twi": aligned_array(FACTOR_PATHS["twi"], dem_ds)[0],
        "drainage": aligned_array(FACTOR_PATHS["distance_to_drainage_m"], dem_ds)[0],
    }
    ndvi_cube = aligned_array(FACTOR_PATHS["ndvi"], dem_ds)
    ndvi_ds = gdal.Open(str(FACTOR_PATHS["ndvi"]))
    descriptions = [ndvi_ds.GetRasterBand(i + 1).GetDescription() for i in range(ndvi_ds.RasterCount)]
    growing = [i for i, text in enumerate(descriptions) if any(f"{month:02d}" in text[-2:] for month in (5, 6, 7, 8, 9))]
    if not growing:
        growing = list(range(ndvi_cube.shape[0]))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        raw["ndvi"] = np.nanmedian(ndvi_cube[growing], axis=0)
    ndvi_ds = None

    scaling = {}
    indicators = {}
    indicators["slope"], scaling["slope"] = robust_scale_in_mask(raw["slope"], domain_mask)
    indicators["relief"], scaling["relief"] = robust_scale_in_mask(raw["relief"], domain_mask)
    indicators["twi"], scaling["twi"] = robust_scale_in_mask(raw["twi"], domain_mask)
    indicators["drainage"], scaling["drainage"] = robust_scale_in_mask(raw["drainage"], domain_mask, invert=True)
    indicators["ndvi"], scaling["ndvi"] = robust_scale_in_mask(raw["ndvi"], domain_mask, invert=True)
    indicators["environment"] = sum(ENVIRONMENT_WEIGHTS[name] * indicators[name] for name in ENVIRONMENT_WEIGHTS).astype(np.float32)

    rows, cols, coordinates = pixel_centres(domain_mask, grid["gt"])
    tracks = {name: load_insar_track(path, target_srs) for name, path in INSAR_TRACKS.items()}
    indicators["insar"], insar_stats = local_insar_score(tracks, coordinates, dem.shape, rows, cols)
    slides = {name: load_union(path, target_srs) for name, path in LANDSLIDES.items()}
    indicators["landslide"] = landslide_score(slides, coordinates, dem.shape, rows, cols)
    indicators["composite"] = (
        COMPONENT_WEIGHTS["environment"] * indicators["environment"]
        + COMPONENT_WEIGHTS["insar"] * indicators["insar"]
        + COMPONENT_WEIGHTS["landslide"] * indicators["landslide"]
    ).astype(np.float32)
    for array in indicators.values():
        array[~domain_mask] = np.nan

    zone_rows = []
    for zone, mask in zone_masks.items():
        for indicator, array in indicators.items():
            values = finite_values(array, mask)
            zone_rows.append({
                "zone": zone,
                "zone_cn": ZONE_LABELS[zone],
                "indicator": indicator,
                "indicator_cn": INDICATOR_LABELS[indicator],
                "pixel_count": int(values.size),
                "area_km2": float(values.size * PIXEL_SIZE_M**2 / 1_000_000),
                "mean_score": float(np.mean(values)),
                "median_score": float(np.median(values)),
                "p90_score": float(np.percentile(values, 90)),
                "high_score_share_ge_0_6": float(np.mean(values >= 0.6)),
            })

    segment_rows = []
    source_layer.ResetReading()
    for feature in source_layer:
        line = feature.GetGeometryRef().Clone()
        segment_zones = make_zone_geometries(line)
        record = {"segment_id": int(feature["segment_id"]), "start_m": float(feature["start_m"]), "end_m": float(feature["end_m"])}
        for zone, geometry in segment_zones.items():
            mask = rasterize_geometry(geometry, grid) & domain_mask
            for indicator, array in indicators.items():
                record[f"{zone}_{indicator}"] = mean_or_nan(array, mask)
        segment_rows.append(record)
    segment_rows.sort(key=lambda row: row["start_m"])

    split_indices = np.array_split(np.arange(len(segment_rows)), SPATIAL_BLOCKS)
    block_ids = np.empty(len(segment_rows), dtype=int)
    for block, indices in enumerate(split_indices):
        block_ids[indices] = block

    comparisons = []
    for control_zone in ("rural_control_50_550m",):
        for indicator in indicators:
            protection_values = np.asarray([row[f"protection_0_50m_{indicator}"] for row in segment_rows])
            control_values = np.asarray([row[f"{control_zone}_{indicator}"] for row in segment_rows])
            protection_mean = next(row["mean_score"] for row in zone_rows if row["zone"] == "protection_0_50m" and row["indicator"] == indicator)
            control_mean = next(row["mean_score"] for row in zone_rows if row["zone"] == control_zone and row["indicator"] == indicator)
            ratio = protection_mean / control_mean if control_mean > 0 else float("nan")
            ci_low, ci_high, probability = bootstrap_ratio_ci(protection_values, control_values, block_ids, rng)
            comparisons.append({
                "comparison": f"protection_0_50m_vs_{control_zone}",
                "control_zone": control_zone,
                "control_zone_cn": ZONE_LABELS[control_zone],
                "indicator": indicator,
                "indicator_cn": INDICATOR_LABELS[indicator],
                "protection_mean": protection_mean,
                "control_mean": control_mean,
                "protection_control_ratio": ratio,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "probability_ratio_gt_1": probability,
                "interpretation": interpretation(ratio, ci_low, ci_high),
            })

    summary_path = OUT_DIR / "zone_indicator_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(zone_rows[0].keys()))
        writer.writeheader(); writer.writerows(zone_rows)
    comparison_path = OUT_DIR / "protection_vs_control_comparison.csv"
    with comparison_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0].keys()))
        writer.writeheader(); writer.writerows(comparisons)
    segment_path = OUT_DIR / "segment_zone_composite_scores.csv"
    segment_fields = ["segment_id", "start_m", "end_m"] + [f"{zone}_composite" for zone in zones]
    with segment_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=segment_fields)
        writer.writeheader()
        for row in segment_rows:
            writer.writerow({key: row[key] for key in segment_fields})

    zone_path = OUT_DIR / "protection_control_zones.gpkg"
    raster_path = OUT_DIR / "protection_control_components_10m.tif"
    figure_path = OUT_DIR / "广武长城_保护范围与建设控制地带综合影响比较.png"
    write_zones(zone_path, zones, target_srs)
    write_component_raster(raster_path, [indicators[name] for name in indicators],
                           [INDICATOR_LABELS[name] for name in indicators], dem_ds)
    plot_results(zone_rows, segment_rows, figure_path)

    key_row = next(item for item in comparisons if item["indicator"] == "composite")
    key = {"rural_control_50_550m": key_row}
    metadata = {
        "distance_origin": "existing Great Wall line used as proxy for wall-body outer edge",
        "zones": ZONE_LABELS,
        "normalization": "common 5th-95th percentile scaling over 0-550 m",
        "segment_count": len(segment_rows),
        "spatial_bootstrap_blocks": SPATIAL_BLOCKS,
        "scaling_percentiles": scaling,
        "component_weights": COMPONENT_WEIGHTS,
        "environment_weights": ENVIRONMENT_WEIGHTS,
        "insar_statistics": insar_stats,
        "key_composite_comparisons": key,
        "limitations": [
            "line geometry is not the surveyed wall-body outer edge",
            "the analysis assumes the Guangwu Great Wall section is in a rural/outlying area",
            "published legal boundary text and maps take precedence over distance buffers",
            "exploratory association, not causal effect",
            "known landslide inventory contains two mapped zones",
        ],
        "outputs": {"zone_summary": str(summary_path), "comparisons": str(comparison_path),
                    "segment_scores": str(segment_path), "zones": str(zone_path),
                    "components": str(raster_path), "figure": str(figure_path)},
    }
    (OUT_DIR / "experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"key_composite_comparisons": key, "outputs": metadata["outputs"]}, ensure_ascii=False, indent=2))
    source_ds = None
    dem_ds = None


if __name__ == "__main__":
    main()
