from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from osgeo import ogr


ROOT = Path(r"D:\project\shuozhouinflucefactor")
SOURCE = (
    ROOT
    / "process"
    / "wall_segment_risk_from_pixels"
    / "guangwu_wall_50m_segment_risk_from_10m_pixels.gpkg"
)
SOURCE_LAYER = "wall_segment_pixel_aggregated_risk"
OUTPUT_DIR = ROOT / "process" / "wall_segment_risk_from_pixels"
OUTPUT = OUTPUT_DIR / "guangwu_wall_10m_precomputed_interpolated_display.gpkg"
OUTPUT_LAYER = "wall_display_10m_precomputed"
STYLE = OUTPUT.with_suffix(".qml")
README = OUTPUT_DIR / "README_10m预计算显示线.md"
TARGET_LENGTH_M = 10.0
STYLE_CLASSES = 40


def read_segments() -> tuple[list[dict], object]:
    source = ogr.Open(str(SOURCE))
    if source is None:
        raise FileNotFoundError(SOURCE)
    layer = source.GetLayerByName(SOURCE_LAYER)
    if layer is None:
        raise RuntimeError(f"Layer not found: {SOURCE_LAYER}")
    srs = layer.GetSpatialRef().Clone()
    rows: list[dict] = []
    for feature in layer:
        rows.append(
            {
                "segment_id": int(feature.GetField("segment_id")),
                "start_m": float(feature.GetField("start_m")),
                "end_m": float(feature.GetField("end_m")),
                "risk_score": float(feature.GetField("risk_score")),
                "risk_class": str(feature.GetField("risk_class")),
                "geometry": feature.GetGeometryRef().Clone(),
            }
        )
    source = None
    rows.sort(key=lambda item: (item["start_m"], item["segment_id"]))
    return rows, srs


def geometry_coordinates(geometry: ogr.Geometry) -> np.ndarray:
    if geometry.GetGeometryName().upper() != "LINESTRING":
        raise ValueError(f"Expected LINESTRING, got {geometry.GetGeometryName()}")
    return np.asarray(
        [(geometry.GetX(index), geometry.GetY(index)) for index in range(geometry.GetPointCount())],
        dtype=float,
    )


def point_at(coords: np.ndarray, cumulative: np.ndarray, distance: float) -> np.ndarray:
    distance = float(np.clip(distance, 0.0, cumulative[-1]))
    index = int(np.searchsorted(cumulative, distance, side="right") - 1)
    index = min(max(index, 0), len(coords) - 2)
    span = cumulative[index + 1] - cumulative[index]
    if span <= 0:
        return coords[index].copy()
    fraction = (distance - cumulative[index]) / span
    return coords[index] + fraction * (coords[index + 1] - coords[index])


def line_substring(geometry: ogr.Geometry, start_distance: float, end_distance: float) -> ogr.Geometry:
    coords = geometry_coordinates(geometry)
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
    )
    if cumulative[-1] <= 0:
        raise ValueError("Zero-length source segment")
    start_distance = float(np.clip(start_distance, 0.0, cumulative[-1]))
    end_distance = float(np.clip(end_distance, start_distance, cumulative[-1]))
    points = [point_at(coords, cumulative, start_distance)]
    for index in range(1, len(coords) - 1):
        if start_distance < cumulative[index] < end_distance:
            points.append(coords[index])
    points.append(point_at(coords, cumulative, end_distance))

    cleaned = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - cleaned[-1]) > 1e-9:
            cleaned.append(point)
    if len(cleaned) == 1:
        cleaned.append(cleaned[0].copy())

    output = ogr.Geometry(ogr.wkbLineString)
    for x, y in cleaned:
        output.AddPoint_2D(float(x), float(y))
    return output


def display_class(score: float) -> str:
    if score < 20.0:
        return "Low"
    if score < 40.0:
        return "Moderate-low"
    if score < 60.0:
        return "Moderate"
    if score < 80.0:
        return "High"
    return "Very high"


def interpolate_score(chainage: float, centers: np.ndarray, scores: np.ndarray) -> float:
    return float(np.interp(chainage, centers, scores))


def create_output(segments: list[dict], srs: object) -> tuple[int, float, float]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    driver = ogr.GetDriverByName("GPKG")
    if OUTPUT.exists():
        driver.DeleteDataSource(str(OUTPUT))
    dataset = driver.CreateDataSource(str(OUTPUT))
    layer = dataset.CreateLayer(
        OUTPUT_LAYER,
        srs=srs,
        geom_type=ogr.wkbLineString,
        options=["SPATIAL_INDEX=YES"],
    )
    fields = [
        ("display_id", ogr.OFTInteger),
        ("segment_id", ogr.OFTInteger),
        ("sub_id", ogr.OFTInteger),
        ("start_m", ogr.OFTReal),
        ("end_m", ogr.OFTReal),
        ("length_m", ogr.OFTReal),
        ("risk_score", ogr.OFTReal),
        ("risk_start", ogr.OFTReal),
        ("risk_end", ogr.OFTReal),
        ("display_risk", ogr.OFTReal),
        ("risk_class", ogr.OFTString),
        ("display_cls", ogr.OFTString),
    ]
    for name, field_type in fields:
        definition = ogr.FieldDefn(name, field_type)
        if field_type == ogr.OFTReal:
            definition.SetWidth(20)
            definition.SetPrecision(6)
        elif field_type == ogr.OFTString:
            definition.SetWidth(24)
        layer.CreateField(definition)

    centers = np.asarray([(item["start_m"] + item["end_m"]) / 2.0 for item in segments])
    scores = np.asarray([item["risk_score"] for item in segments])
    display_id = 0
    display_values: list[float] = []
    for item in segments:
        chainage_length = item["end_m"] - item["start_m"]
        piece_count = max(1, int(math.ceil(chainage_length / TARGET_LENGTH_M - 1e-9)))
        geometry_length = item["geometry"].Length()
        for sub_index in range(piece_count):
            display_id += 1
            fraction_start = sub_index / piece_count
            fraction_end = (sub_index + 1) / piece_count
            start_m = item["start_m"] + fraction_start * chainage_length
            end_m = item["start_m"] + fraction_end * chainage_length
            midpoint_m = (start_m + end_m) / 2.0
            risk_start = interpolate_score(start_m, centers, scores)
            risk_end = interpolate_score(end_m, centers, scores)
            display_risk = interpolate_score(midpoint_m, centers, scores)
            sub_geometry = line_substring(
                item["geometry"],
                fraction_start * geometry_length,
                fraction_end * geometry_length,
            )

            feature = ogr.Feature(layer.GetLayerDefn())
            values = {
                "display_id": display_id,
                "segment_id": item["segment_id"],
                "sub_id": sub_index + 1,
                "start_m": start_m,
                "end_m": end_m,
                "length_m": end_m - start_m,
                "risk_score": item["risk_score"],
                "risk_start": risk_start,
                "risk_end": risk_end,
                "display_risk": display_risk,
                "risk_class": item["risk_class"],
                "display_cls": display_class(display_risk),
            }
            for name, value in values.items():
                feature.SetField(name, value)
            feature.SetGeometry(sub_geometry)
            if layer.CreateFeature(feature) != ogr.OGRERR_NONE:
                raise RuntimeError(f"Failed to write display feature {display_id}")
            feature = None
            display_values.append(display_risk)

    dataset = None
    return display_id, float(min(display_values)), float(max(display_values))


def interpolate_color(position: float) -> tuple[int, int, int]:
    stops = [
        (0.00, (91, 181, 82)),
        (0.30, (177, 215, 91)),
        (0.58, (250, 221, 102)),
        (0.78, (238, 124, 46)),
        (1.00, (202, 59, 51)),
    ]
    for (left_x, left_color), (right_x, right_color) in zip(stops[:-1], stops[1:]):
        if position <= right_x:
            local = (position - left_x) / (right_x - left_x)
            return tuple(
                int(round(left_color[index] + local * (right_color[index] - left_color[index])))
                for index in range(3)
            )
    return stops[-1][1]


def write_style(minimum: float, maximum: float) -> None:
    ranges = []
    symbols = []
    for index in range(STYLE_CLASSES):
        lower = minimum + (maximum - minimum) * index / STYLE_CLASSES
        upper = minimum + (maximum - minimum) * (index + 1) / STYLE_CLASSES
        red, green, blue = interpolate_color((index + 0.5) / STYLE_CLASSES)
        ranges.append(
            f'      <range lower="{lower:.9f}" upper="{upper:.9f}" '
            f'symbol="{index}" label="{lower:.1f}–{upper:.1f}" render="true"/>'
        )
        symbols.append(
            f'      <symbol name="{index}" type="line" alpha="1">'
            f'<layer class="SimpleLine"><Option type="Map">'
            f'<Option name="line_color" type="QString" value="{red},{green},{blue},255"/>'
            f'<Option name="line_width" type="QString" value="2.2"/>'
            f'<Option name="line_width_unit" type="QString" value="MM"/>'
            f'<Option name="capstyle" type="QString" value="round"/>'
            f'<Option name="joinstyle" type="QString" value="round"/>'
            f'</Option></layer></symbol>'
        )
    xml = "\n".join(
        [
            "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>",
            '<qgis version="3.34" styleCategories="Symbology">',
            '  <renderer-v2 type="graduatedSymbol" attr="display_risk" graduatedMethod="GraduatedColor" symbollevels="0" enableorderby="0">',
            "    <ranges>",
            *ranges,
            "    </ranges>",
            "    <symbols>",
            *symbols,
            "    </symbols>",
            "  </renderer-v2>",
            "  <layerGeometryType>1</layerGeometryType>",
            "</qgis>",
        ]
    )
    STYLE.write_text(xml, encoding="utf-8")


def write_readme(feature_count: int, minimum: float, maximum: float) -> None:
    README.write_text(
        f"""# 10 m预计算风险显示线

- 源数据：`{SOURCE.name}` 的 `{SOURCE_LAYER}` 图层。
- 输出图层：`{OUTPUT_LAYER}`，共 {feature_count} 条约10 m子段。
- `risk_score`：所属原始50 m分段得分，不作修改。
- `display_risk`：根据相邻50 m分段中心得分沿里程线性插值后的子段中心值。
- `risk_start`、`risk_end`：每条显示子段两端的预计算插值值。
- 显示值范围：{minimum:.6f}–{maximum:.6f}。
- 本图层仅用于制图显示；统计分析、风险等级汇总和结构失稳关联仍使用原始50 m分段。

## QGIS加载

1. 加载 `{OUTPUT.name}` 中的 `{OUTPUT_LAYER}`。
2. 若样式未自动加载，在“图层属性—符号系统—样式—加载样式”中选择 `{STYLE.name}`。
3. 保持“简单线”渲染，不要再选择“内插线”。
""",
        encoding="utf-8",
    )


def main() -> None:
    segments, srs = read_segments()
    feature_count, minimum, maximum = create_output(segments, srs)
    write_style(minimum, maximum)
    write_readme(feature_count, minimum, maximum)
    print(f"output={OUTPUT}")
    print(f"layer={OUTPUT_LAYER}")
    print(f"features={feature_count}")
    print(f"display_risk_min={minimum:.6f}")
    print(f"display_risk_max={maximum:.6f}")
    print(f"style={STYLE}")


if __name__ == "__main__":
    main()
