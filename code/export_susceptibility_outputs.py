from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from osgeo import gdal


ROOT = Path(r"D:\project\shuozhouinflucefactor")
BASE_GPKG = ROOT / "process" / "insar" / "slope_unit_insar_stats.gpkg"
RESULT_CSV = (
    ROOT / "process" / "instability_susceptibility" / "instability_results.csv"
)
CORE_GPKG = ROOT / "process" / "factor_domain" / "analysis_domains.gpkg"
REFERENCE_RASTER = (
    ROOT / "data" / "ndvi" / "sentinel2_monthly" / "ndvi_202211_10m.tif"
)
OUTPUT_DIR = ROOT / "process" / "instability_susceptibility"
OUTPUT_GPKG = OUTPUT_DIR / "instability_models.gpkg"
OUTPUT_META = OUTPUT_DIR / "spatial_output_metadata.json"

RESULT_FIELDS = [
    "SU_ID",
    "POS_label",
    "POS_event",
    "IV_score",
    "FIRTH_prob",
    "RF_prob",
    "RF_repeat_mean",
    "RF_repeat_std",
    "ENS_score",
    "INST_class",
    "INST_name",
    "PRIORITY",
    "PRIORITY_name",
]


def rasterize_field(
    field: str,
    filename: str,
    gdal_type: int,
    nodata: float,
) -> None:
    reference = gdal.Open(str(REFERENCE_RASTER))
    target = gdal.GetDriverByName("GTiff").Create(
        str(OUTPUT_DIR / filename),
        reference.RasterXSize,
        reference.RasterYSize,
        1,
        gdal_type,
        options=["COMPRESS=DEFLATE", "TILED=YES"],
    )
    target.SetGeoTransform(reference.GetGeoTransform())
    target.SetProjection(reference.GetProjection())
    band = target.GetRasterBand(1)
    band.Fill(nodata)
    band.SetNoDataValue(nodata)
    vector = gdal.OpenEx(str(OUTPUT_GPKG), gdal.OF_VECTOR)
    layer = vector.GetLayerByName("core_1km_results")
    result = gdal.RasterizeLayer(
        target,
        [1],
        layer,
        options=[f"ATTRIBUTE={field}", "ALL_TOUCHED=FALSE"],
    )
    if result != 0:
        raise RuntimeError(f"Rasterizing {field} failed with code {result}")
    band.SetDescription(field)
    target.FlushCache()
    vector = None
    target = None


def make_preview(full: gpd.GeoDataFrame, core: gpd.GeoDataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 12), dpi=160)
    continuous = [
        ("IV_score", "Information value score", "viridis"),
        ("FIRTH_prob", "Firth probability", "viridis"),
        ("RF_repeat_mean", "Random-forest mean probability", "viridis"),
    ]
    for axis, (field, title, cmap) in zip(axes[0], continuous):
        core.plot(
            column=field,
            cmap=cmap,
            legend=True,
            edgecolor="0.55",
            linewidth=0.2,
            ax=axis,
        )
        full.loc[full["POS_label"] == 1].boundary.plot(
            ax=axis, color="red", linewidth=1.3
        )
        axis.set_title(title)
        axis.set_axis_off()

    class_colors = ["#2C7BB6", "#ABD9E9", "#FFFFBF", "#FDAE61", "#D7191C"]
    activity_colors = ["#BDBDBD", "#2CA25F", "#FEC44F", "#DE2D26"]
    priority_colors = ["#4575B4", "#91BFDB", "#FFFFBF", "#FC8D59", "#D73027"]
    panels = [
        ("INST_class", "Static instability susceptibility", class_colors, 1, 5),
        ("INSAR_act", "InSAR activity class", activity_colors, 0, 3),
        ("PRIORITY", "Monitoring priority", priority_colors, 1, 5),
    ]
    for axis, (field, title, colors, minimum, maximum) in zip(axes[1], panels):
        cmap = ListedColormap(colors)
        boundaries = np.arange(minimum - 0.5, maximum + 1.5, 1)
        norm = BoundaryNorm(boundaries, cmap.N)
        core.plot(
            column=field,
            cmap=cmap,
            norm=norm,
            legend=True,
            edgecolor="0.55",
            linewidth=0.2,
            ax=axis,
        )
        full.loc[full["POS_label"] == 1].boundary.plot(
            ax=axis, color="black", linewidth=1.2
        )
        axis.set_title(title)
        axis.set_axis_off()
    fig.suptitle(
        "Slope-unit instability susceptibility and InSAR monitoring priority\n"
        "(outlines mark three mapped instability-event groups)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(
        OUTPUT_DIR / "instability_model_maps_preview.png",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def write_report(results: gpd.GeoDataFrame) -> None:
    model = results[results["model_keep"] == 1]
    positives = model[model["POS_label"] == 1]
    priority_counts = model["PRIORITY_name"].value_counts()
    class_counts = model["INST_name"].value_counts()
    event_counts = positives["POS_event"].value_counts()
    report = f"""# 坡单元不稳定性易发性模型结果

## 样本

- 建模坡单元：{len(model)} 个。
- 不稳定正样本坡单元：{len(positives)} 个，来自 3 个独立事件组。
- 各事件组坡单元数：

{event_counts.to_string()}

- InSAR 稳定负样本：{int(model["INSAR_stable"].sum())} 个。
- “非滑坡区域1”作为不稳定事件 U1 进入正样本，不再仅作为负样本排除区。

## 模型

1. 信息量模型：解释各因子分级与不稳定坡体分布的相对关系。
2. Firth 惩罚逻辑回归：小样本主模型。
3. 随机森林：非线性探索性对照。
4. 三模型百分位等权集成：生成静态易发性五级分区。
5. InSAR 活动性叠加：生成监测优先级，不把 InSAR 当作普通静态因子。

## 五级易发性坡单元数

{class_counts.to_string()}

## 监测优先级坡单元数

{priority_counts.to_string()}

## 解释限制

- 只有 3 个独立不稳定事件；同一事件内的正样本坡单元存在明显空间相关性。
- 训练 AUC 只能描述样本拟合，不能作为独立精度。
- 采用 3 折逐事件空间留出验证，但独立事件仍然极少。
- 距道路和距断层可能反映三个事件的空间位置，不能直接解释为因果作用。
- 随机森林仅作为探索性对照，不能宣称具有稳定区域泛化能力。
- 结果应表述为“活动斜坡/坡体不稳定性的相对易发性与监测优先级”，避免表述为经过充分验证的滑坡发生概率。
- U1 未通过原 C 指数滑坡识别，本模型将其视为已确认形变不稳定区，而不是确认滑坡。
"""
    (OUTPUT_DIR / "README_模型结果说明.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = gpd.read_file(BASE_GPKG, layer="slope_units")
    results = pd.read_csv(RESULT_CSV)
    merged = base.merge(results[RESULT_FIELDS], on="SU_ID", how="left")
    core_area = gpd.read_file(CORE_GPKG, layer="core_1km")
    if core_area.crs != merged.crs:
        core_area = core_area.to_crs(merged.crs)
    core_geometry = core_area.geometry.union_all()
    core = merged[merged.geometry.intersects(core_geometry)].copy()
    core.geometry = core.geometry.intersection(core_geometry)
    core = core[core.geometry.notna() & ~core.geometry.is_empty]

    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()
    merged.to_file(
        OUTPUT_GPKG, layer="full_slope_unit_results", driver="GPKG"
    )
    core.to_file(
        OUTPUT_GPKG, layer="core_1km_results", driver="GPKG", mode="a"
    )

    rasterize_field(
        "IV_score", "information_value_score_core_10m.tif", gdal.GDT_Float32, -9999
    )
    rasterize_field(
        "FIRTH_prob", "firth_probability_core_10m.tif", gdal.GDT_Float32, -9999
    )
    rasterize_field(
        "RF_repeat_mean",
        "random_forest_probability_core_10m.tif",
        gdal.GDT_Float32,
        -9999,
    )
    rasterize_field(
        "INST_class",
        "ensemble_instability_class_core_10m.tif",
        gdal.GDT_Byte,
        0,
    )
    rasterize_field(
        "INSAR_act", "insar_activity_class_core_10m.tif", gdal.GDT_Byte, 255
    )
    rasterize_field(
        "PRIORITY", "monitoring_priority_core_10m.tif", gdal.GDT_Byte, 0
    )
    make_preview(merged, core)
    write_report(merged)

    metadata = {
        "full_layer": f"{OUTPUT_GPKG}|layer=full_slope_unit_results",
        "core_layer": f"{OUTPUT_GPKG}|layer=core_1km_results",
        "full_feature_count": len(merged),
        "core_feature_count": len(core),
        "core_raster_grid": {
            "reference": str(REFERENCE_RASTER),
            "crs": "EPSG:32649",
            "pixel_size_m": 10,
            "shape": [685, 471],
        },
        "raster_outputs": [
            "information_value_score_core_10m.tif",
            "firth_probability_core_10m.tif",
            "random_forest_probability_core_10m.tif",
            "ensemble_instability_class_core_10m.tif",
            "insar_activity_class_core_10m.tif",
            "monitoring_priority_core_10m.tif",
        ],
    }
    OUTPUT_META.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
