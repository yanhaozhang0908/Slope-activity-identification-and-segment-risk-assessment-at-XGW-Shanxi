"""Clip the thesis slope-unit map and its positive markers to the 1 km core."""

import os
from pathlib import Path

os.add_dll_directory(r"C:\Program Files\QGIS 3.40.10\bin")
os.environ["GDAL_DATA"] = r"C:\Program Files\QGIS 3.40.10\apps\gdal\share\gdal"
os.environ["PROJ_DATA"] = r"C:\Program Files\QGIS 3.40.10\share\proj"

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"D:\project\shuozhouinflucefactor")
OUT = ROOT / "outputs" / "susceptibility_unassessed_overlay_20260917"
FULL = ROOT / "process" / "factor_domain" / "slope_units_full_with_core_relation.gpkg"
CORE = ROOT / "process" / "factor_domain" / "analysis_domains.gpkg"
RESULTS = ROOT / "process" / "mapping_unit_model_optimization" / "slope_unit_optimized_ensemble.csv"
UNITS_OUT = OUT / "广武长城_1km缓冲区裁剪坡单元_含未评价.gpkg"
POINTS_OUT = OUT / "广武长城_1km缓冲区内活动正样本图示点.gpkg"
TABLE_OUT = OUT / "广武长城_1km缓冲区裁剪坡单元_属性.csv"
PREVIEW_OUT = OUT / "广武长城_1km缓冲区裁剪坡单元_预览.png"

full = gpd.read_file(FULL, layer="slope_units").to_crs(32649)
core = gpd.read_file(CORE, layer="core_1km").to_crs(32649)
result = pd.read_csv(RESULTS)[["SU_ID", "OPT_score", "OPT_class", "LABEL"]]
if len(full) != 270 or full["SU_ID"].nunique() != 270 or result["SU_ID"].nunique() != 270:
    raise RuntimeError("Source slope-unit counts differ from the thesis experiment")

joined = full[["SU_ID", "geometry"]].merge(result, on="SU_ID", how="left", validate="one_to_one")
core_shape = core.geometry.union_all()
clipped = gpd.clip(joined, core_shape, keep_geom_type=True)
clipped = clipped[~clipped.geometry.is_empty].copy().sort_values("SU_ID")
clipped["class_code"] = clipped["OPT_class"].fillna(0).astype(int)
names = {0: "未评价", 1: "低易发", 2: "较低易发", 3: "中易发", 4: "较高易发", 5: "高易发"}
clipped["class_name"] = clipped["class_code"].map(names)
clipped["area_in_1km_km2"] = clipped.geometry.area / 1_000_000
clipped["positive_label"] = clipped["LABEL"].fillna(-1).astype(int)

if len(clipped) != 270 or clipped["SU_ID"].nunique() != 270:
    raise RuntimeError(f"Clipped slope units: {len(clipped)}; expected 270")
if (clipped["class_code"] == 0).sum() != 58 or (clipped["positive_label"] == 1).sum() != 20:
    raise RuntimeError("Class or positive-label counts changed during clipping")

coverage = clipped.geometry.union_all()
gap_m2 = core_shape.difference(coverage).area
overlap_m2 = clipped.geometry.area.sum() - coverage.area
if gap_m2 > 100 or overlap_m2 > 100:
    raise RuntimeError(f"Clipped unit topology mismatch: gap={gap_m2:.2f} m2, overlap={overlap_m2:.2f} m2")

clipped[["SU_ID", "OPT_score", "class_code", "class_name", "positive_label", "area_in_1km_km2", "geometry"]].to_file(
    UNITS_OUT, layer="slope_units_1km", driver="GPKG"
)
clipped.drop(columns="geometry").to_csv(TABLE_OUT, index=False, encoding="utf-8-sig")

positive = clipped.loc[clipped["positive_label"] == 1, ["SU_ID", "class_code", "geometry"]].copy()
positive.geometry = positive.geometry.representative_point()
positive.to_file(POINTS_OUT, layer="positive_markers_1km", driver="GPKG")

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"], "axes.unicode_minus": False})
colors = {0: "#d9d9d9", 1: "#1a9850", 2: "#91cf60", 3: "#fee08b", 4: "#fc8d59", 5: "#d73027"}
fig, ax = plt.subplots(figsize=(7.0, 9.0))
for code, color in colors.items():
    subset = clipped[clipped["class_code"] == code]
    subset.plot(ax=ax, color=color, edgecolor="#666666" if code == 0 else "#dddddd", linewidth=0.45 if code == 0 else 0.15)
core.boundary.plot(ax=ax, color="#225ea8", linewidth=1.1, linestyle="--")
positive.plot(ax=ax, facecolor="none", edgecolor="#00aaaa", markersize=24, linewidth=0.9)
ax.set_aspect("equal")
ax.set_title("1 km缓冲区内坡单元（灰色为未评价）")
ax.set_xlabel("东坐标（m，UTM 49N）")
ax.set_ylabel("北坐标（m，UTM 49N）")
fig.tight_layout()
fig.savefig(PREVIEW_OUT, dpi=240)
plt.close(fig)

print(f"units={len(clipped)}, evaluated={(clipped['class_code'] > 0).sum()}, unassessed={(clipped['class_code'] == 0).sum()}")
print(f"positive_points={len(positive)}, core_km2={core_shape.area / 1e6:.6f}, units_km2={clipped.geometry.area.sum() / 1e6:.6f}")
print(f"gap_m2={gap_m2:.4f}, overlap_m2={overlap_m2:.4f}")
print(UNITS_OUT)
print(POINTS_OUT)
