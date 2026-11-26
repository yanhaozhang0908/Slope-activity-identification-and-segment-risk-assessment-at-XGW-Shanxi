from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


ROOT = Path(r"D:\project\shuozhouinflucefactor")
CORE_GPKG = ROOT / "process" / "dem" / "analysis_boundaries.gpkg"
FULL_SLOPE_GPKG = ROOT / "process" / "dem" / "slope_units_main_full.gpkg"
OUTPUT_DIR = ROOT / "process" / "factor_domain"
OUTPUT_GPKG = OUTPUT_DIR / "analysis_domains.gpkg"
OUTPUT_UNITS = OUTPUT_DIR / "slope_units_full_with_core_relation.gpkg"
OUTPUT_CSV = OUTPUT_DIR / "slope_unit_core_relation.csv"
OUTPUT_JSON = OUTPUT_DIR / "factor_domain_metadata.json"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    core = gpd.read_file(CORE_GPKG, layer="study_area")
    units = gpd.read_file(FULL_SLOPE_GPKG, layer="slope_units")
    if core.crs != units.crs:
        core = core.to_crs(units.crs)

    core_geometry = core.geometry.union_all()
    domain_geometry = units.geometry.union_all()
    domain = gpd.GeoDataFrame(
        {
            "name": ["完整坡单元因子计算区"],
            "unit_count": [len(units)],
            "area_km2": [domain_geometry.area / 1_000_000.0],
        },
        geometry=[domain_geometry],
        crs=units.crs,
    )
    core_output = gpd.GeoDataFrame(
        {
            "name": ["长城周边1 km核心研究区"],
            "area_km2": [core_geometry.area / 1_000_000.0],
        },
        geometry=[core_geometry],
        crs=units.crs,
    )

    intersection_area = units.geometry.intersection(core_geometry).area
    total_area = units.geometry.area
    units["full_area_km2"] = total_area / 1_000_000.0
    units["core_area_km2"] = intersection_area / 1_000_000.0
    units["core_ratio"] = intersection_area / total_area
    units["centroid_in_core"] = units.geometry.centroid.within(core_geometry).astype(int)
    units["boundary_unit"] = (units["core_ratio"] < 0.999).astype(int)
    units["retain_rule"] = (
        (units["centroid_in_core"] == 1) | (units["core_ratio"] >= 0.5)
    ).astype(int)

    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()
    core_output.to_file(OUTPUT_GPKG, layer="core_1km", driver="GPKG")
    domain.to_file(
        OUTPUT_GPKG,
        layer="factor_domain_full_slope_units",
        driver="GPKG",
        mode="a",
    )
    if OUTPUT_UNITS.exists():
        OUTPUT_UNITS.unlink()
    units.to_file(OUTPUT_UNITS, layer="slope_units", driver="GPKG")

    columns = [
        column
        for column in [
            "SU_ID",
            "use_model",
            "full_area_km2",
            "core_area_km2",
            "core_ratio",
            "centroid_in_core",
            "boundary_unit",
            "retain_rule",
        ]
        if column in units.columns
    ]
    units.drop(columns="geometry")[columns].to_csv(
        OUTPUT_CSV, index=False, encoding="utf-8-sig"
    )

    summary = {
        "core_area_km2": core_geometry.area / 1_000_000.0,
        "factor_domain_area_km2": domain_geometry.area / 1_000_000.0,
        "factor_domain_outside_core_km2": (
            domain_geometry.difference(core_geometry).area / 1_000_000.0
        ),
        "slope_unit_count": len(units),
        "fully_inside_unit_count": int((units["core_ratio"] > 0.999).sum()),
        "boundary_unit_count": int((units["boundary_unit"] == 1).sum()),
        "recommended_retained_unit_count": int((units["retain_rule"] == 1).sum()),
        "recommended_rule": (
            "坡单元质心位于1 km核心区内，或与核心区面积重叠比例不低于50%；"
            "含已知滑坡的坡单元后续应强制保留。"
        ),
        "core_layer": f"{OUTPUT_GPKG}|layer=core_1km",
        "factor_domain_layer": (
            f"{OUTPUT_GPKG}|layer=factor_domain_full_slope_units"
        ),
        "full_slope_units": f"{OUTPUT_UNITS}|layer=slope_units",
        "crs": str(units.crs),
    }
    OUTPUT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
