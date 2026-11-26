from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask


ROOT = Path(r"D:\project\shuozhouinflucefactor")
SOIL_DIR = ROOT / "process" / "soil"
DEM_DIR = ROOT / "process" / "dem"
SOIL_RASTER = SOIL_DIR / "soilgrids_wrb_most_probable_utm49n_250m.tif"
RAT_JSON = SOIL_DIR / "MostProbable.rat.json"


def load_classes() -> dict[int, str]:
    raw = json.loads(RAT_JSON.read_text(encoding="utf-8"))
    return {int(code): name for code, name in raw.items() if code.isdigit()}


def make_legend(classes: dict[int, str]) -> None:
    pd.DataFrame(
        [{"code": code, "wrb_class": name} for code, name in sorted(classes.items())]
    ).to_csv(SOIL_DIR / "soilgrids_wrb_classes.csv", index=False, encoding="utf-8-sig")


def zonal_majority(units: gpd.GeoDataFrame, classes: dict[int, str]) -> gpd.GeoDataFrame:
    with rasterio.open(SOIL_RASTER) as source:
        projected = units.to_crs(source.crs)
        nodata = source.nodata
        codes: list[int] = []
        ratios: list[float] = []
        for geometry in projected.geometry:
            data, _ = mask(source, [geometry], crop=True, filled=False, all_touched=True)
            values = data[0].compressed()
            if nodata is not None:
                values = values[values != nodata]
            if values.size == 0:
                centroid = geometry.centroid
                value = next(source.sample([(centroid.x, centroid.y)]))[0]
                values = np.array([value])
            unique, counts = np.unique(values.astype(int), return_counts=True)
            index = int(np.argmax(counts))
            codes.append(int(unique[index]))
            ratios.append(float(counts[index] / counts.sum()))

    result = units.copy()
    result["SOILcode"] = codes
    result["SOILtype"] = [classes.get(code, "Unknown") for code in codes]
    result["SOILratio"] = ratios
    return result


def update_slope_units(classes: dict[int, str]) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for filename in ["slope_units_main_full.gpkg", "slope_units_main_clipped.gpkg"]:
        path = DEM_DIR / filename
        units = gpd.read_file(path, layer="slope_units")
        units = zonal_majority(units, classes)
        units.to_file(path, layer="slope_units", driver="GPKG")
        summaries[filename] = {
            "feature_count": int(len(units)),
            "classes": {
                str(key): int(value)
                for key, value in units["SOILtype"].value_counts().to_dict().items()
            },
            "majority_ratio_min": float(units["SOILratio"].min()),
        }
    return summaries


def make_preview(classes: dict[int, str]) -> None:
    with rasterio.open(SOIL_RASTER) as source:
        array = source.read(1, masked=True)
        bounds = source.bounds
        crs = source.crs

    study = gpd.read_file(DEM_DIR / "analysis_boundaries.gpkg", layer="study_area").to_crs(crs)
    fig, axis = plt.subplots(figsize=(8, 7), dpi=160)
    axis.imshow(
        array,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        interpolation="nearest",
        cmap="YlOrBr",
    )
    study.boundary.plot(ax=axis, color="red", linewidth=1.2, label="Study area")
    present = np.unique(array.compressed()).astype(int)
    label = ", ".join(f"{code}: {classes.get(code, 'Unknown')}" for code in present)
    axis.set_title(f"SoilGrids WRB Most Probable Class (250 m)\n{label}")
    axis.set_xlabel("Easting (m), EPSG:32649")
    axis.set_ylabel("Northing (m)")
    axis.legend(loc="upper right")
    axis.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(SOIL_DIR / "soilgrids_wrb_preview.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    classes = load_classes()
    make_legend(classes)
    summaries = update_slope_units(classes)
    make_preview(classes)

    with rasterio.open(SOIL_RASTER) as source:
        array = source.read(1, masked=True)
        codes, counts = np.unique(array.compressed().astype(int), return_counts=True)
        raster_classes = {
            classes.get(int(code), str(int(code))): int(count)
            for code, count in zip(codes, counts)
        }
        metadata = {
            "source": "ISRIC SoilGrids WRB MostProbable 250 m",
            "source_url": "https://files.isric.org/soilgrids/latest/data/wrb/MostProbable.vrt",
            "license": "CC BY 4.0",
            "source_product_note": "WRB categorical layer originates from the SoilGrids 2017 WRB product re-run published in the latest WebDAV directory.",
            "native_subset": "soilgrids_wrb_most_probable_native_250m.tif",
            "analysis_raster": SOIL_RASTER.name,
            "analysis_crs": str(source.crs),
            "pixel_size_m": list(source.res),
            "raster_shape": list(source.shape),
            "classes_in_processing_extent": raster_classes,
            "slope_unit_summary": summaries,
            "modelling_limitation": "The processing extent contains only Cambisols, so this factor has zero spatial discrimination and should not be entered into the susceptibility model.",
        }
    (SOIL_DIR / "soilgrids_wrb_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
