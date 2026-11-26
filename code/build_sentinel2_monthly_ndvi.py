from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from scipy.ndimage import binary_dilation


ROOT = Path(r"D:\project\shuozhouinflucefactor")
NDVI_DIR = ROOT / "data" / "ndvi" / "sentinel2_monthly"
ITEMS_JSON = NDVI_DIR / "sentinel2_stac_items.json"
STUDY_GPKG = ROOT / "process" / "dem" / "analysis_boundaries.gpkg"
REFERENCE_DEM = ROOT / "data" / "gf7dem" / "gf7dem_10m.tif"
OUTPUT_STACK = NDVI_DIR / "sentinel2_ndvi_monthly_202211_202512.tif"
OUTPUT_STATS = NDVI_DIR / "sentinel2_ndvi_monthly_stats.csv"
OUTPUT_META = NDVI_DIR / "sentinel2_ndvi_metadata.json"

START_MONTH = "2022-11"
END_MONTH = "2025-12"
NODATA = -9999.0
BOA_ADD_OFFSET = -1000.0
QUANTIFICATION_VALUE = 10000.0
VALID_SCL = {4, 5, 6, 7}
MAX_CANDIDATES_PER_MONTH = 2
ACCEPT_VALID_RATIO = 0.85
REPAIR_CANDIDATES_PER_MONTH = 8
REPAIR_VALID_RATIO_THRESHOLD = 0.5


def target_grid() -> tuple[Affine, int, int, np.ndarray, tuple[float, float, float, float]]:
    study_wgs84 = gpd.read_file(STUDY_GPKG, layer="study_area").to_crs("EPSG:4326")
    study = study_wgs84.to_crs("EPSG:32649")
    minx, miny, maxx, maxy = study.total_bounds
    with rasterio.open(REFERENCE_DEM) as source:
        base = source.transform
        col0 = math.floor((minx - base.c) / base.a)
        col1 = math.ceil((maxx - base.c) / base.a)
        row0 = math.floor((base.f - maxy) / abs(base.e))
        row1 = math.ceil((base.f - miny) / abs(base.e))
        width = col1 - col0
        height = row1 - row0
        transform = base * Affine.translation(col0, row0)
    shapes = [(geometry, 1) for geometry in study.geometry]
    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    ).astype(bool)
    return transform, width, height, mask, tuple(study_wgs84.total_bounds)


def load_items() -> dict[str, list[dict]]:
    raw = json.loads(ITEMS_JSON.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for feature in raw["features"]:
        date = datetime.fromisoformat(
            feature["properties"]["datetime"].replace("Z", "+00:00")
        )
        key = date.strftime("%Y-%m")
        if START_MONTH <= key <= END_MONTH:
            grouped[key].append(feature)
    for key in grouped:
        grouped[key].sort(
            key=lambda item: (
                item["properties"].get("eo:cloud_cover", 100.0),
                item["properties"]["datetime"],
            )
        )
    return grouped


def read_item_cube(
    item: dict,
    bbox_wgs84: tuple[float, float, float, float],
    transform: Affine,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bbox_text = ",".join(f"{value:.8f}" for value in bbox_wgs84)
    query = urlencode(
        [
            ("collection", "sentinel-2-l2a"),
            ("item", item["id"]),
            ("assets", "B04"),
            ("assets", "B08"),
            ("assets", "SCL"),
            ("asset_as_band", "true"),
            ("width", str(width)),
            ("height", str(height)),
            ("dst_crs", "EPSG:32649"),
            ("resampling", "nearest"),
            ("return_mask", "false"),
        ]
    )
    url = (
        "https://planetarycomputer.microsoft.com/api/data/v1/item/bbox/"
        f"{bbox_text}.tif?{query}"
    )
    with urlopen(url, timeout=180) as response:
        content = response.read()
    with MemoryFile(content) as memory:
        with memory.open() as source:
            with WarpedVRT(
                source,
                crs="EPSG:32649",
                transform=transform,
                width=width,
                height=height,
                resampling=Resampling.nearest,
            ) as vrt:
                cube = vrt.read([1, 2, 3])
    return cube[0], cube[1], cube[2]


def calculate_ndvi(
    item: dict,
    bbox_wgs84: tuple[float, float, float, float],
    transform: Affine,
    width: int,
    height: int,
    study_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    red_dn, nir_dn, scl = read_item_cube(
        item,
        bbox_wgs84,
        transform,
        width,
        height,
    )
    red_dn = red_dn.astype("float32")
    nir_dn = nir_dn.astype("float32")
    scl = scl.astype("uint8")

    red = (red_dn + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE
    nir = (nir_dn + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE
    scl_invalid = ~np.isin(scl, list(VALID_SCL))
    scl_invalid = binary_dilation(scl_invalid, iterations=2)
    valid = (
        study_mask
        & ~scl_invalid
        & (red_dn > 0)
        & (nir_dn > 0)
        & (red > 0)
        & (nir > 0)
        & ((red + nir) > 0)
    )
    ndvi = np.full((height, width), np.nan, dtype="float32")
    ndvi[valid] = (nir[valid] - red[valid]) / (nir[valid] + red[valid])
    ndvi[(ndvi < -1.0) | (ndvi > 1.0)] = np.nan
    valid_ratio = float(np.isfinite(ndvi[study_mask]).sum() / study_mask.sum())
    return ndvi, valid_ratio


def raster_profile(transform: Affine, width: int, height: int) -> dict:
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32649",
        "transform": transform,
        "nodata": NODATA,
        "compress": "DEFLATE",
        "tiled": True,
    }


def write_month(
    month: str,
    ndvi: np.ndarray,
    transform: Affine,
    item: dict | None,
) -> Path:
    path = NDVI_DIR / f"ndvi_{month.replace('-', '')}_10m.tif"
    output = np.where(np.isfinite(ndvi), ndvi, NODATA).astype("float32")
    with rasterio.open(path, "w", **raster_profile(transform, ndvi.shape[1], ndvi.shape[0])) as target:
        target.write(output, 1)
        target.set_band_description(1, f"NDVI_{month}")
        target.update_tags(
            source="Sentinel-2 L2A via Microsoft Planetary Computer",
            month=month,
            item_id="" if item is None else item["id"],
            formula="(B08_BOA-B04_BOA)/(B08_BOA+B04_BOA)",
            boa_add_offset=BOA_ADD_OFFSET,
            quantification_value=QUANTIFICATION_VALUE,
            scl_valid_classes="4,5,6,7",
        )
    return path


def make_preview(stats: pd.DataFrame) -> None:
    dates = pd.to_datetime(stats["month"] + "-01")
    fig, axis = plt.subplots(figsize=(12, 4.5), dpi=150)
    axis.plot(dates, stats["ndvi_median"], marker="o", markersize=3, linewidth=1)
    axis.fill_between(
        dates,
        stats["ndvi_p10"],
        stats["ndvi_p90"],
        alpha=0.2,
        label="P10–P90",
    )
    axis.set_ylim(-0.2, 1.0)
    axis.set_ylabel("NDVI")
    axis.set_xlabel("Month")
    axis.set_title("Monthly Sentinel-2 NDVI in the 1 km buffer")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(NDVI_DIR / "sentinel2_ndvi_monthly_preview.png", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repair-low",
        action="store_true",
        help="Reprocess only existing months whose valid ratio is below 0.5.",
    )
    return parser.parse_args()


def read_existing_month(month: str) -> np.ndarray:
    path = NDVI_DIR / f"ndvi_{month.replace('-', '')}_10m.tif"
    with rasterio.open(path) as source:
        return source.read(1, masked=True).filled(np.nan).astype("float32")


def main() -> None:
    args = parse_args()
    NDVI_DIR.mkdir(parents=True, exist_ok=True)
    transform, width, height, study_mask, bbox_wgs84 = target_grid()
    grouped = load_items()
    months = pd.period_range(START_MONTH, END_MONTH, freq="M").astype(str).tolist()
    arrays: list[np.ndarray] = []
    rows: list[dict] = []
    existing_rows: dict[str, dict] = {}
    repair_months: set[str] = set()
    if args.repair_low and OUTPUT_STATS.exists():
        existing = pd.read_csv(OUTPUT_STATS)
        existing_rows = existing.set_index("month").to_dict(orient="index")
        repair_months = set(
            existing.loc[
                existing["valid_ratio"] < REPAIR_VALID_RATIO_THRESHOLD,
                "month",
            ].astype(str)
        )

    for month in months:
        if args.repair_low and month not in repair_months:
            arrays.append(read_existing_month(month))
            rows.append({"month": month, **existing_rows[month]})
            continue

        candidate_limit = (
            REPAIR_CANDIDATES_PER_MONTH
            if args.repair_low
            else MAX_CANDIDATES_PER_MONTH
        )
        candidates = grouped.get(month, [])[:candidate_limit]
        best_array = np.full((height, width), np.nan, dtype="float32")
        best_ratio = -1.0
        best_item: dict | None = None
        attempted = 0
        error_messages: list[str] = []
        for item in candidates:
            attempted += 1
            try:
                array, ratio = calculate_ndvi(
                    item,
                    bbox_wgs84,
                    transform,
                    width,
                    height,
                    study_mask,
                )
            except Exception as error:
                error_messages.append(f"{item['id']}: {error!r}")
                continue
            if ratio > best_ratio:
                best_array = array
                best_ratio = ratio
                best_item = item
            if best_ratio >= ACCEPT_VALID_RATIO:
                break

        write_month(month, best_array, transform, best_item)
        arrays.append(best_array)
        values = best_array[np.isfinite(best_array)]
        values_for_stats = (
            values
            if best_ratio >= REPAIR_VALID_RATIO_THRESHOLD
            else np.array([], dtype="float32")
        )
        rows.append(
            {
                "month": month,
                "scene_id": "" if best_item is None else best_item["id"],
                "acquisition_utc": (
                    "" if best_item is None else best_item["properties"]["datetime"]
                ),
                "catalog_cloud_percent": (
                    np.nan
                    if best_item is None
                    else best_item["properties"].get("eo:cloud_cover", np.nan)
                ),
                "candidate_scenes_available": len(grouped.get(month, [])),
                "candidate_scenes_attempted": attempted,
                "valid_ratio": max(best_ratio, 0.0),
                "ndvi_mean": (
                    np.nan
                    if values_for_stats.size == 0
                    else float(np.mean(values_for_stats))
                ),
                "ndvi_median": (
                    np.nan
                    if values_for_stats.size == 0
                    else float(np.median(values_for_stats))
                ),
                "ndvi_p10": (
                    np.nan
                    if values_for_stats.size == 0
                    else float(np.percentile(values_for_stats, 10))
                ),
                "ndvi_p90": (
                    np.nan
                    if values_for_stats.size == 0
                    else float(np.percentile(values_for_stats, 90))
                ),
                "errors": " | ".join(error_messages),
            }
        )
        print(
            f"{month}: scene={rows[-1]['scene_id'] or 'NONE'}, "
            f"valid={rows[-1]['valid_ratio']:.3f}, "
            f"median={rows[-1]['ndvi_median']}"
        )

    stats = pd.DataFrame(rows)
    stats.to_csv(OUTPUT_STATS, index=False, encoding="utf-8-sig")

    profile = raster_profile(transform, width, height)
    profile["count"] = len(months)
    with rasterio.open(OUTPUT_STACK, "w", **profile) as target:
        for index, (month, array) in enumerate(zip(months, arrays), start=1):
            target.write(np.where(np.isfinite(array), array, NODATA).astype("float32"), index)
            target.set_band_description(index, f"NDVI_{month}")
        target.update_tags(
            source="Sentinel-2 L2A via Microsoft Planetary Computer",
            time_range=f"{START_MONTH}/{END_MONTH}",
            formula="(B08_BOA-B04_BOA)/(B08_BOA+B04_BOA)",
        )
    make_preview(stats)

    metadata = {
        "source": "Sentinel-2 Level-2A, Microsoft Planetary Computer STAC",
        "stac_url": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "collection": "sentinel-2-l2a",
        "time_range": [START_MONTH, END_MONTH],
        "output_crs": "EPSG:32649",
        "pixel_size_m": 10,
        "grid_shape": [height, width],
        "monthly_band_count": len(months),
        "method": "For each month, test up to two scenes ordered by catalog cloud cover and retain the scene with the highest SCL-masked valid coverage; months below 50% valid coverage were re-tested with up to eight scenes.",
        "radiometry": {
            "boa_add_offset": BOA_ADD_OFFSET,
            "quantification_value": QUANTIFICATION_VALUE,
            "reference": "Sentinel-2 L2A processing baseline 04.00+",
        },
        "cloud_mask": {
            "source": "Sentinel-2 SCL at 20 m, nearest-neighbour aligned to the 10 m grid",
            "retained_classes": sorted(VALID_SCL),
            "invalid_mask_dilation_pixels_10m": 2,
        },
        "months_with_any_valid_pixels": int((stats["valid_ratio"] > 0).sum()),
        "months_valid_ratio_ge_0_8": int((stats["valid_ratio"] >= 0.8).sum()),
        "limitations": [
            "Monthly rasters are best-clear-scene snapshots, not within-month median composites.",
            "Snow/ice is masked; winter months may have low or missing valid coverage.",
            "NDVI is used as a time-varying surface condition indicator, not as a direct landslide observation.",
        ],
    }
    OUTPUT_META.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
