from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from osgeo import gdal


ROOT = Path(r"D:\project\shuozhouinflucefactor")
OUTPUT_DIR = ROOT / "data" / "landcover" / "annual_10m"
DOMAIN_GPKG = ROOT / "process" / "dem" / "analysis_boundaries.gpkg"
DOMAIN_LAYER = "study_area"
REFERENCE_RASTER = (
    ROOT / "data" / "ndvi" / "sentinel2_monthly" / "ndvi_202211_10m.tif"
)
DOMAIN_NAME = "Great Wall 1 km core area"
SERVICE_URL = (
    "https://ic.imagery1.arcgis.com/arcgis/rest/services/"
    "Sentinel2_10m_LandCover/ImageServer"
)
YEARS = (2022, 2023, 2024, 2025)
NODATA = 0

CLASSES = {
    1: ("Water", "水体", "#419BDF"),
    2: ("Trees", "林地", "#397D49"),
    4: ("Flooded vegetation", "淹水植被", "#7A87C6"),
    5: ("Crops", "耕地", "#E49635"),
    7: ("Built area", "建设用地", "#C4281B"),
    8: ("Bare ground", "裸地", "#A59B8F"),
    9: ("Snow/Ice", "冰雪", "#A8EBFF"),
    10: ("Clouds", "云", "#B39FE1"),
    11: ("Rangeland", "草地/灌草地", "#E6E26B"),
}


def target_grid() -> tuple[dict, np.ndarray]:
    reference = gdal.Open(str(REFERENCE_RASTER))
    if reference is None:
        raise FileNotFoundError(REFERENCE_RASTER)
    vector = gdal.OpenEx(str(DOMAIN_GPKG), gdal.OF_VECTOR)
    if vector is None:
        raise FileNotFoundError(DOMAIN_GPKG)
    layer = vector.GetLayerByName(DOMAIN_LAYER)
    if layer is None:
        raise RuntimeError(f"{DOMAIN_LAYER} layer is missing")

    reference_transform = reference.GetGeoTransform()
    minx, maxx, miny, maxy = layer.GetExtent()
    col0 = math.floor((minx - reference_transform[0]) / reference_transform[1])
    col1 = math.ceil((maxx - reference_transform[0]) / reference_transform[1])
    row0 = math.floor(
        (reference_transform[3] - maxy) / abs(reference_transform[5])
    )
    row1 = math.ceil(
        (reference_transform[3] - miny) / abs(reference_transform[5])
    )
    width = col1 - col0
    height = row1 - row0
    geotransform = (
        reference_transform[0] + col0 * reference_transform[1],
        reference_transform[1],
        0.0,
        reference_transform[3] + row0 * reference_transform[5],
        0.0,
        reference_transform[5],
    )
    profile = {
        "width": width,
        "height": height,
        "geotransform": geotransform,
        "projection": reference.GetProjection(),
        "crs": "EPSG:32649",
    }
    mask_dataset = gdal.GetDriverByName("MEM").Create(
        "", profile["width"], profile["height"], 1, gdal.GDT_Byte
    )
    mask_dataset.SetGeoTransform(profile["geotransform"])
    mask_dataset.SetProjection(profile["projection"])
    mask_dataset.GetRasterBand(1).Fill(0)
    result = gdal.RasterizeLayer(
        mask_dataset, [1], layer, burn_values=[1], options=["ALL_TOUCHED=FALSE"]
    )
    if result != 0:
        raise RuntimeError(f"RasterizeLayer failed with code {result}")
    mask = mask_dataset.GetRasterBand(1).ReadAsArray().astype(bool)
    vector = None
    mask_dataset = None
    reference = None
    return profile, mask


def query_available_years() -> list[dict]:
    params = urlencode(
        {
            "where": "Year in (2022,2023,2024,2025)",
            "outFields": "OBJECTID,Name,Year",
            "returnGeometry": "false",
            "orderByFields": "Year",
            "f": "json",
        }
    )
    request = Request(
        f"{SERVICE_URL}/query?{params}",
        headers={"User-Agent": "Codex-landcover-retrieval/1.0"},
    )
    with urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
    if "error" in result:
        raise RuntimeError(result["error"])
    return [feature["attributes"] for feature in result["features"]]


def download_year(year: int, profile: dict) -> tuple[np.ndarray, str]:
    width = profile["width"]
    height = profile["height"]
    geotransform = profile["geotransform"]
    left = geotransform[0]
    top = geotransform[3]
    right = left + width * geotransform[1]
    bottom = top + height * geotransform[5]
    mosaic_rule = {
        "mosaicMethod": "esriMosaicAttribute",
        "where": f"Year = {year}",
        "sortField": "Year",
        "sortValue": str(year),
        "ascending": True,
        "mosaicOperation": "MT_FIRST",
    }
    params = urlencode(
        {
            "bbox": f"{left},{bottom},{right},{top}",
            "bboxSR": "32649",
            "size": f"{width},{height}",
            "imageSR": "32649",
            "format": "tiff",
            "pixelType": "U8",
            "noData": str(NODATA),
            "noDataInterpretation": "esriNoDataMatchAny",
            "interpolation": "RSP_NearestNeighbor",
            "mosaicRule": json.dumps(mosaic_rule, separators=(",", ":")),
            "f": "image",
        }
    )
    url = f"{SERVICE_URL}/exportImage?{params}"
    request = Request(url, headers={"User-Agent": "Codex-landcover-retrieval/1.0"})
    with urlopen(request, timeout=300) as response:
        payload = response.read()
        content_type = response.headers.get("Content-Type", "")
    if "json" in content_type.lower() or payload.startswith(b"{"):
        raise RuntimeError(payload.decode("utf-8", errors="replace"))

    memory_path = f"/vsimem/landcover_{year}.tif"
    gdal.FileFromMemBuffer(memory_path, payload)
    source = gdal.Open(memory_path)
    if source is None:
        gdal.Unlink(memory_path)
        raise RuntimeError(f"GDAL could not open the {year} response")
    warped = gdal.Warp(
        "",
        source,
        format="MEM",
        width=width,
        height=height,
        outputBounds=(left, bottom, right, top),
        dstSRS=profile["projection"],
        outputType=gdal.GDT_Byte,
        resampleAlg=gdal.GRA_NearestNeighbour,
        srcNodata=NODATA,
        dstNodata=NODATA,
    )
    if warped is None:
        source = None
        gdal.Unlink(memory_path)
        raise RuntimeError(f"GDAL warp failed for {year}")
    output = warped.GetRasterBand(1).ReadAsArray().astype("uint8")
    warped = None
    source = None
    gdal.Unlink(memory_path)
    return output, url


def write_raster(
    path: Path,
    array: np.ndarray,
    profile: dict,
    year: int,
    mask: np.ndarray,
) -> None:
    masked = np.where(mask, array, NODATA).astype("uint8")
    driver = gdal.GetDriverByName("GTiff")
    target = driver.Create(
        str(path),
        profile["width"],
        profile["height"],
        1,
        gdal.GDT_Byte,
        options=["COMPRESS=DEFLATE", "TILED=YES"],
    )
    target.SetGeoTransform(profile["geotransform"])
    target.SetProjection(profile["projection"])
    band = target.GetRasterBand(1)
    band.WriteArray(masked)
    band.SetNoDataValue(NODATA)
    band.SetDescription(f"Annual_land_cover_{year}")
    target.SetMetadata(
        {
            "source": "Esri-Impact Observatory Sentinel-2 10m Land Use/Land Cover Time Series",
            "source_service": SERVICE_URL,
            "year": str(year),
            "license": "CC BY 4.0",
            "resampling": "nearest",
            "study_area": DOMAIN_NAME,
        }
    )
    target.FlushCache()
    target = None


def write_array(
    path: Path,
    array: np.ndarray,
    profile: dict,
    gdal_type: int,
    nodata: int,
    description: str,
) -> None:
    target = gdal.GetDriverByName("GTiff").Create(
        str(path),
        profile["width"],
        profile["height"],
        1,
        gdal_type,
        options=["COMPRESS=DEFLATE", "TILED=YES"],
    )
    target.SetGeoTransform(profile["geotransform"])
    target.SetProjection(profile["projection"])
    band = target.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(nodata)
    band.SetDescription(description)
    target.FlushCache()
    target = None


def summarize_years(arrays: dict[int, np.ndarray], mask: np.ndarray) -> None:
    pixel_area_km2 = 100.0 / 1_000_000.0
    rows = []
    for year, array in arrays.items():
        counts = Counter(array[mask].tolist())
        valid_count = sum(counts[class_id] for class_id in CLASSES)
        for class_id, (english, chinese, _) in CLASSES.items():
            count = counts[class_id]
            rows.append(
                {
                    "year": year,
                    "class_id": class_id,
                    "class_en": english,
                    "class_zh": chinese,
                    "pixel_count": count,
                    "area_km2": count * pixel_area_km2,
                    "percent_of_valid": (
                        count / valid_count * 100.0 if valid_count else np.nan
                    ),
                }
            )
    with (OUTPUT_DIR / "landcover_annual_class_statistics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_change_products(
    arrays: dict[int, np.ndarray], mask: np.ndarray, profile: dict
) -> None:
    first = arrays[YEARS[0]]
    last = arrays[YEARS[-1]]
    valid = mask & (first != NODATA) & (last != NODATA)

    change = np.full(first.shape, 255, dtype="uint8")
    change[valid] = (first[valid] != last[valid]).astype("uint8")
    write_array(
        OUTPUT_DIR / "landcover_change_2022_2025.tif",
        change,
        profile,
        gdal.GDT_Byte,
        255,
        "0=unchanged;1=changed;255=nodata",
    )

    transition = np.zeros(first.shape, dtype="uint16")
    transition[valid] = first[valid].astype("uint16") * 100 + last[valid]
    write_array(
        OUTPUT_DIR / "landcover_transition_2022_2025.tif",
        transition,
        profile,
        gdal.GDT_UInt16,
        0,
        "transition_code=class_2022*100+class_2025",
    )

    transition_counts = Counter(transition[valid].tolist())
    rows = []
    for code, count in sorted(transition_counts.items()):
        from_id, to_id = divmod(code, 100)
        rows.append(
            {
                "transition_code": code,
                "from_class_id": from_id,
                "from_class_zh": CLASSES.get(from_id, ("", "未知", ""))[1],
                "to_class_id": to_id,
                "to_class_zh": CLASSES.get(to_id, ("", "未知", ""))[1],
                "pixel_count": count,
                "area_km2": count * 0.0001,
                "changed": from_id != to_id,
            }
        )
    with (OUTPUT_DIR / "landcover_transition_2022_2025.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_representative_products(
    arrays: dict[int, np.ndarray], mask: np.ndarray, profile: dict
) -> None:
    stack = np.stack([arrays[year] for year in YEARS])
    valid = mask & np.all(stack != NODATA, axis=0)
    mode = np.zeros(stack.shape[1:], dtype="uint8")
    agreement = np.zeros(stack.shape[1:], dtype="uint8")
    best_count = np.zeros(stack.shape[1:], dtype="uint8")
    latest = arrays[YEARS[-1]]
    for class_id in CLASSES:
        count = np.sum(stack == class_id, axis=0).astype("uint8")
        replace = (count > best_count) | (
            (count == best_count) & (latest == class_id)
        )
        mode[replace] = class_id
        best_count[replace] = count[replace]
    mode[~valid] = NODATA
    agreement[valid] = best_count[valid]
    write_array(
        OUTPUT_DIR / "landcover_mode_2022_2025_10m.tif",
        mode,
        profile,
        gdal.GDT_Byte,
        NODATA,
        "Modal annual class for 2022-2025; ties prefer 2025",
    )
    write_array(
        OUTPUT_DIR / "landcover_agreement_count_2022_2025.tif",
        agreement,
        profile,
        gdal.GDT_Byte,
        NODATA,
        "Number of annual maps agreeing with modal class; range 1-4",
    )


def write_class_table() -> None:
    with (OUTPUT_DIR / "landcover_classes.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["class_id", "class_en", "class_zh", "color_hex"])
        for class_id, values in CLASSES.items():
            writer.writerow([class_id, *values])


def make_preview(arrays: dict[int, np.ndarray], mask: np.ndarray) -> None:
    colors = ["#FFFFFF"] + [CLASSES[key][2] for key in CLASSES]
    class_ids = [0] + list(CLASSES)
    remapped = {}
    for year, array in arrays.items():
        display = np.full(array.shape, np.nan, dtype="float32")
        for display_id, class_id in enumerate(class_ids):
            display[(array == class_id) & mask] = display_id
        remapped[year] = display

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1), cmap.N)
    fig, axes = plt.subplots(2, 2, figsize=(10, 12), dpi=160)
    for axis, year in zip(axes.flat, YEARS):
        axis.imshow(remapped[year], cmap=cmap, norm=norm)
        axis.set_title(str(year))
        axis.set_axis_off()
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=8,
            markerfacecolor=CLASSES[class_id][2],
            markeredgecolor="none",
            label=f"{class_id} {CLASSES[class_id][0]}",
        )
        for class_id in CLASSES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle(f"Sentinel-2 10 m Annual Land Cover — {DOMAIN_NAME}")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(
        OUTPUT_DIR / "landcover_2022_2025_preview.png",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    global OUTPUT_DIR, DOMAIN_GPKG, DOMAIN_LAYER, REFERENCE_RASTER, DOMAIN_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        choices=["core", "full"],
        default="core",
        help="core=1 km study area; full=complete slope units intersecting core",
    )
    args = parser.parse_args()
    if args.domain == "full":
        OUTPUT_DIR = (
            ROOT / "data" / "landcover" / "annual_10m_full_slope_units"
        )
        DOMAIN_GPKG = (
            ROOT / "process" / "factor_domain" / "analysis_domains.gpkg"
        )
        DOMAIN_LAYER = "factor_domain_full_slope_units"
        REFERENCE_RASTER = ROOT / "data" / "gf7dem" / "gf7dem_10m.tif"
        DOMAIN_NAME = "Complete slope-unit factor domain intersecting the 1 km core"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profile, mask = target_grid()
    catalog_records = query_available_years()
    available = {record["Year"] for record in catalog_records}
    missing = set(YEARS) - available
    if missing:
        raise RuntimeError(f"Official service is missing years: {sorted(missing)}")

    arrays: dict[int, np.ndarray] = {}
    request_urls: dict[int, str] = {}
    for year in YEARS:
        array, url = download_year(year, profile)
        invalid_classes = sorted(set(np.unique(array)) - set(CLASSES) - {NODATA})
        if invalid_classes:
            raise RuntimeError(
                f"{year} contains unexpected class values: {invalid_classes}"
            )
        if not np.any(mask & (array != NODATA)):
            raise RuntimeError(f"{year} contains no valid pixels in the study area")
        path = OUTPUT_DIR / f"landcover_{year}_10m.tif"
        write_raster(path, array, profile, year, mask)
        arrays[year] = np.where(mask, array, NODATA).astype("uint8")
        request_urls[year] = url
        print(f"{year}: {path.name}; classes={sorted(np.unique(arrays[year]))}")

    write_class_table()
    summarize_years(arrays, mask)
    write_change_products(arrays, mask, profile)
    write_representative_products(arrays, mask, profile)
    make_preview(arrays, mask)

    metadata = {
        "dataset": "Sentinel-2 10m Land Use/Land Cover Time Series",
        "producers": ["Impact Observatory", "Microsoft", "Esri"],
        "source_service": SERVICE_URL,
        "official_catalog_records": catalog_records,
        "years": list(YEARS),
        "temporal_interpretation": (
            "Annual land-cover snapshots for 2022, 2023, 2024 and 2025; "
            "land cover is not a monthly variable."
        ),
        "output_crs": str(profile["crs"]),
        "pixel_size_m": [
            abs(profile["geotransform"][1]),
            abs(profile["geotransform"][5]),
        ],
        "grid_shape": [profile["height"], profile["width"]],
        "aligned_to": str(REFERENCE_RASTER),
        "mask": f"{DOMAIN_GPKG}|layer={DOMAIN_LAYER}",
        "resampling": "nearest neighbour",
        "nodata": NODATA,
        "class_ids": {
            str(key): {"english": value[0], "chinese": value[1]}
            for key, value in CLASSES.items()
        },
        "license": "CC BY 4.0",
        "limitations": [
            "This is a global/regional deep-learning product; visually verify local class boundaries before slope-unit modelling.",
            "The annual label represents the dominant class inferred from Sentinel-2 observations within each calendar year.",
            "Class 10 is cloud; treat it as missing rather than a terrain condition.",
            "For susceptibility modelling, use one representative land-cover year or a stable/modal class, not four highly correlated annual layers simultaneously.",
            "landcover_mode_2022_2025_10m.tif is the recommended susceptibility factor; class ties are resolved in favour of the 2025 label.",
        ],
        "export_request_urls": request_urls,
    }
    (OUTPUT_DIR / "landcover_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
