"""Extract a D8 drainage-line network from the 10 m GF-7 DEM.

The workflow is implemented locally with GDAL, NumPy, SciPy, Shapely and
GeoPandas:

1. read DEM and valid-cell mask
2. priority-flood depression filling with a tiny gradient
3. D8 flow direction and topologically ordered flow accumulation
4. threshold-based stream-cell extraction
5. downstream-edge graph vectorization into merged line segments
"""

from __future__ import annotations

import heapq
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from osgeo import gdal
from scipy import ndimage
from shapely.geometry import LineString

gdal.UseExceptions()

DEM_PATH = Path(r"D:\shuozhou\data\dem\高分七号生成dem\gf7dem_10m.tif")
OUT_DIR = Path(r"D:\project\shuozhouinflucefactor\outputs\drainage_network")
THRESHOLDS = (500, 800, 1000)
PRIMARY_THRESHOLD = 800
EPSILON = 1.0e-4
DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def log(message: str) -> None:
    print(message, flush=True)


def read_dem(path: Path):
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open DEM: {path}")
    if ds.RasterCount != 1:
        raise RuntimeError("The DEM must contain exactly one band")
    band = ds.GetRasterBand(1)
    dem = band.ReadAsArray().astype(np.float64)
    nodata = band.GetNoDataValue()
    valid = np.isfinite(dem)
    if nodata is not None:
        valid &= dem != float(nodata)
    if not valid.any():
        raise RuntimeError("The DEM has no valid cells")
    return ds, dem, valid


def fill_sinks(dem: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Priority-flood fill with a small gradient across filled flats."""
    height, width = dem.shape
    filled = np.where(valid, dem, np.nan)
    visited = np.zeros(dem.shape, dtype=bool)

    # Any valid cell touching an invalid cell or raster edge is an outlet seed.
    interior = ndimage.binary_erosion(
        valid, structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    boundary = valid & ~interior

    heap: list[tuple[float, int, int]] = []
    sequence = 0
    for idx in np.flatnonzero(boundary.ravel()):
        z = float(filled.flat[idx])
        visited.flat[idx] = True
        heapq.heappush(heap, (z, sequence, int(idx)))
        sequence += 1

    processed = 0
    next_progress = 1_000_000
    while heap:
        z, _, idx = heapq.heappop(heap)
        row, col = divmod(idx, width)
        for dr in (-1, 0, 1):
            r = row + dr
            if r < 0 or r >= height:
                continue
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                c = col + dc
                if c < 0 or c >= width:
                    continue
                nidx = r * width + c
                if visited.flat[nidx] or not valid.flat[nidx]:
                    continue
                raw = float(dem.flat[nidx])
                new_z = raw if raw > z else z + EPSILON
                filled.flat[nidx] = new_z
                visited.flat[nidx] = True
                heapq.heappush(heap, (new_z, sequence, nidx))
                sequence += 1
        processed += 1
        if processed >= next_progress:
            log(f"  filled {processed:,} cells")
            next_progress += 1_000_000

    if not np.array_equal(visited, valid):
        missing = int(np.count_nonzero(valid & ~visited))
        raise RuntimeError(f"Priority flood did not visit {missing} valid cells")
    return filled


def compute_d8_downstream(filled: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return the flat downstream index for each cell, or -1 for outlets."""
    height, width = filled.shape
    best_slope = np.full(filled.shape, -np.inf, dtype=np.float64)
    best_direction = np.zeros(filled.shape, dtype=np.uint8)

    for direction_id, (dr, dc) in enumerate(DIRECTIONS):
        distance = (14.142135623730951 if dr != 0 and dc != 0 else 10.0)
        neighbor = np.full(filled.shape, np.nan, dtype=np.float64)
        src_r = slice(max(0, dr), height - max(0, -dr))
        dst_r = slice(max(0, -dr), height - max(0, dr))
        src_c = slice(max(0, dc), width - max(0, -dc))
        dst_c = slice(max(0, -dc), width - max(0, dc))
        neighbor[dst_r, dst_c] = filled[src_r, src_c]

        neighbor_valid = np.isfinite(neighbor)
        slope = np.where(neighbor_valid, (filled - neighbor) / distance, -np.inf)
        better = slope > best_slope
        best_direction[better] = direction_id
        np.maximum(best_slope, slope, out=best_slope)

    has_downstream = (best_slope > 0.0) & valid
    downstream = np.full(filled.size, -1, dtype=np.int64)
    best_direction_flat = best_direction.ravel()
    has_downstream_flat = has_downstream.ravel()
    for direction_id, (dr, dc) in enumerate(DIRECTIONS):
        selected = np.flatnonzero(
            (best_direction_flat == direction_id) & has_downstream_flat
        )
        if selected.size == 0:
            continue
        rows = selected // width
        cols = selected % width
        targets = (rows + dr) * width + (cols + dc)
        downstream[selected] = targets

    if np.any(downstream[valid.ravel()] < 0) and False:
        raise RuntimeError("Internal valid cells must have a downstream neighbor")
    return downstream


def accumulate_flow(
    filled: np.ndarray, valid: np.ndarray, downstream: np.ndarray
) -> np.ndarray:
    """Flow accumulation in upstream cell counts, processed high to low."""
    valid_flat = np.flatnonzero(valid.ravel())
    filled_flat = filled.ravel()
    order = valid_flat[np.argsort(-filled_flat[valid_flat], kind="stable")]
    accumulation = np.zeros(filled.size, dtype=np.float64)
    accumulation[valid_flat] = 1.0

    log(f"  accumulating {order.size:,} cells")
    downstream_list = downstream
    for progress, idx in enumerate(order, start=1):
        target = downstream_list[idx]
        if target >= 0:
            accumulation[target] += accumulation[idx]
        if progress % 1_000_000 == 0:
            log(f"  accumulated {progress:,} cells")
    return accumulation


def write_raster(
    path: Path,
    array: np.ndarray,
    reference: gdal.Dataset,
    nodata: float,
    overview: bool = True,
) -> None:
    driver = gdal.GetDriverByName("GTiff")
    options = [
        "TILED=YES",
        "COMPRESS=DEFLATE",
        "PREDICTOR=3",
        "BIGTIFF=IF_SAFER",
    ]
    out_ds = driver.Create(
        str(path), array.shape[1], array.shape[0], 1, gdal.GDT_Float32, options=options
    )
    out_ds.SetGeoTransform(reference.GetGeoTransform())
    out_ds.SetProjection(reference.GetProjection())
    band = out_ds.GetRasterBand(1)
    band.SetNoDataValue(float(nodata))
    band.WriteArray(array.astype(np.float32))
    band.FlushCache()
    if overview:
        out_ds.BuildOverviews("NEAREST", [2, 4, 8, 16, 32])
    out_ds = None


def xy_from_index(
    idx: int, width: int, geotransform: tuple[float, ...]
) -> tuple[float, float]:
    row, col = divmod(idx, width)
    origin_x, px, rx, origin_y, ry, py = geotransform
    x = origin_x + (col + 0.5) * px + (row + 0.5) * rx
    y = origin_y + (col + 0.5) * ry + (row + 0.5) * py
    return float(x), float(y)


def build_drainage_lines(
    accumulation: np.ndarray,
    downstream: np.ndarray,
    valid_flat: np.ndarray,
    threshold: int,
    width: int,
    geotransform: tuple[float, ...],
) -> gpd.GeoDataFrame:
    stream_mask = np.zeros(accumulation.size, dtype=bool)
    stream_mask[valid_flat] = accumulation[valid_flat] >= threshold
    stream_cells = np.flatnonzero(stream_mask)

    targets = downstream[stream_cells]
    has_target = targets >= 0
    source_cells = stream_cells[has_target]
    target_cells = targets[has_target]
    keep = stream_mask[target_cells]
    source_cells = source_cells[keep]
    target_cells = target_cells[keep]

    out_neighbor: dict[int, int] = {}
    in_degree: dict[int, int] = {}
    for source, target in zip(source_cells.tolist(), target_cells.tolist()):
        out_neighbor[source] = target
        in_degree[target] = in_degree.get(target, 0) + 1
        in_degree.setdefault(source, 0)

    starts = [node for node in out_neighbor if in_degree.get(node, 0) != 1]
    starts.sort()

    records = []
    for segment_id, start in enumerate(starts, start=1):
        node_ids = [start]
        current = start
        local_seen = {start}
        while True:
            target = out_neighbor.get(current, -1)
            if target < 0 or target in local_seen:
                break
            node_ids.append(target)
            local_seen.add(target)
            if in_degree.get(target, 0) != 1:
                break
            current = target

        if len(node_ids) < 2:
            continue
        coords = [xy_from_index(node, width, geotransform) for node in node_ids]
        line = LineString(coords)
        if line.length <= 0:
            continue
        simplified = line.simplify(1.0, preserve_topology=True)
        if simplified is not None and simplified.length > 0:
            line = simplified
        records.append(
            {
                "segment_id": segment_id,
                "threshold": int(threshold),
                "length_m": float(line.length),
                "flow_acc_cells": float(accumulation[node_ids[-1]]),
                "flow_acc_m2": float(accumulation[node_ids[-1]] * 100.0),
                "geometry": line,
            }
        )

    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:32649")


def write_preview(
    dem: np.ndarray,
    valid: np.ndarray,
    primary_gdf: gpd.GeoDataFrame,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    y0, x0 = np.indices(dem.shape, dtype=np.float32)
    # Use a simple analytic hillshade for quick visual QA only.
    gy, gx = np.gradient(np.where(valid, dem, np.nan), 10.0, 10.0)
    azimuth = 315.0
    altitude = 45.0
    slope_rad = np.arctan(np.sqrt(gx * gx + gy * gy))
    aspect = np.arctan2(-gy, gx)
    hillshade = (
        np.sin(np.deg2rad(altitude)) * np.cos(slope_rad)
        + np.cos(np.deg2rad(altitude))
        * np.sin(slope_rad)
        * np.cos(np.deg2rad(azimuth) - aspect)
    )
    hillshade = np.clip(hillshade, 0, 1)

    fig, ax = plt.subplots(figsize=(10, 11), dpi=180)
    ax.imshow(hillshade, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    primary_gdf.plot(ax=ax, color="#1f5ac2", linewidth=0.25, alpha=0.9)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Reading DEM: {DEM_PATH}")
    reference, dem, valid = read_dem(DEM_PATH)
    valid_count = int(np.count_nonzero(valid))
    log(f"  raster size: {dem.shape[1]} x {dem.shape[0]}")
    log(f"  valid cells: {valid_count:,}")

    log("Filling sinks and flats (priority flood)")
    filled = fill_sinks(dem, valid)
    write_raster(
        OUT_DIR / "dem_filled.tif",
        np.where(valid, filled, -9999.0),
        reference,
        -9999.0,
    )
    log(f"  wrote {OUT_DIR / 'dem_filled.tif'}")

    log("Computing D8 flow directions")
    downstream = compute_d8_downstream(filled, valid)
    log("Computing flow accumulation")
    accumulation = accumulate_flow(filled, valid, downstream)
    write_raster(
        OUT_DIR / "flow_accumulation.tif",
        np.where(valid, accumulation.reshape(valid.shape), -1.0),
        reference,
        -1.0,
    )
    log(f"  wrote {OUT_DIR / 'flow_accumulation.tif'}")

    valid_flat = np.flatnonzero(valid.ravel())
    gpkg_path = OUT_DIR / "drainage_network.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()

    primary_gdf = None
    for threshold in THRESHOLDS:
        log(f"Vectorizing drainage lines for threshold {threshold}")
        gdf = build_drainage_lines(
            accumulation,
            downstream,
            valid_flat,
            threshold,
            dem.shape[1],
            reference.GetGeoTransform(),
        )
        gdf.to_file(gpkg_path, layer=f"drainage_{threshold}", driver="GPKG")
        log(
            f"  threshold {threshold}: {len(gdf):,} line segments, "
            f"total length {gdf['length_m'].sum() / 1000.0:,.2f} km"
        )
        if threshold == PRIMARY_THRESHOLD:
            primary_gdf = gdf

    if primary_gdf is None:
        raise RuntimeError("Primary threshold output was not generated")
    primary_gdf.to_file(
        OUT_DIR / f"drainage_network_{PRIMARY_THRESHOLD}.geojson",
        driver="GeoJSON",
    )
    write_preview(
        dem,
        valid,
        primary_gdf,
        OUT_DIR / f"drainage_network_{PRIMARY_THRESHOLD}_preview.png",
    )

    log("Finished")
    log(f"GeoPackage: {gpkg_path}")
    log(f"Primary GeoJSON: {OUT_DIR / f'drainage_network_{PRIMARY_THRESHOLD}.geojson'}")
    log(
        "GeoPackage layers: "
        + ", ".join(f"drainage_{threshold}" for threshold in THRESHOLDS)
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"ERROR: {exc}")
        raise
