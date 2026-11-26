from pathlib import Path
import csv
import math

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from scipy import ndimage
from shapely.geometry import shape


OUT = Path(r"D:\project\shuozhouinflucefactor\process\dem")
DEM_PATH = OUT / "dem_10m_hydro.tif"
FDIR_PATH = OUT / "flow_direction_saga.tif"
ACC_PATH = OUT / "flow_accumulation_m2.tif"
SLOPE_PATH = OUT / "slope_deg.tif"
ASPECT_PATH = OUT / "aspect_deg.tif"
PCURV_PATH = OUT / "profile_curvature.tif"
PLCURV_PATH = OUT / "plan_curvature.tif"
TWI_PATH = OUT / "twi.tif"
RELIEF_PATH = OUT / "local_relief_300m.tif"
TPI_PATH = OUT / "tpi_300m.tif"
TRI_PATH = OUT / "tri_3x3.tif"
DIST_DRAIN_PATH = OUT / "distance_to_drainage_m.tif"
BOUNDARY_PATH = OUT / "analysis_boundaries.gpkg"

# 10 m cells: 10,000 / 30,000 / 50,000 m2 correspond to 100 / 300 / 500 cells.
SCALES = {
    "fine": {"channel_area_m2": 10_000.0, "min_unit_cells": 100},
    "medium": {"channel_area_m2": 30_000.0, "min_unit_cells": 300},
    "coarse": {"channel_area_m2": 50_000.0, "min_unit_cells": 500},
}

# SAGA direction convention: N, NE, E, SE, S, SW, W, NW.
DR = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int8)
DC = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int8)


def read_grid(path):
    with rasterio.open(path) as src:
        a = src.read(1, masked=True)
        return a.filled(np.nan).astype(np.float64), src.profile.copy(), src.transform, src.crs


def build_downstream(fdir, valid):
    nrow, ncol = fdir.shape
    rr, cc = np.indices(fdir.shape)
    downstream = np.full(fdir.size, -1, dtype=np.int64)
    good = valid & np.isfinite(fdir) & (fdir >= 0) & (fdir <= 7)
    k = fdir[good].astype(np.int8)
    r2 = rr[good] + DR[k]
    c2 = cc[good] + DC[k]
    inside = (r2 >= 0) & (r2 < nrow) & (c2 >= 0) & (c2 < ncol)
    src_flat = np.flatnonzero(good)[inside]
    dst_flat = r2[inside] * ncol + c2[inside]
    dst_valid = valid.ravel()[dst_flat]
    downstream[src_flat[dst_valid]] = dst_flat[dst_valid]
    return downstream


def segment_channels(channel, downstream, acc):
    ch = channel.ravel()
    target = downstream.copy()
    links = ch & (target >= 0)
    links[links] &= ch[target[links]]

    indegree = np.zeros(ch.size, dtype=np.int16)
    np.add.at(indegree, target[links], 1)
    sole_up = np.full(ch.size, -1, dtype=np.int64)
    sole_up[target[links]] = np.flatnonzero(links)

    seg = np.full(ch.size, -1, dtype=np.int32)
    next_id = 0
    order = np.flatnonzero(ch)
    order = order[np.argsort(acc.ravel()[order], kind="stable")]
    for idx in order:
        if indegree[idx] == 1 and seg[sole_up[idx]] >= 0:
            seg[idx] = seg[sole_up[idx]]
        else:
            next_id += 1
            seg[idx] = next_id
    return seg, indegree, next_id


def assign_hillslopes(valid, channel, downstream, acc, channel_seg, n_segments):
    nrow, ncol = valid.shape
    flat_valid = valid.ravel()
    ch = channel.ravel()
    entry = np.full(flat_valid.size, -1, dtype=np.int64)
    entry[ch] = np.flatnonzero(ch)

    order = np.flatnonzero(flat_valid)
    order = order[np.argsort(acc.ravel()[order], kind="stable")][::-1]
    for idx in order:
        if entry[idx] >= 0:
            continue
        dst = downstream[idx]
        if dst >= 0:
            entry[idx] = entry[dst]

    seg = np.full(flat_valid.size, -1, dtype=np.int32)
    has_entry = entry >= 0
    seg[has_entry] = channel_seg[entry[has_entry]]

    # Small drainage systems which never reach the channel threshold are
    # assigned by their own outlet. This avoids joining neighbouring edge
    # catchments into one artificial oversized polygon.
    for _ in range(3):
        changed = 0
        for idx in order:
            if not flat_valid[idx] or seg[idx] >= 0:
                continue
            dst = downstream[idx]
            if dst >= 0 and seg[dst] >= 0:
                seg[idx] = seg[dst]
                changed += 1
            elif dst < 0:
                n_segments += 1
                seg[idx] = n_segments
                changed += 1
        if changed == 0:
            break

    # Extremely rare unresolved flat cells are retained as small units.
    unresolved = (flat_valid & (seg < 0)).reshape(valid.shape)
    if unresolved.any():
        labels, nlab = ndimage.label(unresolved, structure=np.ones((3, 3), dtype=np.uint8))
        for lab in range(1, nlab + 1):
            n_segments += 1
            seg[labels.ravel() == lab] = n_segments

    side = np.ones(flat_valid.size, dtype=np.int8)
    ids = np.flatnonzero(has_entry & ~ch)
    ent = entry[ids]
    ent_dst = downstream[ent]

    # Local channel tangent. At outlets, use the reverse direction of the
    # nearest upstream channel cell when available.
    er, ec = np.divmod(ent, ncol)
    tr = np.zeros(ids.size, dtype=np.int16)
    tc = np.ones(ids.size, dtype=np.int16)
    ok = (ent_dst >= 0) & ch[np.maximum(ent_dst, 0)]
    if ok.any():
        dr2, dc2 = np.divmod(ent_dst[ok], ncol)
        tr[ok] = dr2 - er[ok]
        tc[ok] = dc2 - ec[ok]

    ir, ic = np.divmod(ids, ncol)
    tx = tc.astype(np.float64)
    ty = -tr.astype(np.float64)
    vx = (ic - ec).astype(np.float64)
    vy = -(ir - er).astype(np.float64)
    cross = tx * vy - ty * vx
    side[ids] = np.where(cross < 0, 2, 1)

    raw = np.zeros(flat_valid.size, dtype=np.int64)
    raw[flat_valid] = seg[flat_valid].astype(np.int64) * 2 + side[flat_valid]
    return raw.reshape(valid.shape)


def connected_relabel(raw, valid):
    labels = np.zeros(raw.shape, dtype=np.int32)
    next_id = 0
    for value in np.unique(raw[valid]):
        if value <= 0:
            continue
        part, n = ndimage.label(raw == value, structure=np.ones((3, 3), dtype=np.uint8))
        for j in range(1, n + 1):
            next_id += 1
            labels[part == j] = next_id
    return labels


def merge_small(labels, min_cells, valid):
    labels = labels.copy()
    nrow, ncol = labels.shape
    for _ in range(3):
        counts = np.bincount(labels.ravel())
        small = np.flatnonzero((counts > 0) & (counts < min_cells))
        small = small[small != 0]
        if small.size == 0:
            break
        small = small[np.argsort(counts[small])]
        changed = 0
        for lab in small:
            pix = np.flatnonzero(labels.ravel() == lab)
            if pix.size == 0:
                continue
            r, c = np.divmod(pix, ncol)
            neighbours = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                ok = (rr >= 0) & (rr < nrow) & (cc >= 0) & (cc < ncol)
                vals = labels[rr[ok], cc[ok]]
                vals = vals[(vals > 0) & (vals != lab)]
                if vals.size:
                    neighbours.append(vals)
            if neighbours:
                vals = np.concatenate(neighbours)
                target = int(np.bincount(vals).argmax())
                labels.ravel()[pix] = target
                changed += 1
        if changed == 0:
            break

    unique = np.unique(labels[valid])
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lut[unique] = np.arange(1, unique.size + 1, dtype=np.int32)
    return lut[labels]


def write_raster(path, data, profile):
    p = profile.copy()
    p.update(dtype="int32", count=1, nodata=0, compress="deflate", predictor=2, tiled=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(data.astype(np.int32), 1)


def circular_aspect(values):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    rad = np.deg2rad(values)
    s, c = np.mean(np.sin(rad)), np.mean(np.cos(rad))
    mean = (np.rad2deg(np.arctan2(s, c)) + 360.0) % 360.0
    return float(mean), float(np.hypot(s, c))


def polygonize_and_stats(labels, transform, crs, arrays, scale_name):
    records = []
    for geom, value in features.shapes(labels, mask=labels > 0, transform=transform):
        records.append({"SU_ID": int(value), "geometry": shape(geom)})
    gdf = gpd.GeoDataFrame(records, crs=crs)
    if gdf.empty:
        return gdf
    gdf = gdf.dissolve(by="SU_ID", as_index=False)
    gdf["scale"] = scale_name

    stats = []
    for su_id in gdf["SU_ID"].to_numpy():
        m = labels == su_id
        elev = arrays["elev"][m]
        slope = arrays["slope"][m]
        pcurv = arrays["pcurv"][m]
        plcurv = arrays["plcurv"][m]
        twi = arrays["twi"][m]
        relief = arrays["relief"][m]
        tpi = arrays["tpi"][m]
        tri = arrays["tri"][m]
        dist_drain = arrays["dist_drain"][m]
        asp_mean, asp_r = circular_aspect(arrays["aspect"][m])
        stats.append(
            {
                "SU_ID": int(su_id),
                "cell_n": int(m.sum()),
                "elev_mean": float(np.nanmean(elev)),
                "elev_min": float(np.nanmin(elev)),
                "elev_max": float(np.nanmax(elev)),
                "slope_mean": float(np.nanmean(slope)),
                "slope_std": float(np.nanstd(slope)),
                "aspect_mean": asp_mean,
                "aspect_R": asp_r,
                "pcurv_mean": float(np.nanmean(pcurv)),
                "plcurv_mean": float(np.nanmean(plcurv)),
                "twi_mean": float(np.nanmean(twi)),
                "relief_mean": float(np.nanmean(relief)),
                "tpi_mean": float(np.nanmean(tpi)),
                "tri_mean": float(np.nanmean(tri)),
                "drain_dist_mean": float(np.nanmean(dist_drain)),
            }
        )
    stat_df = pd.DataFrame(stats)
    gdf = gdf.merge(stat_df, on="SU_ID", how="left")
    gdf["area_km2"] = gdf.geometry.area / 1e6
    gdf["perim_km"] = gdf.geometry.length / 1000.0
    gdf["compact"] = 4.0 * math.pi * (gdf.geometry.area) / np.maximum(gdf.geometry.length**2, 1e-9)
    gdf["unit_type"] = np.select(
        [gdf["slope_mean"] < 5.0, gdf["aspect_R"] < 0.30],
        ["flat_or_gentle", "mixed_aspect"],
        default="slope",
    )
    gdf["use_model"] = (gdf["unit_type"] == "slope").astype(np.int16)
    return gdf


def main():
    # SAGA's binary grid driver preserves the transform but not CRS metadata
    # when converted back to GeoTIFF. Restore the known source CRS.
    terrain_grids = [
        DEM_PATH, FDIR_PATH, ACC_PATH, SLOPE_PATH, ASPECT_PATH,
        PCURV_PATH, PLCURV_PATH, TWI_PATH, OUT / "northness.tif",
        OUT / "eastness.tif", OUT / "hillshade.tif",
    ]
    for path in terrain_grids:
        with rasterio.open(path, "r+") as ds:
            if ds.crs is None:
                ds.crs = "EPSG:32649"

    elev, profile, transform, crs = read_grid(DEM_PATH)
    fdir, _, _, _ = read_grid(FDIR_PATH)
    acc, _, _, _ = read_grid(ACC_PATH)
    slope, _, _, _ = read_grid(SLOPE_PATH)
    aspect, _, _, _ = read_grid(ASPECT_PATH)
    pcurv, _, _, _ = read_grid(PCURV_PATH)
    plcurv, _, _, _ = read_grid(PLCURV_PATH)
    twi, _, _, _ = read_grid(TWI_PATH)
    relief, _, _, _ = read_grid(RELIEF_PATH)
    tpi, _, _, _ = read_grid(TPI_PATH)
    tri, _, _, _ = read_grid(TRI_PATH)
    dist_drain, _, _, _ = read_grid(DIST_DRAIN_PATH)
    valid = np.isfinite(elev)
    downstream = build_downstream(fdir, valid)
    study_geom = gpd.read_file(BOUNDARY_PATH, layer="study_area").geometry.union_all()

    arrays = {
        "elev": elev, "slope": slope, "aspect": aspect,
        "pcurv": pcurv, "plcurv": plcurv, "twi": twi,
        "relief": relief, "tpi": tpi, "tri": tri, "dist_drain": dist_drain,
    }
    gpkg = OUT / "slope_units_multiscale.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    comparison = []
    all_results = {}

    for scale_name, pars in SCALES.items():
        channel = valid & (acc >= pars["channel_area_m2"])
        ch_seg, _, nseg = segment_channels(channel, downstream, acc)
        raw = assign_hillslopes(valid, channel, downstream, acc, ch_seg, nseg)
        labels = connected_relabel(raw, valid)
        labels = merge_small(labels, pars["min_unit_cells"], valid)
        raster_path = OUT / f"slope_units_{scale_name}.tif"
        write_raster(raster_path, labels, profile)

        gdf = polygonize_and_stats(labels, transform, crs, arrays, scale_name)
        gdf["intersects"] = gdf.intersects(study_geom).astype(np.int16)
        gdf.to_file(gpkg, layer=f"{scale_name}_all", driver="GPKG")
        selected = gdf[gdf["intersects"] == 1].copy()
        selected.to_file(gpkg, layer=f"{scale_name}_study_full", driver="GPKG")
        all_results[scale_name] = (labels, gdf, selected)

        comparison.append(
            {
                "scale": scale_name,
                "channel_area_m2": pars["channel_area_m2"],
                "min_unit_cells": pars["min_unit_cells"],
                "all_unit_n": len(gdf),
                "study_unit_n": len(selected),
                "study_median_area_km2": float(selected.area_km2.median()),
                "study_mean_area_km2": float(selected.area_km2.mean()),
                "study_min_area_km2": float(selected.area_km2.min()),
                "study_max_area_km2": float(selected.area_km2.max()),
                "study_median_aspect_R": float(selected.aspect_R.median()),
            }
        )

    # Medium scale is the recommended main product.
    labels, _, selected = all_results["medium"]
    main_full = OUT / "slope_units_main_full.gpkg"
    main_clip = OUT / "slope_units_main_clipped.gpkg"
    for path in (main_full, main_clip):
        if path.exists():
            path.unlink()
    selected.to_file(main_full, layer="slope_units", driver="GPKG")

    clipped = gpd.clip(selected, gpd.GeoDataFrame(geometry=[study_geom], crs=crs))
    clipped["clip_area"] = clipped.geometry.area / 1e6
    clipped.to_file(main_clip, layer="slope_units", driver="GPKG")

    # Shapefile field names are shortened for ArcGIS compatibility.
    shp = selected.rename(
        columns={
            "elev_mean": "elev_mean",
            "aspect_mean": "asp_mean",
            "pcurv_mean": "pcurv_m",
            "plcurv_mean": "plcurv_m",
            "slope_mean": "slope_m",
            "slope_std": "slope_sd",
            "relief_mean": "relf_mean",
            "drain_dist_mean": "drain_m",
            "area_km2": "area_km2",
            "perim_km": "perim_km",
        }
    )
    shp.to_file(OUT / "slope_units_main_full.shp", driver="ESRI Shapefile", encoding="UTF-8")

    with open(OUT / "slope_unit_scale_comparison.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=comparison[0].keys())
        writer.writeheader()
        writer.writerows(comparison)

    # Quick-look image for visual QA.
    hill, _, hill_transform, _ = read_grid(OUT / "hillshade.tif")
    extent = [
        hill_transform.c,
        hill_transform.c + hill.shape[1] * hill_transform.a,
        hill_transform.f + hill.shape[0] * hill_transform.e,
        hill_transform.f,
    ]
    fig, ax = plt.subplots(figsize=(9, 10), dpi=180)
    lo, hi = np.nanpercentile(hill, [2, 98])
    ax.imshow(hill, cmap="gray", extent=extent, origin="upper", vmin=lo, vmax=hi)
    selected.boundary.plot(ax=ax, linewidth=0.35, color="#e66101")
    gpd.GeoSeries([study_geom], crs=crs).boundary.plot(ax=ax, linewidth=1.2, color="#1f78b4")
    ax.set_title("Medium-scale slope units (orange) and study area (blue)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "slope_units_main_preview.png", bbox_inches="tight")
    plt.close(fig)

    for row in comparison:
        print(row)


if __name__ == "__main__":
    main()
