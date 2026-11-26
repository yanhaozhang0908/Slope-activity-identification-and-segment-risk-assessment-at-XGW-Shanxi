from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage


OUT = Path(r"D:\project\shuozhouinflucefactor\process\dem")
DEM = OUT / "dem_10m_analysis_extent.tif"
ACC = OUT / "flow_accumulation_m2.tif"


def write_grid(path, array, profile, dtype="float32", nodata=-9999.0):
    p = profile.copy()
    p.update(
        dtype=dtype,
        count=1,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        predictor=3 if dtype == "float32" else 2,
    )
    with rasterio.open(path, "w", **p) as dst:
        dst.write(array.astype(dtype), 1)


def main():
    with rasterio.open(DEM) as src:
        dem_ma = src.read(1, masked=True)
        profile = src.profile.copy()
    with rasterio.open(ACC) as src:
        acc_ma = src.read(1, masked=True)

    valid = ~np.ma.getmaskarray(dem_ma)
    dem = dem_ma.filled(np.nan).astype(np.float64)
    acc = acc_ma.filled(np.nan).astype(np.float64)

    # Nearest-valid padding prevents NoData margins from contaminating focal statistics.
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = dem[tuple(nearest)]

    radius_cells = 30  # 300 m radius at 10 m resolution
    size = radius_cells * 2 + 1
    local_max = ndimage.maximum_filter(filled, size=size, mode="nearest")
    local_min = ndimage.minimum_filter(filled, size=size, mode="nearest")
    relief = local_max - local_min
    tpi = filled - ndimage.uniform_filter(filled, size=size, mode="nearest")

    tri_sq = np.zeros_like(filled, dtype=np.float64)
    for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        shifted = ndimage.shift(filled, shift=(dr, dc), order=0, mode="nearest", prefilter=False)
        tri_sq += (filled - shifted) ** 2
    tri = np.sqrt(tri_sq)

    channel = valid & np.isfinite(acc) & (acc >= 30_000.0)
    distance = ndimage.distance_transform_edt(~channel, sampling=(10.0, 10.0))

    for array in (relief, tpi, tri, distance):
        array[~valid] = -9999.0
    channel_out = np.where(valid, channel.astype(np.uint8), 255).astype(np.uint8)

    write_grid(OUT / "local_relief_300m.tif", relief, profile)
    write_grid(OUT / "tpi_300m.tif", tpi, profile)
    write_grid(OUT / "tri_3x3.tif", tri, profile)
    write_grid(OUT / "distance_to_drainage_m.tif", distance, profile)
    write_grid(OUT / "drainage_network_main.tif", channel_out, profile, dtype="uint8", nodata=255)

    print(f"valid_cells={valid.sum()}")
    print(f"channel_cells={channel.sum()}")
    print(f"relief_range={np.nanmin(relief[valid]):.3f},{np.nanmax(relief[valid]):.3f}")
    print(f"distance_max_m={np.nanmax(distance[valid]):.3f}")


if __name__ == "__main__":
    main()
