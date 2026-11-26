"""Extract the Topographic Wetness Index (TWI) for the full GF-7 DEM coverage."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from scipy import ndimage

gdal.UseExceptions()

DEM_PATH = Path(
    r"D:\project\shuozhouinflucefactor\outputs\drainage_network\dem_filled.tif"
)
ACC_PATH = Path(
    r"D:\project\shuozhouinflucefactor\outputs\drainage_network\flow_accumulation.tif"
)
OUT_DIR = Path(r"D:\project\shuozhouinflucefactor\outputs\twi_dem_coverage")
TWI_PATH = OUT_DIR / "twi_dem_coverage.tif"
PREVIEW_PATH = OUT_DIR / "twi_dem_coverage_preview.png"
NODATA = -9999.0
CELL_SIZE = 10.0


def read_raster(path: Path):
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open raster: {path}")
    band = ds.GetRasterBand(1)
    array = band.ReadAsArray().astype(np.float64)
    nodata = band.GetNoDataValue()
    valid = np.isfinite(array)
    if nodata is not None:
        valid &= array != float(nodata)
    return ds, array, valid


def write_twi(path: Path, array: np.ndarray, valid: np.ndarray, reference: gdal.Dataset):
    driver = gdal.GetDriverByName("GTiff")
    options = [
        "TILED=YES",
        "COMPRESS=DEFLATE",
        "PREDICTOR=3",
        "BIGTIFF=IF_SAFER",
    ]
    out = driver.Create(
        str(path), array.shape[1], array.shape[0], 1, gdal.GDT_Float32, options=options
    )
    out.SetGeoTransform(reference.GetGeoTransform())
    out.SetProjection(reference.GetProjection())
    band = out.GetRasterBand(1)
    band.SetNoDataValue(float(NODATA))
    band.WriteArray(np.where(valid, array, NODATA).astype(np.float32))
    band.FlushCache()
    out.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32])
    out = None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dem_ds, dem, dem_valid = read_raster(DEM_PATH)
    acc_ds, accumulation, acc_valid = read_raster(ACC_PATH)
    if dem.shape != accumulation.shape:
        raise RuntimeError("DEM and flow accumulation rasters have different shapes")
    valid = dem_valid & acc_valid
    if not valid.any():
        raise RuntimeError("No overlapping valid cells")

    # Fill NoData with nearest valid elevations so slope can be computed everywhere.
    nearest = ndimage.distance_transform_edt(
        ~dem_valid, return_distances=False, return_indices=True
    )
    filled_dem = dem[tuple(nearest)]

    # Slope in radians from central differences on the filled DEM.
    gy, gx = np.gradient(filled_dem, CELL_SIZE, CELL_SIZE)
    slope_rad = np.arctan(np.sqrt(gx * gx + gy * gy))

    # Specific catchment area: upstream cell count * cell area / cell width.
    upstream_area = accumulation * CELL_SIZE * CELL_SIZE
    specific_catchment_area = upstream_area / CELL_SIZE

    with np.errstate(divide="ignore", invalid="ignore"):
        slope_tangent = np.tan(slope_rad)
        slope_tangent = np.where(slope_tangent > 1.0e-8, slope_tangent, 1.0e-8)
        twi = np.log(specific_catchment_area / slope_tangent)
    twi[~valid] = NODATA

    write_twi(TWI_PATH, twi, valid, dem_ds)

    # Preview with a wetness-oriented blue ramp.
    min_value = float(np.nanmin(twi[valid]))
    max_value = float(np.nanmax(twi[valid]))
    fig, ax = plt.subplots(figsize=(9, 10), dpi=180)
    image = ax.imshow(
        np.ma.masked_invalid(np.where(valid, twi, np.nan)),
        cmap="YlGnBu",
        vmin=min_value,
        vmax=max_value,
        interpolation="nearest",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("TWI")
    ax.set_axis_off()
    ax.set_title("Topographic Wetness Index")
    fig.tight_layout()
    fig.savefig(PREVIEW_PATH, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)

    print(f"output={TWI_PATH}")
    print(f"preview={PREVIEW_PATH}")
    print(f"valid_cells={int(valid.sum())}")
    print(f"twi_min={min_value:.4f}")
    print(f"twi_max={max_value:.4f}")
    print(f"twi_mean={float(np.nanmean(twi[valid])):.4f}")


if __name__ == "__main__":
    main()
