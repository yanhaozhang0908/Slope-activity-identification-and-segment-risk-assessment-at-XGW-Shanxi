"""Create a QGIS-ready hillshade GeoTIFF from the project DEM."""

from pathlib import Path

from osgeo import gdal


SOURCE = Path(r"D:\project\shuozhouinflucefactor\process\dem\dem_10m_analysis_extent.tif")
OUTPUT = Path(r"D:\project\shuozhouinflucefactor\process\dem\dem_10m_hillshade_az315_alt42.tif")


def main() -> None:
    gdal.UseExceptions()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    source_ds = gdal.Open(str(SOURCE), gdal.GA_ReadOnly)
    if source_ds is None:
        raise FileNotFoundError(SOURCE)

    options = gdal.DEMProcessingOptions(
        format="GTiff",
        azimuth=315.0,
        altitude=42.0,
        zFactor=1.15,
        computeEdges=True,
        creationOptions=[
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
            "BIGTIFF=IF_SAFER",
        ],
    )
    hillshade_ds = gdal.DEMProcessing(
        str(OUTPUT), source_ds, "hillshade", options=options
    )
    if hillshade_ds is None:
        raise RuntimeError("Hillshade generation failed")
    hillshade_ds.FlushCache()
    hillshade_ds = None
    source_ds = None

    output_ds = gdal.Open(str(OUTPUT), gdal.GA_Update)
    output_ds.GetRasterBand(1).ComputeStatistics(False)
    output_ds.BuildOverviews("AVERAGE", [2, 4, 8, 16])
    output_ds.FlushCache()
    output_ds = None

    check_ds = gdal.Open(str(OUTPUT), gdal.GA_ReadOnly)
    band = check_ds.GetRasterBand(1)
    print(f"output={OUTPUT}")
    print(f"size={check_ds.RasterXSize}x{check_ds.RasterYSize}")
    print(f"geotransform={check_ds.GetGeoTransform()}")
    print(f"projection={check_ds.GetProjectionRef()}")
    print(f"datatype={gdal.GetDataTypeName(band.DataType)}")
    print(f"nodata={band.GetNoDataValue()}")
    print(f"statistics={band.GetStatistics(False, True)}")
    print(f"overviews={band.GetOverviewCount()}")
    check_ds = None


if __name__ == "__main__":
    main()
