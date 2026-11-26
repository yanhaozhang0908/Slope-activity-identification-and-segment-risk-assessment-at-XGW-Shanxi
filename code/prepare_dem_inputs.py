from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.mask import mask


DEM = Path(r"D:\project\shuozhouinflucefactor\data\gf7dem\gf7dem_10m.tif")
STUDY = Path(r"D:\shuozhou\data\shapefile\guangwu\buffer1000m\buffer_1000m_all.shp")
OUT = Path(r"D:\project\shuozhouinflucefactor\process\dem")
CRS = "EPSG:32649"
ANALYSIS_MARGIN_M = 2000


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    study = gpd.read_file(STUDY).to_crs(CRS)
    study_geom = study.geometry.union_all()
    processing_geom = study_geom.buffer(ANALYSIS_MARGIN_M)

    boundary_file = OUT / "analysis_boundaries.gpkg"
    gpd.GeoDataFrame(
        {"name": ["study_area_1km_buffer"]}, geometry=[study_geom], crs=CRS
    ).to_file(boundary_file, layer="study_area", driver="GPKG")
    gpd.GeoDataFrame(
        {"name": ["study_area_plus_2km_margin"]}, geometry=[processing_geom], crs=CRS
    ).to_file(boundary_file, layer="processing_extent", driver="GPKG")

    out_dem = OUT / "dem_10m_analysis_extent.tif"
    with rasterio.open(DEM) as src:
        data, transform = mask(
            src,
            [processing_geom.__geo_interface__],
            crop=True,
            all_touched=True,
            nodata=-9999.0,
        )
        profile = src.profile.copy()
        profile.update(
            height=data.shape[1],
            width=data.shape[2],
            transform=transform,
            nodata=-9999.0,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        with rasterio.open(out_dem, "w", **profile) as dst:
            dst.write(data)

    print(f"study_area_km2={study_geom.area / 1e6:.6f}")
    print(f"processing_area_km2={processing_geom.area / 1e6:.6f}")
    print(f"dem={out_dem}")


if __name__ == "__main__":
    main()
