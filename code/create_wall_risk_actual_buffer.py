"""Create the polygon display layer matching the risk model's actual buffer."""

from pathlib import Path

from osgeo import gdal, ogr


SOURCE = Path(
    r"D:\shuozhou\result\风险评价\result\guangwu_wall_50m_geohazard_risk_100m_buffer.gpkg"
)
OUTPUT = Path(
    r"D:\project\shuozhouinflucefactor\process\qgis\guangwu_wall_risk_actual_100m_buffer.gpkg"
)
SOURCE_LAYER = "wall_segment_risk"
OUTPUT_LAYER = "wall_risk_buffer_100m"

# Verified in evaluate_wall_100m_segment_risk.py.
SEGMENT_LENGTH_M = 50.0
BUFFER_DISTANCE_M = 100.0


def main() -> None:
    gdal.UseExceptions()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    source_ds = gdal.OpenEx(str(SOURCE), gdal.OF_VECTOR)
    if source_ds is None:
        raise FileNotFoundError(SOURCE)
    source_layer = source_ds.GetLayerByName(SOURCE_LAYER)
    if source_layer is None:
        raise RuntimeError(f"Layer not found: {SOURCE_LAYER}")

    if OUTPUT.exists():
        ogr.GetDriverByName("GPKG").DeleteDataSource(str(OUTPUT))
    output_ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(OUTPUT))
    output_layer = output_ds.CreateLayer(
        OUTPUT_LAYER,
        srs=source_layer.GetSpatialRef(),
        geom_type=ogr.wkbPolygon,
        options=["GEOMETRY_NAME=geom", "SPATIAL_INDEX=YES"],
    )

    source_defn = source_layer.GetLayerDefn()
    for index in range(source_defn.GetFieldCount()):
        output_layer.CreateField(source_defn.GetFieldDefn(index))
    output_layer.CreateField(ogr.FieldDefn("segment_m", ogr.OFTReal))
    output_layer.CreateField(ogr.FieldDefn("buffer_m", ogr.OFTReal))
    output_layer.CreateField(ogr.FieldDefn("display_width_m", ogr.OFTReal))

    output_defn = output_layer.GetLayerDefn()
    source_layer.ResetReading()
    feature_count = 0
    for source_feature in source_layer:
        line = source_feature.GetGeometryRef()
        if line is None or line.IsEmpty():
            continue
        polygon = line.Buffer(BUFFER_DISTANCE_M, 16)
        if polygon is None or polygon.IsEmpty():
            continue

        output_feature = ogr.Feature(output_defn)
        for index in range(source_defn.GetFieldCount()):
            output_feature.SetField(index, source_feature.GetField(index))
        output_feature.SetField("segment_m", SEGMENT_LENGTH_M)
        output_feature.SetField("buffer_m", BUFFER_DISTANCE_M)
        output_feature.SetField("display_width_m", BUFFER_DISTANCE_M * 2.0)
        output_feature.SetGeometry(polygon)
        output_layer.CreateFeature(output_feature)
        output_feature = None
        feature_count += 1

    output_layer.SyncToDisk()
    output_ds = None
    source_ds = None

    check_ds = gdal.OpenEx(str(OUTPUT), gdal.OF_VECTOR)
    check_layer = check_ds.GetLayerByName(OUTPUT_LAYER)
    print(f"output={OUTPUT}")
    print(f"layer={OUTPUT_LAYER}")
    print(f"features={check_layer.GetFeatureCount()}")
    print(f"geometry={ogr.GeometryTypeToName(check_layer.GetLayerDefn().GetGeomType())}")
    print(f"extent={check_layer.GetExtent()}")
    print(f"created={feature_count}")
    check_ds = None


if __name__ == "__main__":
    main()
