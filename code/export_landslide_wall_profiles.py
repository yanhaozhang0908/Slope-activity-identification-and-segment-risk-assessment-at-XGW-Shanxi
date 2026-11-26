"""Export the mapped shortest connections and DEM elevation profiles to GeoPackage."""
from pathlib import Path
import csv
import json
import math

import numpy as np
from osgeo import gdal, ogr, osr

from analyze_landslide_wall_impact import ROOT, DEM_PATH, sample_array


def main():
    gdal.UseExceptions()
    ogr.UseExceptions()
    out = ROOT / "outputs" / "landslide_wall_profiles"
    out.mkdir(parents=True, exist_ok=True)
    gpkg = out / "landslide_wall_profiles.gpkg"
    if gpkg.exists():
        raise FileExistsError(gpkg)
    records = json.loads((ROOT / "process/landslide_wall_impact/landslide_wall_impact_metrics.json").read_text(encoding="utf-8"))
    dem_ds = gdal.Open(str(DEM_PATH))
    gt = dem_ds.GetGeoTransform()
    dem = dem_ds.GetRasterBand(1).ReadAsArray().astype(float)
    nodata = dem_ds.GetRasterBand(1).GetNoDataValue()
    if nodata is not None:
        dem[np.isclose(dem, nodata)] = np.nan
    srs = osr.SpatialReference()
    srs.ImportFromWkt(dem_ds.GetProjection())
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(gpkg))

    def layer(name, geom_type, fields):
        result = ds.CreateLayer(name, srs, geom_type)
        for key, kind in fields:
            result.CreateField(ogr.FieldDefn(key, kind))
        return result

    common = [("zone", ogr.OFTInteger), ("zone_name", ogr.OFTString)]
    lines = layer("shortest_connections_3d", ogr.wkbLineString25D, common + [
        ("length_m", ogr.OFTReal), ("start_z_m", ogr.OFTReal),
        ("end_z_m", ogr.OFTReal), ("rise_m", ogr.OFTReal),
        ("sample_count", ogr.OFTInteger), ("dem_res_m", ogr.OFTReal)])
    fields = common + [("sample_id", ogr.OFTInteger), ("distance_m", ogr.OFTReal),
                       ("elevation_m", ogr.OFTReal), ("x_m", ogr.OFTReal),
                       ("y_m", ogr.OFTReal), ("position", ogr.OFTString)]
    points = layer("elevation_samples_5m", ogr.wkbPoint25D, fields)
    endpoints = layer("profile_endpoints", ogr.wkbPoint25D, fields)

    def add(target, geometry, attributes):
        feature = ogr.Feature(target.GetLayerDefn())
        for key, value in attributes.items():
            feature.SetField(key, value)
        feature.SetGeometry(geometry)
        target.CreateFeature(feature)

    rows = []
    summaries = []
    for record in records:
        zone = int(record["name"][-1])
        start = np.array([record["nearest_slide_x"], record["nearest_slide_y"]])
        end = np.array([record["nearest_wall_x"], record["nearest_wall_y"]])
        length = float(np.linalg.norm(end - start))
        assert abs(length - record["min_distance_to_wall_m"]) < 1e-6
        distances = np.linspace(0, length, math.ceil(length / 5) + 1)
        coords = start + distances[:, None] / length * (end - start)
        elevations = [float(sample_array(dem, x, y, gt)) for x, y in coords]
        if not np.isfinite(elevations).all():
            raise ValueError(f"Missing DEM elevations in zone {zone}")
        geometry = ogr.Geometry(ogr.wkbLineString25D)
        zone_rows = []
        for i, ((x, y), distance, z) in enumerate(zip(coords, distances, elevations)):
            geometry.AddPoint(float(x), float(y), z)
            position = "landslide_boundary" if i == 0 else "wall_line" if i == len(distances) - 1 else "interior"
            row = dict(zone=zone, zone_name=record["name"], sample_id=i,
                       distance_m=float(distance), elevation_m=z,
                       x_m=float(x), y_m=float(y), position=position)
            point = ogr.Geometry(ogr.wkbPoint25D)
            point.AddPoint(float(x), float(y), z)
            add(points, point, row)
            if position != "interior":
                add(endpoints, point, row)
            zone_rows.append(row)
        summary = dict(zone=zone, zone_name=record["name"], length_m=length,
                       start_z_m=elevations[0], end_z_m=elevations[-1],
                       rise_m=elevations[-1] - elevations[0],
                       sample_count=len(distances), dem_res_m=abs(gt[1]))
        add(lines, geometry, summary)
        summaries.append(summary)
        rows.extend(zone_rows)
        write_csv(out / f"zone{zone}_elevation_profile.csv", zone_rows)
    write_csv(out / "all_elevation_samples.csv", rows)
    write_csv(out / "profile_summary.csv", summaries)
    ds = None
    check = ogr.Open(str(gpkg))
    assert check.GetLayerByName("shortest_connections_3d").GetFeatureCount() == 2
    assert check.GetLayerByName("elevation_samples_5m").GetFeatureCount() == len(rows)
    assert check.GetLayerByName("profile_endpoints").GetFeatureCount() == 4
    for feature in check.GetLayerByName("shortest_connections_3d"):
        assert abs(feature.GetGeometryRef().Length() - feature["length_m"]) < 1e-5
        assert feature.GetGeometryRef().GetCoordinateDimension() == 3
    check = None
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(gpkg)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
