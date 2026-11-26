from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\project\shuozhouinflucefactor")
UNITS_GPKG = (
    ROOT / "process" / "static_factors" / "slope_units_static_factors_full.gpkg"
)
TRACKS = {
    "T113": (
        ROOT
        / "data"
        / "insar"
        / "insar_expansion_boundary"
        / "113_boundary200m_65.shp"
    ),
    "T40": (
        ROOT
        / "data"
        / "insar"
        / "insar_expansion_boundary"
        / "40_boundary200m_65.shp"
    ),
}
OUTPUT_DIR = ROOT / "process" / "insar_expanded"
OUTPUT_GPKG = OUTPUT_DIR / "slope_unit_insar_stats.gpkg"
OUTPUT_CSV = OUTPUT_DIR / "slope_unit_insar_stats.csv"
OUTPUT_META = OUTPUT_DIR / "insar_unit_stats_metadata.json"
OUTPUT_PREVIEW = OUTPUT_DIR / "insar_slope_unit_activity_preview.png"


def percentile90(series: pd.Series) -> float:
    return float(np.nanpercentile(series.to_numpy(dtype="float64"), 90))


def aggregate_track(
    units: gpd.GeoDataFrame, track_name: str, path: Path
) -> tuple[pd.DataFrame, dict]:
    points = gpd.read_file(
        path,
        columns=["velocity", "coherence", "geometry"],
        engine="pyogrio",
    )
    points = points[
        points.geometry.notna()
        & ~points.geometry.is_empty
        & np.isfinite(points["velocity"])
        & np.isfinite(points["coherence"])
        & (points["coherence"] >= 0.65)
    ].copy()
    raw_median = float(points["velocity"].median())
    raw_mad = float(
        np.median(np.abs(points["velocity"].to_numpy() - raw_median))
    )
    points["v_center"] = points["velocity"] - raw_median
    points["v_abs_center"] = points["v_center"].abs()
    if points.crs != units.crs:
        points = points.to_crs(units.crs)

    joined = gpd.sjoin(
        points[["velocity", "coherence", "v_center", "v_abs_center", "geometry"]],
        units[["SU_ID", "geometry"]],
        how="inner",
        predicate="within",
    )
    grouped = joined.groupby("SU_ID")
    stats = grouped.agg(
        point_n=("velocity", "size"),
        v_raw_mean=("velocity", "mean"),
        v_raw_median=("velocity", "median"),
        v_center_mean=("v_center", "mean"),
        v_center_median=("v_center", "median"),
        v_center_std=("v_center", "std"),
        v_abs_median=("v_abs_center", "median"),
        coherence_median=("coherence", "median"),
    )
    stats["v_abs_p90"] = grouped["v_abs_center"].apply(percentile90)
    stats = stats.reset_index()

    # The expanded InSAR data cover complete model slope units.  Estimate
    # nominal point density only inside those units and normalize coverage by
    # complete-unit area, not by the old 1 km clipped area.
    model_units = units[
        (units["retain_rule"] == 1) & (units["use_model"] == 1)
    ]
    model_ids = set(model_units["SU_ID"])
    joined_model_count = int(joined["SU_ID"].isin(model_ids).sum())
    total_full_area = float(model_units["full_area_km2"].sum())
    nominal_density = joined_model_count / total_full_area
    full_area_by_id = units.set_index("SU_ID")["full_area_km2"]
    stats["coverage_norm"] = (
        stats["point_n"]
        / (stats["SU_ID"].map(full_area_by_id) * nominal_density)
    ).clip(upper=1.0)

    rename = {
        column: f"{track_name}_{column}"
        for column in stats.columns
        if column != "SU_ID"
    }
    stats = stats.rename(columns=rename)
    metadata = {
        "path": str(path),
        "valid_point_count": len(points),
        "joined_point_count": len(joined),
        "raw_velocity_median": raw_median,
        "raw_velocity_mad": raw_mad,
        "velocity_centering": "subtract the track-wide median",
        "joined_model_point_count": joined_model_count,
        "nominal_points_per_km2_full_model_units": nominal_density,
        "coverage_area_basis": "complete slope-unit area (full_area_km2)",
        "minimum_coherence": 0.65,
    }
    return stats, metadata


def make_preview(units: gpd.GeoDataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), dpi=160)
    for axis, field, title in [
        (axes[0], "T113_v_center_median", "Track 113 centered median LOS velocity"),
        (axes[1], "T40_v_center_median", "Track 40 centered median LOS velocity"),
    ]:
        units.plot(
            column=field,
            ax=axis,
            cmap="RdBu_r",
            legend=True,
            edgecolor="0.55",
            linewidth=0.25,
            missing_kwds={"color": "lightgray"},
        )
        units.loc[units["LS_label"] == 1].boundary.plot(
            ax=axis, color="yellow", linewidth=1.5
        )
        axis.set_title(title)
        axis.set_axis_off()
    fig.suptitle(
        "Slope-unit InSAR statistics (yellow outlines: mapped landslide units)"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PREVIEW, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    units = gpd.read_file(UNITS_GPKG, layer="slope_units")
    metadata = {}
    for track_name, path in TRACKS.items():
        stats, track_metadata = aggregate_track(units, track_name, path)
        units = units.merge(stats, on="SU_ID", how="left")
        metadata[track_name] = track_metadata

    median_fields = ["T113_v_center_median", "T40_v_center_median"]
    p90_fields = ["T113_v_abs_p90", "T40_v_abs_p90"]
    coverage_fields = ["T113_coverage_norm", "T40_coverage_norm"]
    units["INSAR_absmed_max"] = units[median_fields].abs().max(axis=1)
    units["INSAR_absp90_max"] = units[p90_fields].max(axis=1)
    units["INSAR_cov_min"] = units[coverage_fields].min(axis=1)
    units["INSAR_cov_max"] = units[coverage_fields].max(axis=1)
    units["INSAR_tracks"] = units[coverage_fields].notna().sum(axis=1)
    units["INSAR_stable"] = (
        (units["INSAR_tracks"] == 2)
        & (units["INSAR_cov_min"] >= 0.60)
        & (units["INSAR_absmed_max"] <= 1.0)
        & (units["INSAR_absp90_max"] <= 3.0)
        & (units["LS_label"] == 0)
        & (units["UNSTABLE"] == 0)
    ).astype(int)
    insufficient = (units["INSAR_tracks"] == 0) | (
        units["INSAR_cov_max"].fillna(0) < 0.30
    )
    strong = (units["INSAR_absmed_max"] >= 3.0) | (
        units["INSAR_absp90_max"] >= 6.0
    )
    stable_signal = (
        (units["INSAR_tracks"] == 2)
        & (units["INSAR_cov_min"] >= 0.60)
        & (units["INSAR_absmed_max"] <= 1.0)
        & (units["INSAR_absp90_max"] <= 3.0)
    )
    units["INSAR_act"] = 2
    units.loc[stable_signal, "INSAR_act"] = 1
    units.loc[strong, "INSAR_act"] = 3
    units.loc[insufficient, "INSAR_act"] = 0
    units["INSAR_act_name"] = units["INSAR_act"].map(
        {0: "覆盖不足", 1: "稳定", 2: "弱/不确定形变", 3: "显著形变"}
    )

    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()
    units.to_file(OUTPUT_GPKG, layer="slope_units", driver="GPKG")
    units.drop(columns="geometry").to_csv(
        OUTPUT_CSV, index=False, encoding="utf-8-sig"
    )
    make_preview(units)

    summary = {
        "unit_count": len(units),
        "units_with_track_113": int(units["T113_point_n"].notna().sum()),
        "units_with_track_40": int(units["T40_point_n"].notna().sum()),
        "units_with_both_tracks": int((units["INSAR_tracks"] == 2).sum()),
        "tracks": metadata,
        "combined_fields": {
            "INSAR_absmed_max": "maximum absolute centered unit-median velocity across tracks",
            "INSAR_absp90_max": "maximum 90th percentile absolute centered velocity across tracks",
            "INSAR_cov_min": "minimum normalized coverage across tracks",
            "INSAR_cov_max": "maximum normalized coverage across tracks",
            "INSAR_tracks": "number of tracks with valid observations",
            "INSAR_stable": (
                "stable negative candidate: two tracks, minimum normalized "
                "coverage >=0.60, maximum absolute centered median <=1.0, "
                "maximum absolute p90 <=3.0 mm/yr, and not mapped landslide/unstable"
            ),
            "INSAR_act": "0 insufficient, 1 stable, 2 weak/uncertain, 3 strong",
        },
        "stable_negative_candidate_count": int(units["INSAR_stable"].sum()),
        "activity_class_counts": {
            str(key): int(value)
            for key, value in units["INSAR_act"].value_counts().sort_index().items()
        },
        "important_note": (
            "These InSAR fields are reserved for stable-negative selection, "
            "activity classification and final dynamic correction; they are not "
            "used as ordinary static susceptibility predictors."
        ),
    }
    OUTPUT_META.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
