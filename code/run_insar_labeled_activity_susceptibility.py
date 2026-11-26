from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(r"D:\project\shuozhouinflucefactor")
sys.path.insert(0, str(ROOT / ".python_packages"))
sys.path.insert(0, str(ROOT / "scripts"))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from osgeo import gdal
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from run_susceptibility_models import firth_fit


INSAR_CSV = ROOT / "process" / "insar_expanded" / "slope_unit_insar_stats.csv"
STATIC_CSV = ROOT / "process" / "expanded_factors" / "slope_units_expanded_factors.csv"
BASE_GPKG = ROOT / "process" / "expanded_factors" / "slope_units_expanded_factors.gpkg"
CORE_GPKG = ROOT / "process" / "factor_domain" / "analysis_domains.gpkg"
REFERENCE_RASTER = ROOT / "data" / "ndvi" / "sentinel2_monthly" / "ndvi_202211_10m.tif"
SCENARIO = os.environ.get("INSAR_LABEL_SCENARIO", "loose").lower()
SCENARIO_CONFIG = {
    "loose": {"median": 1.0, "p90": 3.0, "cn": "宽松"},
    "medium": {"median": 2.0, "p90": 5.0, "cn": "中等"},
    "strict": {"median": 3.0, "p90": 6.0, "cn": "严格"},
}
if SCENARIO not in SCENARIO_CONFIG:
    raise RuntimeError(f"Unknown INSAR_LABEL_SCENARIO: {SCENARIO}")
OUTPUT_TAG = os.environ.get("INSAR_OUTPUT_TAG", SCENARIO)
OUTPUT_DIR = ROOT / "process" / f"insar_labeled_activity_susceptibility_{OUTPUT_TAG}"
RESULT_CSV = OUTPUT_DIR / "activity_susceptibility_results.csv"
OUTPUT_GPKG = OUTPUT_DIR / "activity_susceptibility_models.gpkg"
METRICS_CSV = OUTPUT_DIR / "spatial_cv_metrics.csv"
FOLD_CSV = OUTPUT_DIR / "spatial_fold_predictions.csv"
SCREEN_CSV = OUTPUT_DIR / "factor_screening.csv"
IV_CSV = OUTPUT_DIR / "information_value_table.csv"
FIRTH_CSV = OUTPUT_DIR / "firth_coefficients.csv"
RF_CSV = OUTPUT_DIR / "rf_feature_importance.csv"
ELASTIC_CSV = OUTPUT_DIR / "elastic_net_coefficients.csv"
META_JSON = OUTPUT_DIR / "model_metadata.json"

MEDIAN_THRESHOLD = SCENARIO_CONFIG[SCENARIO]["median"]
P90_THRESHOLD = SCENARIO_CONFIG[SCENARIO]["p90"]
SCENARIO_CN = SCENARIO_CONFIG[SCENARIO]["cn"]
MIN_COVERAGE = 0.60
SPATIAL_FOLDS = int(os.environ.get("INSAR_SPATIAL_FOLDS", "5"))
RANDOM_SEED = 42
CORRELATION_LIMIT = 0.75

CANDIDATE_FEATURES = [
    "slope_mean", "slope_std", "northness", "eastness", "pcurv_mean",
    "plcurv_mean", "twi_mean", "tpi_mean", "drain_dist_mean", "elev_range",
    "compact", "ROADdist_m", "FAULTdist_m", "BUILTdist_mean_m",
    "LCfrac_forest", "LCfrac_cropland", "LCfrac_built", "LCfrac_grass",
    "LC_entropy", "LC_edge_fraction", "LC_change_fraction",
    "SG_clay_top_pct", "SG_sand_top_pct", "SG_silt_top_pct",
    "SG_bdod_top_gcm3", "SG_soc_top_gkg", "SG_cfvo_top_pct",
    "SG_cec_top_cmolkg", "SG_ph_top", "SG_clay_vertical_diff",
    "SG_sand_vertical_diff", "SG_bdod_vertical_diff", "SG_cfvo_vertical_diff",
]

FIRTH_FEATURES = [
    "slope_mean", "twi_mean", "drain_dist_mean", "ROADdist_m",
    "LC_entropy", "SG_clay_top_pct",
]

IV_FEATURES = [
    "slope_mean", "elev_range", "twi_mean", "drain_dist_mean",
    "ROADdist_m", "FAULTdist_m", "LC_entropy", "SG_clay_top_pct",
]

CLASS_NAMES = {
    0: "未评价", 1: "极低易发", 2: "低易发", 3: "中易发",
    4: "高易发", 5: "极高易发",
}


def load_data() -> pd.DataFrame:
    static = pd.read_csv(STATIC_CSV)
    insar = pd.read_csv(INSAR_CSV)[[
        "SU_ID", "INSAR_absmed_max", "INSAR_absp90_max", "INSAR_cov_min",
        "INSAR_cov_max", "INSAR_tracks", "T113_point_n", "T40_point_n",
        "T113_coherence_median", "T40_coherence_median",
    ]]
    data = static.merge(insar, on="SU_ID", how="left")
    model = (data["retain_rule"] == 1) & (data["use_model"] == 1)
    quality = (
        model
        & (data["INSAR_tracks"] == 2)
        & (data["INSAR_cov_min"] >= MIN_COVERAGE)
    )
    positive = quality & (
        (data["INSAR_absmed_max"] > MEDIAN_THRESHOLD)
        | (data["INSAR_absp90_max"] > P90_THRESHOLD)
    )
    # Keep the negative definition fixed across all sensitivity scenarios.
    # Units between the stable and positive thresholds remain unlabeled.
    negative = quality & (
        (data["INSAR_absmed_max"] <= 1.0)
        & (data["INSAR_absp90_max"] <= 3.0)
    )
    data["LABEL"] = -1
    data.loc[negative, "LABEL"] = 0
    data.loc[positive, "LABEL"] = 1
    data["LABEL_name"] = data["LABEL"].map(
        {-1: "未用于训练", 0: "InSAR稳定负样本", 1: "InSAR活动正样本"}
    )
    data["LABEL_quality"] = quality.astype(int)
    return data


def screen_features(data: pd.DataFrame, model_mask: np.ndarray) -> tuple[list[str], pd.DataFrame]:
    scoped = data.loc[model_mask, CANDIDATE_FEATURES]
    retained: list[str] = []
    rows = []
    for feature in CANDIDATE_FEATURES:
        values = scoped[feature]
        reason = ""
        conflict = ""
        if values.isna().any():
            reason = "missing_values"
        elif values.nunique() < 5 or values.std() < 1e-8:
            reason = "low_variance"
        else:
            correlations = {
                existing: abs(scoped[[feature, existing]].corr(method="spearman").iloc[0, 1])
                for existing in retained
            }
            if correlations:
                conflict, maximum = max(correlations.items(), key=lambda item: item[1])
                if maximum >= CORRELATION_LIMIT:
                    reason = f"correlated_rho={maximum:.3f}"
            if not reason:
                retained.append(feature)
        rows.append({
            "factor": feature,
            "retained": int(not reason),
            "reason_excluded": reason,
            "correlated_with": conflict if reason.startswith("correlated") else "",
            "unique_count": int(values.nunique()),
            "standard_deviation": float(values.std()),
        })
    return retained, pd.DataFrame(rows)


def standardize(train: pd.DataFrame, prediction: pd.DataFrame, features: list[str]):
    train_values = train[features].to_numpy(dtype="float64")
    prediction_values = prediction[features].to_numpy(dtype="float64")
    means = train_values.mean(axis=0)
    scales = train_values.std(axis=0, ddof=0)
    scales[scales == 0] = 1.0
    return (train_values - means) / scales, (prediction_values - means) / scales, means, scales


def iv_fit_predict(data: pd.DataFrame, train_indices: np.ndarray, prediction_indices: np.ndarray):
    train = data.loc[train_indices]
    prediction = data.loc[prediction_indices]
    positive_total = int(train["LABEL"].sum())
    negative_total = int((train["LABEL"] == 0).sum())
    score = np.zeros(len(prediction), dtype="float64")
    rows = []
    for feature in IV_FEATURES:
        _, raw_edges = pd.qcut(train[feature], q=4, duplicates="drop", retbins=True)
        edges = np.unique(raw_edges)
        edges[0], edges[-1] = -np.inf, np.inf
        train_groups = pd.cut(train[feature], edges, include_lowest=True)
        pred_groups = pd.cut(prediction[feature], edges, include_lowest=True)
        categories = train_groups.cat.categories
        mapping = {}
        for category in categories:
            members = train_groups == category
            pos = int((train.loc[members, "LABEL"] == 1).sum())
            neg = int((train.loc[members, "LABEL"] == 0).sum())
            pos_rate = (pos + 0.5) / (positive_total + 0.5 * len(categories))
            neg_rate = (neg + 0.5) / (negative_total + 0.5 * len(categories))
            value = float(np.log(pos_rate / neg_rate))
            mapping[category] = value
            rows.append({
                "factor": feature, "group": str(category),
                "positive_count": pos, "negative_count": neg,
                "information_value": value,
            })
        score += pred_groups.map(mapping).astype(float).fillna(0).to_numpy()
    return score, pd.DataFrame(rows)


def fit_firth(data: pd.DataFrame, train_indices: np.ndarray, prediction_indices: np.ndarray):
    train = data.loc[train_indices]
    prediction = data.loc[prediction_indices]
    train_x, pred_x, means, scales = standardize(train, prediction, FIRTH_FEATURES)
    train_matrix = np.column_stack([np.ones(len(train)), train_x])
    pred_matrix = np.column_stack([np.ones(len(prediction)), pred_x])
    coefficients, errors, diagnostics = firth_fit(
        train_matrix, train["LABEL"].to_numpy(dtype=int)
    )
    probabilities = expit(np.clip(pred_matrix @ coefficients, -35, 35))
    table = pd.DataFrame({
        "term": ["Intercept", *FIRTH_FEATURES],
        "coefficient": coefficients,
        "std_error": errors,
        "odds_ratio": np.exp(coefficients),
        "ci95_low": np.exp(coefficients - 1.96 * errors),
        "ci95_high": np.exp(coefficients + 1.96 * errors),
        "standardization_mean": [np.nan, *means],
        "standardization_std": [np.nan, *scales],
    })
    return probabilities, table, diagnostics


def fit_elastic(data: pd.DataFrame, train_indices: np.ndarray, prediction_indices: np.ndarray, features: list[str]):
    train = data.loc[train_indices]
    prediction = data.loc[prediction_indices]
    train_x, pred_x, _, _ = standardize(train, prediction, features)
    model = LogisticRegression(
        penalty="elasticnet", solver="saga", C=0.3, l1_ratio=0.5,
        class_weight="balanced", max_iter=20000, tol=1e-6,
        random_state=RANDOM_SEED,
    )
    model.fit(train_x, train["LABEL"].to_numpy(dtype=int))
    table = pd.DataFrame({"feature": features, "standardized_coefficient": model.coef_[0]})
    table["absolute_coefficient"] = table["standardized_coefficient"].abs()
    return model.predict_proba(pred_x)[:, 1], table.sort_values("absolute_coefficient", ascending=False)


def fit_rf(data: pd.DataFrame, train_indices: np.ndarray, prediction_indices: np.ndarray, features: list[str], seed: int):
    model = RandomForestClassifier(
        n_estimators=1000, max_depth=6, min_samples_leaf=3,
        max_features="sqrt", class_weight="balanced_subsample",
        random_state=seed, n_jobs=1,
    )
    model.fit(data.loc[train_indices, features], data.loc[train_indices, "LABEL"].to_numpy(dtype=int))
    importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    return model.predict_proba(data.loc[prediction_indices, features])[:, 1], importance


def assign_spatial_folds(data: pd.DataFrame, labeled_indices: np.ndarray) -> pd.Series:
    # The corridor is predominantly north-south. Rank-based Y blocks produce
    # five contiguous spatial holdouts of nearly equal sample size.
    ranks = data.loc[labeled_indices, "Yc"].rank(method="first")
    fold = pd.Series(-1, index=data.index, dtype=int)
    fold.loc[labeled_indices] = pd.qcut(ranks, q=SPATIAL_FOLDS, labels=False).astype(int).to_numpy()
    return fold


def evaluate(name: str, target: np.ndarray, scores: np.ndarray, fold: str) -> dict:
    return {
        "model": name,
        "fold": fold,
        "sample_count": int(len(target)),
        "positive_count": int(target.sum()),
        "negative_count": int((1 - target).sum()),
        "roc_auc": float(roc_auc_score(target, scores)),
        "pr_auc": float(average_precision_score(target, scores)),
    }


def spatial_cross_validation(data: pd.DataFrame, labeled_indices: np.ndarray, features: list[str]):
    folds = assign_spatial_folds(data, labeled_indices)
    oof = pd.DataFrame(index=labeled_indices, columns=["IV", "FIRTH", "ELASTIC", "RF"], dtype=float)
    rows = []
    for fold_id in range(SPATIAL_FOLDS):
        test_indices = labeled_indices[folds.loc[labeled_indices].to_numpy() == fold_id]
        train_indices = labeled_indices[folds.loc[labeled_indices].to_numpy() != fold_id]
        target = data.loc[test_indices, "LABEL"].to_numpy(dtype=int)
        if np.unique(target).size < 2:
            raise RuntimeError(f"Spatial fold {fold_id + 1} lacks one label class")
        iv_scores, _ = iv_fit_predict(data, train_indices, test_indices)
        firth_scores, _, _ = fit_firth(data, train_indices, test_indices)
        elastic_scores, _ = fit_elastic(data, train_indices, test_indices, features)
        rf_scores, _ = fit_rf(data, train_indices, test_indices, features, RANDOM_SEED + fold_id + 1)
        for key, name, scores in [
            ("IV", "Information value", iv_scores),
            ("FIRTH", "Firth logistic regression", firth_scores),
            ("ELASTIC", "Elastic-net logistic regression", elastic_scores),
            ("RF", "Random forest", rf_scores),
        ]:
            oof.loc[test_indices, key] = scores
            rows.append(evaluate(name, target, scores, f"spatial_{fold_id + 1}"))
    for key in oof.columns:
        oof[f"{key}_rank"] = oof[key].rank(method="average", pct=True)
    oof["ENSEMBLE"] = oof[[f"{key}_rank" for key in ["IV", "FIRTH", "ELASTIC", "RF"]]].mean(axis=1)
    target = data.loc[labeled_indices, "LABEL"].to_numpy(dtype=int)
    for key, name in [
        ("IV", "Information value"), ("FIRTH", "Firth logistic regression"),
        ("ELASTIC", "Elastic-net logistic regression"), ("RF", "Random forest"),
        ("ENSEMBLE", "Rank ensemble"),
    ]:
        rows.append(evaluate(name, target, oof[key].to_numpy(dtype=float), "all_oof"))
    oof["SU_ID"] = data.loc[labeled_indices, "SU_ID"].to_numpy()
    oof["LABEL"] = target
    oof["spatial_fold"] = folds.loc[labeled_indices].to_numpy() + 1
    return folds, oof, pd.DataFrame(rows)


def rank_scope(values: pd.Series, mask: np.ndarray) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result.loc[mask] = values.loc[mask].rank(method="average", pct=True)
    return result


def rasterize_field(field: str, filename: str, gdal_type: int, nodata: float) -> None:
    reference = gdal.Open(str(REFERENCE_RASTER))
    target = gdal.GetDriverByName("GTiff").Create(
        str(OUTPUT_DIR / filename), reference.RasterXSize, reference.RasterYSize,
        1, gdal_type, options=["COMPRESS=DEFLATE", "TILED=YES"],
    )
    target.SetGeoTransform(reference.GetGeoTransform())
    target.SetProjection(reference.GetProjection())
    band = target.GetRasterBand(1)
    band.Fill(nodata)
    band.SetNoDataValue(nodata)
    vector = gdal.OpenEx(str(OUTPUT_GPKG), gdal.OF_VECTOR)
    layer = vector.GetLayerByName("core_1km_results")
    status = gdal.RasterizeLayer(target, [1], layer, options=[f"ATTRIBUTE={field}"])
    if status != 0:
        raise RuntimeError(f"Rasterizing {field} failed")
    target.FlushCache()
    vector = None
    target = None


def export_spatial(data: pd.DataFrame) -> gpd.GeoDataFrame:
    base = gpd.read_file(BASE_GPKG, layer="slope_units")
    fields = [
        "SU_ID", "LABEL", "LABEL_name", "LABEL_quality", "spatial_fold",
        "ACT_score", "ACT_class", "ACT_name", "IV_score", "FIRTH_prob",
        "EN_prob", "RF_prob", "INSAR_absmed_max", "INSAR_absp90_max",
        "INSAR_cov_min", "INSAR_tracks", "LS_label", "UNSTABLE",
    ]
    merged = base.drop(columns=[c for c in ["LS_label", "UNSTABLE"] if c in base.columns]).merge(data[fields], on="SU_ID", how="left")
    core_area = gpd.read_file(CORE_GPKG, layer="core_1km").to_crs(merged.crs)
    core_geometry = core_area.geometry.union_all()
    core = merged[merged.geometry.intersects(core_geometry)].copy()
    core.geometry = core.geometry.intersection(core_geometry)
    core = core[core.geometry.notna() & ~core.geometry.is_empty]
    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()
    merged.to_file(OUTPUT_GPKG, layer="full_slope_unit_results", driver="GPKG")
    core.to_file(OUTPUT_GPKG, layer="core_1km_results", driver="GPKG", mode="a")
    rasterize_field("ACT_score", "activity_susceptibility_score_core_10m.tif", gdal.GDT_Float32, -9999)
    rasterize_field("ACT_class", "activity_susceptibility_class_core_10m.tif", gdal.GDT_Byte, 0)
    rasterize_field("LABEL", "insar_training_label_core_10m.tif", gdal.GDT_Int16, -1)
    return merged


def make_plots(data: gpd.GeoDataFrame, metrics: pd.DataFrame, rf_importance: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=160)
    label_colors = ListedColormap(["#BDBDBD", "#2C7BB6", "#D7191C"])
    data.plot(column="LABEL", cmap=label_colors, norm=BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3), ax=axes[0], edgecolor="0.6", linewidth=0.2, legend=True)
    axes[0].set_title("InSAR-derived training label")
    data.plot(column="ACT_score", cmap="viridis", ax=axes[1], edgecolor="0.6", linewidth=0.2, legend=True)
    axes[1].set_title("Static activity susceptibility score")
    class_colors = ListedColormap(["#2C7BB6", "#ABD9E9", "#FFFFBF", "#FDAE61", "#D7191C"])
    data.plot(column="ACT_class", cmap=class_colors, norm=BoundaryNorm(np.arange(0.5, 6.5, 1), 5), ax=axes[2], edgecolor="0.6", linewidth=0.2, legend=True)
    axes[2].set_title("Static activity susceptibility class")
    for axis in axes:
        axis.set_axis_off()
    fig.suptitle(f"InSAR-labeled slope activity susceptibility ({SCENARIO} threshold)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "activity_susceptibility_preview.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    oof = pd.read_csv(FOLD_CSV)
    fig, axis = plt.subplots(figsize=(7, 6), dpi=160)
    model_names = {
        "IV": "Information value",
        "FIRTH": "Firth logistic regression",
        "ELASTIC": "Elastic-net logistic regression",
        "RF": "Random forest",
        "ENSEMBLE": "Rank ensemble",
    }
    short_names = {
        "IV": "IV", "FIRTH": "Firth", "ELASTIC": "Elastic-net",
        "RF": "RF", "ENSEMBLE": "Ensemble",
    }
    for field in ["IV", "FIRTH", "ELASTIC", "RF", "ENSEMBLE"]:
        fpr, tpr, _ = roc_curve(oof["LABEL"], oof[field])
        auc_value = metrics[
            (metrics["fold"] == "all_oof")
            & (metrics["model"] == model_names[field])
        ]["roc_auc"]
        name = short_names[field]
        label = f"{name} (AUC={auc_value.iloc[0]:.3f})"
        axis.plot(fpr, tpr, label=label)
    axis.plot([0, 1], [0, 1], "--", color="0.5")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(f"{SPATIAL_FOLDS}-fold spatial out-of-fold ROC")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "spatial_cv_roc.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    top = rf_importance.head(15).sort_values("importance")
    fig, axis = plt.subplots(figsize=(8, 6), dpi=160)
    axis.barh(top["feature"], top["importance"], color="#54A24B")
    axis.set_title("Random-forest static-factor importance")
    axis.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rf_feature_importance.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    model_mask = ((data["retain_rule"] == 1) & (data["use_model"] == 1)).to_numpy()
    labeled_indices = data.index[data["LABEL"].isin([0, 1])].to_numpy()
    prediction_indices = data.index.to_numpy()
    selected, screening = screen_features(data, model_mask)
    screening.to_csv(SCREEN_CSV, index=False, encoding="utf-8-sig")

    folds, oof, metrics = spatial_cross_validation(data, labeled_indices, selected)
    data["spatial_fold"] = folds
    oof.to_csv(FOLD_CSV, index=False, encoding="utf-8-sig")
    metrics.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")

    data["IV_score"], iv_table = iv_fit_predict(data, labeled_indices, prediction_indices)
    iv_table.to_csv(IV_CSV, index=False, encoding="utf-8-sig")
    data["FIRTH_prob"], firth_table, firth_diagnostics = fit_firth(data, labeled_indices, prediction_indices)
    firth_table.to_csv(FIRTH_CSV, index=False, encoding="utf-8-sig")
    data["EN_prob"], elastic_table = fit_elastic(data, labeled_indices, prediction_indices, selected)
    elastic_table.to_csv(ELASTIC_CSV, index=False, encoding="utf-8-sig")
    data["RF_prob"], rf_importance = fit_rf(data, labeled_indices, prediction_indices, selected, RANDOM_SEED)
    rf_importance.to_csv(RF_CSV, index=False, encoding="utf-8-sig")

    for field in ["IV_score", "FIRTH_prob", "EN_prob", "RF_prob"]:
        data[f"{field}_rank"] = rank_scope(data[field], model_mask)
    data["ACT_score"] = data[[f"{field}_rank" for field in ["IV_score", "FIRTH_prob", "EN_prob", "RF_prob"]]].mean(axis=1)
    data["ACT_class"] = 0
    ranks = data.loc[model_mask, "ACT_score"].rank(method="first", pct=True)
    data.loc[model_mask, "ACT_class"] = np.ceil(ranks * 5).clip(1, 5).astype(int)
    data["ACT_name"] = data["ACT_class"].map(CLASS_NAMES)

    data.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    spatial = export_spatial(data)
    make_plots(spatial, metrics, rf_importance)

    known = data[(data["LS_label"] == 1) | (data["UNSTABLE"] == 1)][[
        "SU_ID", "LS_label", "UNSTABLE", "LABEL", "ACT_score", "ACT_class",
        "INSAR_absmed_max", "INSAR_absp90_max", "INSAR_cov_min",
    ]]
    known.to_csv(OUTPUT_DIR / "known_areas_case_check.csv", index=False, encoding="utf-8-sig")
    all_oof = metrics[metrics["fold"] == "all_oof"].copy()
    metadata = {
        "objective": "Predict InSAR-observed slope activity from static environmental factors",
        "scenario": SCENARIO,
        "label_definition": {
            "quality": "model unit, two tracks, minimum normalized coverage >= 0.60",
            "positive": f"absolute centered slope-unit median > {MEDIAN_THRESHOLD:g} mm/yr OR absolute P90 > {P90_THRESHOLD:g} mm/yr",
            "negative": "absolute centered slope-unit median <= 1 mm/yr AND absolute P90 <= 3 mm/yr",
            "landslide_inventory_used_for_label": False,
            "insar_used_as_predictor": False,
        },
        "sample_counts": {
            "model_units": int(model_mask.sum()),
            "positive": int((data["LABEL"] == 1).sum()),
            "negative": int((data["LABEL"] == 0).sum()),
            "excluded_or_unlabeled": int((model_mask & (data["LABEL"] == -1).to_numpy()).sum()),
            "training_total": int(len(labeled_indices)),
        },
        "spatial_validation": {
            "method": f"{SPATIAL_FOLDS} contiguous north-south Y-coordinate blocks",
            "all_oof_metrics": all_oof.to_dict(orient="records"),
        },
        "selected_static_factors": selected,
        "firth_features": FIRTH_FEATURES,
        "information_value_features": IV_FEATURES,
        "firth_diagnostics": firth_diagnostics,
        "limitations": [
            "The outcome is InSAR-observed activity, not confirmed landslide occurrence.",
            f"The {SCENARIO} threshold trades label quantity against label-noise risk.",
            "Adjacent positive slope units may belong to the same deformation cluster.",
            "The same InSAR data define labels and therefore cannot serve as independent validation.",
            "Spatial cross-validation measures transferability of static factors to held-out corridor sections, not geological ground truth accuracy.",
        ],
    }
    META_JSON.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    metric_lines = ["| 模型 | ROC-AUC | PR-AUC |", "| --- | ---: | ---: |"]
    for _, row in all_oof.iterrows():
        metric_lines.append(
            f"| {row['model']} | {row['roc_auc']:.4f} | {row['pr_auc']:.4f} |"
        )
    metric_table = "\n".join(metric_lines)
    readme = f"""# InSAR 活动坡单元易发性（{SCENARIO_CN}阈值）\n\n- 模型坡单元：{int(model_mask.sum())} 个。\n- InSAR 活动正样本：{int((data['LABEL'] == 1).sum())} 个。\n- InSAR 稳定负样本：{int((data['LABEL'] == 0).sum())} 个。\n- 未参与训练：{int((model_mask & (data['LABEL'] == -1).to_numpy()).sum())} 个。\n- 标签不使用滑坡清单；InSAR不作为预测因子。\n- 验证采用沿研究廊道的{SPATIAL_FOLDS}折连续空间分块。\n\n## 标签\n\n正样本要求两轨、最小覆盖率不低于0.60，并满足坡单元绝对中位速率大于{MEDIAN_THRESHOLD:g} mm/yr或绝对P90大于{P90_THRESHOLD:g} mm/yr。\n\n## 总体空间验证\n\n{metric_table}\n\n结果表示静态环境对2022—2025年InSAR活动形变的相对易发性，不是经现场确认的滑坡发生概率。\n"""
    (OUTPUT_DIR / "README_结果说明.md").write_text(readme, encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
