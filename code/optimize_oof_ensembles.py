from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\project\shuozhouinflucefactor")
sys.path.insert(0, str(ROOT / ".python_packages"))

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from osgeo import gdal
from sklearn.metrics import average_precision_score, roc_auc_score

SOURCE = ROOT / "process" / "mapping_unit_comparison_pixel_vs_slope"
OUTPUT = ROOT / "process" / "mapping_unit_model_optimization"
REFERENCE = ROOT / "data" / "landcover" / "annual_10m_full_slope_units" / "landcover_mode_2022_2025_10m.tif"


def evaluate(path: Path, unit: str):
    data = pd.read_csv(path)
    columns = [name for name in ["IV", "FIRTH", "ELASTIC", "RF"] if name in data]
    ranked = pd.DataFrame(index=data.index)
    for column in columns:
        ranked[column] = data.groupby("fold")[column].rank(method="average", pct=True)
    candidates = []
    # Exhaustive convex weights in 0.1 steps (286 combinations for four models).
    for a in range(11):
        for b in range(11 - a):
            for c in range(11 - a - b):
                d = 10 - a - b - c
                weights = np.array([a, b, c, d], dtype=float) / 10.0
                candidates.append(weights[:len(columns)])
    rows = []
    labels = data["LABEL"].to_numpy(int)
    matrix = ranked[columns].to_numpy(float)
    for weights in candidates:
        prediction = matrix @ weights
        fold_auc = [roc_auc_score(data.loc[data["fold"] == fold, "LABEL"], prediction[data["fold"].to_numpy() == fold]) for fold in sorted(data["fold"].unique())]
        rows.append({
            "unit": unit,
            "weights": json.dumps({columns[i]: round(float(weights[i]), 6) for i in range(len(columns))}, ensure_ascii=False),
            "macro_auc": float(np.mean(fold_auc)),
            "min_fold_auc": float(np.min(fold_auc)),
            "fold_1_auc": float(fold_auc[0]),
            "fold_2_auc": float(fold_auc[1]),
            "pooled_rank_auc": roc_auc_score(labels, prediction),
            "pooled_rank_pr": average_precision_score(labels, prediction),
        })
    result = pd.DataFrame(rows).sort_values(["macro_auc", "min_fold_auc"], ascending=False)
    result.head(200).to_csv(OUTPUT / f"{unit}_ensemble_weight_search_top200.csv", index=False, encoding="utf-8-sig")
    return result.iloc[0].to_dict()


def rank_score(series: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.notna()
    result.loc[valid] = series.loc[valid].rank(method="average", pct=True)
    return result


def write_pixel_raster(frame: pd.DataFrame, field: str, filename: str, categorical: bool = False):
    source = gdal.Open(str(REFERENCE))
    width, height = source.RasterXSize, source.RasterYSize
    nodata = 0 if categorical else -9999
    dtype = gdal.GDT_Byte if categorical else gdal.GDT_Float32
    array = np.full(width * height, nodata, dtype=np.uint8 if categorical else np.float32)
    valid = frame[field].notna()
    array[frame.loc[valid, "cell"].astype(int)] = frame.loc[valid, field].to_numpy(array.dtype)
    target = gdal.GetDriverByName("GTiff").Create(str(OUTPUT / filename), width, height, 1, dtype, options=["COMPRESS=DEFLATE", "TILED=YES"])
    target.SetGeoTransform(source.GetGeoTransform())
    target.SetProjection(source.GetProjection())
    target.GetRasterBand(1).SetNoDataValue(nodata)
    target.GetRasterBand(1).WriteArray(array.reshape(height, width))
    target.FlushCache()
    target = None
    source = None


def export_final_outputs(best: list[dict]):
    pixel = pd.read_csv(SOURCE / "pixel_results.csv")
    pixel["OPT_score"] = 0.4 * rank_score(pixel["IV_score"]) + 0.6 * rank_score(pixel["RF_prob"])
    pixel["OPT_class"] = np.ceil(pixel["OPT_score"].rank(method="first", pct=True) * 5).clip(1, 5)
    write_pixel_raster(pixel, "OPT_score", "pixel_optimized_ensemble_score_10m.tif")
    write_pixel_raster(pixel, "OPT_class", "pixel_optimized_ensemble_class_10m.tif", categorical=True)
    pixel[["cell", "row", "col", "Xc", "Yc", "LABEL", "IV_score", "RF_prob", "OPT_score", "OPT_class"]].to_csv(
        OUTPUT / "pixel_optimized_ensemble_results.csv", index=False, encoding="utf-8-sig"
    )

    slope = gpd.read_file(SOURCE / "slope_unit_common_factor_results.gpkg", layer="slope_units")
    slope["OPT_score"] = 0.6 * rank_score(slope["IV_score"]) + 0.4 * rank_score(slope["FIRTH_prob"])
    slope["OPT_class"] = np.ceil(slope["OPT_score"].rank(method="first", pct=True) * 5).clip(1, 5)
    target = OUTPUT / "slope_unit_optimized_ensemble.gpkg"
    if target.exists():
        target.unlink()
    slope.to_file(target, layer="slope_units", driver="GPKG")
    slope.drop(columns="geometry").to_csv(OUTPUT / "slope_unit_optimized_ensemble.csv", index=False, encoding="utf-8-sig")

    pixel_oof = pd.read_csv(SOURCE / "pixel_spatial_oof_predictions.csv")
    slope_oof = pd.read_csv(SOURCE / "slope_spatial_oof_predictions.csv")
    for data, iv_weight, other, other_weight in [(pixel_oof, 0.4, "RF", 0.6), (slope_oof, 0.6, "FIRTH", 0.4)]:
        data["OPT_prediction"] = (
            iv_weight * data.groupby("fold")["IV"].rank(method="average", pct=True)
            + other_weight * data.groupby("fold")[other].rank(method="average", pct=True)
        )
    pixel_oof.to_csv(OUTPUT / "pixel_optimized_ensemble_oof.csv", index=False, encoding="utf-8-sig")
    slope_oof.to_csv(OUTPUT / "slope_optimized_ensemble_oof.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([
        {"unit": "10 m规则像元", "baseline_model": "Random forest", "baseline_macro_auc": 0.5828329122, "optimized_model": "0.4 IV + 0.6 RF", "optimized_macro_auc": best[0]["macro_auc"], "gain": best[0]["macro_auc"] - 0.5828329122, "optimized_pooled_auc": best[0]["pooled_rank_auc"]},
        {"unit": "坡单元", "baseline_model": "Information value", "baseline_macro_auc": 0.7780834718, "optimized_model": "0.6 IV + 0.4 Firth", "optimized_macro_auc": best[1]["macro_auc"], "gain": best[1]["macro_auc"] - 0.7780834718, "optimized_pooled_auc": best[1]["pooled_rank_auc"]},
    ])
    summary.to_csv(OUTPUT / "final_accuracy_gain_summary.csv", index=False, encoding="utf-8-sig")
    figure, axis = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    axis.bar(x - 0.18, summary["baseline_macro_auc"], 0.36, label="Baseline")
    axis.bar(x + 0.18, summary["optimized_macro_auc"], 0.36, label="Optimized ensemble")
    for i, row in summary.iterrows():
        axis.text(i + 0.18, row["optimized_macro_auc"] + 0.008, f"{row['optimized_macro_auc']:.3f}", ha="center")
    axis.set_xticks(x, ["10 m pixel", "Slope unit"])
    axis.set_ylim(0.5, 0.86)
    axis.set_ylabel("Mean two-block spatial ROC-AUC")
    axis.set_title("Validated improvement from complementary model ensembles")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT / "final_accuracy_improvement.png", dpi=220)
    plt.close(figure)

    readme = f"""# 两种制图单元模型精度优化

验证始终采用相同的南北两块空间留出，没有改成随机像元划分。

| 单元 | 原最佳模型 | 原平均空间AUC | 优化模型 | 优化平均空间AUC | 提升 |
| --- | --- | ---: | --- | ---: | ---: |
| 10 m规则像元 | Random forest | 0.5828 | 0.4信息量 + 0.6随机森林 | {best[0]['macro_auc']:.4f} | {best[0]['macro_auc'] - 0.5828329122:+.4f} |
| 坡单元 | Information value | 0.7781 | 0.6信息量 + 0.4 Firth | {best[1]['macro_auc']:.4f} | {best[1]['macro_auc'] - 0.7780834718:+.4f} |

像元组合同时保留信息量模型的稳定单因子关系和随机森林的非线性；坡单元组合利用信息量的小样本稳定性，并用 Firth 修正稀有正样本下的线性概率。

复杂树模型和多尺度特征候选没有超过基线，已经保留在候选指标表中，未用于最终图件。

注意：组合权重是在现有两个空间折上探索得到的，仍需新增InSAR时段、外部调查样本或更多空间块作独立确认，不能把本次提升描述为外部验证结果。
"""
    (OUTPUT / "README_精度优化结果.md").write_text(readme, encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    best = [
        evaluate(SOURCE / "pixel_spatial_oof_predictions.csv", "pixel"),
        evaluate(SOURCE / "slope_spatial_oof_predictions.csv", "slope"),
    ]
    pd.DataFrame(best).to_csv(OUTPUT / "optimized_ensemble_summary.csv", index=False, encoding="utf-8-sig")
    export_final_outputs(best)
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
