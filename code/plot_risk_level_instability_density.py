from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from osgeo import ogr
from scipy.optimize import minimize
from scipy.stats import chi2, norm, spearmanr


ROOT = Path(r"D:\project\shuozhouinflucefactor")
VALIDATION_DIR = ROOT / "process" / "wall_segment_pixel_risk_instability_validation"
SEGMENT_CSV = VALIDATION_DIR / "validation_segments.csv"
OBJECT_CSV = VALIDATION_DIR / "instability_objects_clustered.csv"
VALIDATION_GPKG = VALIDATION_DIR / "wall_risk_instability_validation.gpkg"

ASSOCIATION_DISTANCE_M = 50.0
N_GROUPS = int(os.getenv("RISK_DENSITY_GROUPS", "5"))
if N_GROUPS < 3 or N_GROUPS > 15:
    raise ValueError("RISK_DENSITY_GROUPS must be between 3 and 15.")
GROUP_LABELS = [f"第{index}组" for index in range(1, N_GROUPS + 1)]
GROUP_SUFFIX = "" if N_GROUPS == 5 else f"_{N_GROUPS}组"
OUTPUT_STEM = VALIDATION_DIR / f"广武长城_基础风险得分排序与结构失稳对象密度{GROUP_SUFFIX}"
TABLE_PATH = VALIDATION_DIR / f"risk_score_{N_GROUPS}group_instability_density_50m.csv"
ASSIGNMENT_PATH = VALIDATION_DIR / f"instability_object_unique_segment_assignment_50m{GROUP_SUFFIX}.csv"
REGRESSION_PATH = VALIDATION_DIR / "continuous_risk_score_count_regression_50m.csv"


def configure_chinese_font() -> str:
    preferred = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
    ]
    available = {item.name for item in font_manager.fontManager.ttflist}
    selected = next((name for name in preferred if name in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": selected,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return selected


def read_segment_geometries(path: Path) -> dict[int, ogr.Geometry]:
    source = ogr.Open(str(path))
    if source is None:
        raise FileNotFoundError(path)
    layer = source.GetLayerByName("validation_segments")
    if layer is None:
        layer = source.GetLayerByName("wall_segment_risk") or source.GetLayer(0)
    geometries: dict[int, ogr.Geometry] = {}
    for feature in layer:
        segment_id = int(feature.GetField("segment_id"))
        geometries[segment_id] = feature.GetGeometryRef().Clone()
    source = None
    return geometries


def assign_objects_uniquely(
    segments: pd.DataFrame,
    objects: pd.DataFrame,
    geometries: dict[int, ogr.Geometry],
) -> pd.DataFrame:
    covered = segments.loc[segments["coverage"].astype(int).eq(1)].copy()
    covered_ids = covered["segment_id"].astype(int).tolist()
    validation_objects = objects.loc[objects["in_coverage"].astype(str).str.lower().eq("true")].copy()

    assignments: list[dict] = []
    for row in validation_objects.itertuples(index=False):
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint_2D(float(row.x), float(row.y))
        distances = np.asarray([point.Distance(geometries[item]) for item in covered_ids], dtype=float)
        nearest_position = int(np.argmin(distances))
        nearest_id = covered_ids[nearest_position]
        nearest_distance = float(distances[nearest_position])
        if nearest_distance <= ASSOCIATION_DISTANCE_M:
            assignments.append(
                {
                    "object_id": int(row.object_id),
                    "segment_id": nearest_id,
                    "distance_to_segment_m": nearest_distance,
                    "risk_score": float(
                        covered.loc[covered["segment_id"].eq(nearest_id), "risk_score"].iloc[0]
                    ),
                }
            )
    return pd.DataFrame(assignments)


def poisson_exact_interval(count: int, exposure_km: float, alpha: float = 0.05) -> tuple[float, float]:
    if exposure_km <= 0:
        return float("nan"), float("nan")
    lower_count = 0.0 if count == 0 else 0.5 * chi2.ppf(alpha / 2.0, 2 * count)
    upper_count = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2 * (count + 1))
    return lower_count / exposure_km, upper_count / exposure_km


def poisson_irr(counts: np.ndarray, risk_score: np.ndarray, length_km: np.ndarray) -> dict:
    z_risk = (risk_score - np.mean(risk_score)) / np.std(risk_score, ddof=0)
    design = np.column_stack([np.ones_like(z_risk), z_risk])
    offset = np.log(length_km)

    def objective(beta: np.ndarray) -> float:
        eta = design @ beta + offset
        mu = np.exp(np.clip(eta, -30, 30))
        return float(np.sum(mu - counts * eta))

    initial_rate = max(float(np.sum(counts) / np.sum(length_km)), 1e-6)
    result = minimize(objective, np.asarray([math.log(initial_rate), 0.0]), method="BFGS")
    beta = np.asarray(result.x, dtype=float)
    eta = design @ beta + offset
    mu = np.exp(np.clip(eta, -30, 30))

    bread_inverse = np.linalg.pinv(design.T @ (mu[:, None] * design))
    score = design * (counts - mu)[:, None]
    meat = score.T @ score
    robust_covariance = bread_inverse @ meat @ bread_inverse
    standard_error = float(np.sqrt(max(robust_covariance[1, 1], 0.0)))
    z_value = float(beta[1] / standard_error) if standard_error > 0 else float("nan")
    p_value = float(2.0 * norm.sf(abs(z_value))) if np.isfinite(z_value) else float("nan")
    return {
        "beta": float(beta[1]),
        "robust_se": standard_error,
        "irr": float(np.exp(beta[1])),
        "irr_lower": float(np.exp(beta[1] - 1.96 * standard_error)),
        "irr_upper": float(np.exp(beta[1] + 1.96 * standard_error)),
        "p_value": p_value,
        "pearson_dispersion": float(np.sum((counts - mu) ** 2 / np.maximum(mu, 1e-12)) / (len(counts) - 2)),
        "converged": bool(result.success),
    }


def circular_shift_test(segments: pd.DataFrame, observed_rho: float) -> float:
    ordered = segments.sort_values("start_m").reset_index(drop=True)
    risk_group = ordered["risk_group"].to_numpy(dtype=int)
    counts = ordered["object_count"].to_numpy(dtype=int)
    lengths = ordered["length_m"].to_numpy(dtype=float) / 1000.0
    null_values = []
    for shift in range(1, len(ordered)):
        shifted_counts = np.roll(counts, shift)
        densities = []
        for group in range(1, N_GROUPS + 1):
            mask = risk_group == group
            densities.append(float(np.sum(shifted_counts[mask]) / np.sum(lengths[mask])))
        rho = spearmanr(np.arange(1, N_GROUPS + 1), densities).statistic
        if np.isfinite(rho):
            null_values.append(float(rho))
    null_array = np.asarray(null_values, dtype=float)
    return float((1 + np.sum(null_array >= observed_rho)) / (1 + len(null_array)))


def continuous_circular_shift_test(segments: pd.DataFrame, observed_beta: float) -> float:
    ordered = segments.sort_values("start_m").reset_index(drop=True)
    risk_score = ordered["risk_score"].to_numpy(dtype=float)
    counts = ordered["object_count"].to_numpy(dtype=float)
    length_km = ordered["length_m"].to_numpy(dtype=float) / 1000.0
    null_beta = []
    for shift in range(1, len(ordered)):
        shifted_counts = np.roll(counts, shift)
        result = poisson_irr(shifted_counts, risk_score, length_km)
        if result["converged"] and np.isfinite(result["beta"]):
            null_beta.append(float(result["beta"]))
    null_array = np.asarray(null_beta, dtype=float)
    return float((1 + np.sum(null_array >= observed_beta)) / (1 + len(null_array)))


def build_statistics() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    segments = pd.read_csv(SEGMENT_CSV)
    objects = pd.read_csv(OBJECT_CSV)
    geometries = read_segment_geometries(VALIDATION_GPKG)
    covered = segments.loc[segments["coverage"].astype(int).eq(1)].copy()
    covered = covered.sort_values(["risk_score", "segment_id"]).reset_index(drop=True)
    covered["risk_group"] = pd.qcut(
        covered["risk_score"].rank(method="first"),
        q=N_GROUPS,
        labels=np.arange(1, N_GROUPS + 1),
    ).astype(int)

    assignments = assign_objects_uniquely(covered, objects, geometries)
    object_counts = assignments.groupby("segment_id").size().rename("object_count")
    covered = covered.merge(object_counts, how="left", left_on="segment_id", right_index=True)
    covered["object_count"] = covered["object_count"].fillna(0).astype(int)

    rows = []
    for group in range(1, N_GROUPS + 1):
        subset = covered.loc[covered["risk_group"].eq(group)]
        count = int(subset["object_count"].sum())
        length_km = float(subset["length_m"].sum() / 1000.0)
        density = count / length_km
        ci_lower, ci_upper = poisson_exact_interval(count, length_km)
        rows.append(
            {
                "risk_group": group,
                "risk_group_label": GROUP_LABELS[group - 1].replace("\n", " "),
                "segment_count": int(len(subset)),
                "risk_score_min": float(subset["risk_score"].min()),
                "risk_score_max": float(subset["risk_score"].max()),
                "risk_score_mean": float(subset["risk_score"].mean()),
                "covered_length_km": length_km,
                "instability_object_count": count,
                "objects_per_km": density,
                "density_ci95_lower": ci_lower,
                "density_ci95_upper": ci_upper,
            }
        )
    summary = pd.DataFrame(rows)
    rho_result = spearmanr(summary["risk_group"], summary["objects_per_km"])
    rho = float(rho_result.statistic)
    grouped_shift_p = circular_shift_test(covered, rho)
    model = poisson_irr(
        covered["object_count"].to_numpy(dtype=float),
        covered["risk_score"].to_numpy(dtype=float),
        covered["length_m"].to_numpy(dtype=float) / 1000.0,
    )
    continuous_shift_p = continuous_circular_shift_test(covered, model["beta"])
    statistics = {
        "covered_segments": int(len(covered)),
        "assigned_objects": int(len(assignments)),
        "spearman_rho": rho,
        "grouped_circular_shift_p": grouped_shift_p,
        "continuous_circular_shift_p": continuous_shift_p,
        **model,
    }
    assignments = assignments.merge(
        covered[["segment_id", "risk_group"]], on="segment_id", how="left"
    )
    return summary, assignments, statistics


def plot(summary: pd.DataFrame, statistics: dict) -> None:
    font_name = configure_chinese_font()
    x = np.arange(1, N_GROUPS + 1)
    density = summary["objects_per_km"].to_numpy(dtype=float)
    lower = summary["density_ci95_lower"].to_numpy(dtype=float)
    upper = summary["density_ci95_upper"].to_numpy(dtype=float)
    y_error = np.vstack([density - lower, upper - density])
    color_map = plt.get_cmap("RdYlGn_r")
    colors = [color_map(value) for value in np.linspace(0.08, 0.92, N_GROUPS)]

    figure_width = 11.2 if N_GROUPS <= 5 else 13.2
    figure, axis = plt.subplots(figsize=(figure_width, 7.0), facecolor="white")
    axis.set_facecolor("#fbfbf9")
    axis.plot(x, density, color="#4a4a4a", linewidth=2.0, zorder=2)
    axis.errorbar(
        x,
        density,
        yerr=y_error,
        fmt="none",
        ecolor="#777777",
        elinewidth=1.4,
        capsize=5,
        capthick=1.4,
        zorder=1,
    )
    axis.scatter(x, density, s=145, c=colors, edgecolor="white", linewidth=1.5, zorder=3)

    vertical_offset = max(float(np.max(upper)) * 0.035, 0.8)
    for row, x_value, y_value in zip(summary.itertuples(index=False), x, density):
        axis.text(
            x_value,
            y_value + vertical_offset,
            f"{y_value:.1f} 个/km\n(n={row.instability_object_count})",
            ha="center",
            va="bottom",
            fontsize=10.5 if N_GROUPS <= 5 else 9.2,
            color="#333333",
        )

    score_range_labels = [
        f"第{int(row.risk_group)}组\n{row.risk_score_min:.1f}–{row.risk_score_max:.1f}"
        for row in summary.itertuples(index=False)
    ]
    axis.set_xticks(x, score_range_labels)
    segment_min = int(summary["segment_count"].min())
    segment_max = int(summary["segment_count"].max())
    segment_text = str(segment_min) if segment_min == segment_max else f"{segment_min}–{segment_max}"
    axis.set_xlabel(
        f"基础风险得分排序组（每组{segment_text}个50 m分段）",
        fontsize=12.5,
        labelpad=12,
    )
    axis.set_ylabel("结构失稳对象密度（个/km）", fontsize=12.5, labelpad=10)
    axis.set_xlim(0.65, N_GROUPS + 0.35)
    axis.set_ylim(0, float(np.max(upper)) * 1.25)
    axis.grid(axis="y", color="#d8d8d8", linewidth=0.8, alpha=0.65)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#777777")
    axis.spines["bottom"].set_color("#777777")
    axis.tick_params(axis="both", labelsize=11)

    figure.suptitle(
        "广武长城基础风险得分排序与结构失稳对象密度",
        fontsize=18,
        fontweight="bold",
        y=0.975,
    )
    axis.set_title(
        f"50 m关联尺度｜按基础风险得分由低到高划分为{N_GROUPS}个等数量组｜每个失稳对象唯一计数",
        fontsize=11.5,
        color="#555555",
        pad=18,
    )
    statistics_text = (
        f"连续计数回归：IRR = {statistics['irr']:.2f} "
        f"（95% CI {statistics['irr_lower']:.2f}–{statistics['irr_upper']:.2f}）\n"
        f"沿长城循环位移检验 p = {statistics['continuous_circular_shift_p']:.3f}\n"
        f"{N_GROUPS}组密度趋势：Spearman ρ = {statistics['spearman_rho']:.3f}"
    )
    axis.text(
        0.03,
        0.965,
        statistics_text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        linespacing=1.55,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": "#c9c9c9", "alpha": 0.94},
    )
    figure.text(
        0.5,
        0.018,
        "注：横轴各组仅用于统计比较，不对应原风险等级；组下方数字为实际基础风险得分范围，误差线为Poisson精确95%置信区间。",
        ha="center",
        fontsize=9.5,
        color="#666666",
    )
    figure.tight_layout(rect=(0.04, 0.06, 0.98, 0.93))
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"font={font_name}")


def main() -> None:
    summary, assignments, statistics = build_statistics()
    summary.to_csv(TABLE_PATH, index=False, encoding="utf-8-sig")
    assignments.to_csv(ASSIGNMENT_PATH, index=False, encoding="utf-8-sig")
    regression_row = {
        "model": "Poisson count regression with log(length_km) offset and sandwich-robust SE",
        "association_distance_m": ASSOCIATION_DISTANCE_M,
        "covered_segment_count": statistics["covered_segments"],
        "assigned_instability_object_count": statistics["assigned_objects"],
        "predictor": "standardized continuous base risk score",
        "coefficient_beta": statistics["beta"],
        "robust_se": statistics["robust_se"],
        "irr_per_1sd_risk_score": statistics["irr"],
        "irr_ci95_lower": statistics["irr_lower"],
        "irr_ci95_upper": statistics["irr_upper"],
        "robust_wald_p": statistics["p_value"],
        "along_wall_circular_shift_p": statistics["continuous_circular_shift_p"],
        "pearson_dispersion": statistics["pearson_dispersion"],
        "converged": statistics["converged"],
    }
    pd.DataFrame([regression_row]).to_csv(REGRESSION_PATH, index=False, encoding="utf-8-sig")
    plot(summary, statistics)
    print(summary.to_string(index=False))
    print(statistics)
    print(OUTPUT_STEM.with_suffix(".png"))


if __name__ == "__main__":
    main()
