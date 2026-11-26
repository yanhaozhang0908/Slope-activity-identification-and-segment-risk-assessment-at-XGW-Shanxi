from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import plot_wall_body_risk_and_instability_maps as base


ROOT = Path(r"D:\project\shuozhouinflucefactor")
VALIDATION_DIR = ROOT / "process" / "wall_risk_independent_instability_validation"
OUT_DIR = ROOT / "process" / "wall_body_risk_maps"
DENSITY_CSV = VALIDATION_DIR / "risk_quantile_instability_density_50m.csv"
REGRESSION_CSV = VALIDATION_DIR / "continuous_risk_score_count_regression_50m.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def plot_full_risk_map(axis, segments: list[dict]):
    lines = [item["line"] for item in segments]
    risks = np.asarray([item["risk"] for item in segments], dtype=float)
    bounds = base.bounds_from_lines(lines, margin=280.0)
    base.draw_relief(axis, bounds)
    collection = LineCollection(
        lines, cmap=base.RISK_CMAP, norm=base.RISK_NORM, linewidths=5.0,
        capstyle="round", joinstyle="round", zorder=5,
    )
    collection.set_array(risks)
    axis.add_collection(collection)
    base.style_axis(axis, bounds)
    base.add_north_arrow(axis)
    base.add_scale_bar(axis, bounds)
    axis.set_title("（a）独立风险评价：全段风险得分", loc="left", fontsize=13,
                   fontweight="bold", pad=9)
    axis.text(
        0.02, 0.985, "结构失稳数据未参与", transform=axis.transAxes,
        ha="left", va="top", fontsize=9, color="#7f1515", fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "#c8c8c8", "alpha": 0.88, "pad": 4},
        zorder=12,
    )
    return collection, risks


def plot_instability_location_map(
    axis, segments: list[dict], objects: list[dict], coverage, segment_status: dict[int, dict]
) -> None:
    coverage_lines = base.geometry_lines(coverage)
    bounds = base.bounds_from_lines(coverage_lines, margin=230.0)
    xmin, xmax, ymin, ymax = bounds
    selected = []
    for item in segments:
        envelope = item["geometry"].GetEnvelope()
        if envelope[1] >= xmin and envelope[0] <= xmax and envelope[3] >= ymin and envelope[2] <= ymax:
            selected.append(item)
    covered = [
        item for item in selected
        if segment_status.get(item["segment_id"], {}).get("coverage", False)
    ]
    uncovered = [item for item in selected if item not in covered]

    base.draw_relief(axis, bounds)
    if uncovered:
        axis.add_collection(
            LineCollection(
                [item["line"] for item in uncovered], colors="#929292",
                linewidths=2.4, linestyles="dashed", alpha=0.88, zorder=2,
            )
        )
    if covered:
        axis.add_collection(
            LineCollection(
                [item["line"] for item in covered], colors="#4c9fb8",
                linewidths=9.0, alpha=0.28, capstyle="round", zorder=3,
            )
        )
        risks = np.asarray([item["risk"] for item in covered], dtype=float)
        collection = LineCollection(
            [item["line"] for item in covered], cmap=base.RISK_CMAP,
            norm=base.RISK_NORM, linewidths=4.8, capstyle="round",
            joinstyle="round", zorder=4,
        )
        collection.set_array(risks)
        axis.add_collection(collection)

    sizes = 22.0 + 5.0 * np.sqrt([item["source_records"] for item in objects])
    axis.scatter(
        [item["x"] for item in objects], [item["y"] for item in objects],
        s=sizes, marker="o", facecolor="white", edgecolor="#a50026",
        linewidth=1.35, alpha=0.96, zorder=8,
    )
    base.style_axis(axis, bounds)
    base.add_north_arrow(axis, x=0.91, y=0.89)
    base.add_scale_bar(axis, bounds)
    axis.set_title("（b）空间验证：结构失稳对象位置", loc="left", fontsize=13,
                   fontweight="bold", pad=9)
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                   markeredgecolor="#a50026", markeredgewidth=1.3, markersize=7,
                   label=f"独立结构失稳对象（n={len(objects)}）"),
            Line2D([0], [0], color="#4c9fb8", linewidth=7, alpha=0.35,
                   label="结构失稳算法有效覆盖区"),
            Line2D([0], [0], color="#929292", linewidth=2.2, linestyle="--",
                   label="未覆盖，结构状态未知"),
        ],
        loc="upper left", fontsize=8.5, frameon=True, framealpha=0.91,
    )
    axis.text(
        0.98, 0.02, "长城颜色与（a）采用同一风险得分标尺",
        transform=axis.transAxes, ha="right", va="bottom", fontsize=8.3,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
        zorder=10,
    )


def plot_density_relation(axis) -> None:
    rows = read_csv_rows(DENSITY_CSV)
    regression = read_csv_rows(REGRESSION_CSV)[0]
    positions = np.arange(1, len(rows) + 1, dtype=float)
    means = np.asarray([float(row["risk_score_mean"]) for row in rows])
    densities = np.asarray([float(row["objects_per_km"]) for row in rows])
    ci_lower = np.asarray([float(row["density_ci95_lower"]) for row in rows])
    ci_upper = np.asarray([float(row["density_ci95_upper"]) for row in rows])
    counts = np.asarray([int(row["instability_object_count"]) for row in rows])
    colors = [base.RISK_CMAP(base.RISK_NORM(value)) for value in means]

    axis.plot(positions, densities, color="#484848", linewidth=2.1, zorder=2)
    axis.errorbar(
        positions, densities, yerr=[densities - ci_lower, ci_upper - densities],
        fmt="none", ecolor="#777777", elinewidth=1.35, capsize=4, capthick=1.2,
        zorder=1,
    )
    axis.scatter(
        positions, densities, s=105, c=colors, edgecolor="white",
        linewidth=1.2, zorder=3,
    )
    for x, density, count in zip(positions, densities, counts):
        axis.text(x, density + 1.2, f"{density:.1f} 个/km\n（n={count}）",
                  ha="center", va="bottom", fontsize=8.8)

    labels = [
        f"{row['risk_group_label']}\n{float(row['risk_score_min']):.1f}–{float(row['risk_score_max']):.1f}"
        for row in rows
    ]
    axis.set_xticks(positions, labels)
    axis.set_xlim(0.65, len(rows) + 0.35)
    axis.set_ylim(0, 43)
    axis.set_ylabel("结构失稳对象密度（个/km）")
    axis.set_xlabel("基础风险得分排序组（每组18个50 m覆盖分段）")
    axis.set_title("（c）定量关联：风险得分与失稳对象密度", loc="left",
                   fontsize=13, fontweight="bold", pad=9)
    axis.grid(axis="y", alpha=0.24)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=9)

    irr = float(regression["irr_per_1sd_risk_score"])
    ci_low = float(regression["irr_ci95_lower"])
    ci_high = float(regression["irr_ci95_upper"])
    shift_p = float(regression["along_wall_circular_shift_p"])
    assigned = int(regression["assigned_instability_object_count"])
    axis.text(
        0.03, 0.97,
        f"连续风险得分回归：IRR={irr:.2f}（95% CI {ci_low:.2f}–{ci_high:.2f}）\n"
        f"沿长城循环位移检验：p={shift_p:.3f}\n"
        f"50 m范围内唯一归属对象：n={assigned}",
        transform=axis.transAxes, ha="left", va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#bdbdbd", "alpha": 0.93, "pad": 5},
    )
    axis.text(
        0.5, -0.18,
        "五组仅用于直观展示；主要证据为连续风险得分计数回归。误差线为Poisson精确95%置信区间。",
        transform=axis.transAxes, ha="center", va="top", fontsize=8.2,
        color="#666666",
    )


def plot_merged_risk_instability_map(
    axis, segments: list[dict], objects: list[dict], segment_status: dict[int, dict]
):
    lines = [item["line"] for item in segments]
    risks = np.asarray([item["risk"] for item in segments], dtype=float)
    bounds = base.bounds_from_lines(lines, margin=280.0)
    covered = [
        item for item in segments
        if segment_status.get(item["segment_id"], {}).get("coverage", False)
    ]

    base.draw_relief(axis, bounds)
    if covered:
        axis.add_collection(
            LineCollection(
                [item["line"] for item in covered], colors="#2995b3",
                linewidths=10.5, alpha=0.26, capstyle="round",
                joinstyle="round", zorder=3,
            )
        )

    collection = LineCollection(
        lines, cmap=base.RISK_CMAP, norm=base.RISK_NORM, linewidths=5.2,
        capstyle="round", joinstyle="round", zorder=5,
    )
    collection.set_array(risks)
    axis.add_collection(collection)

    sizes = 17.0 + 3.2 * np.sqrt([item["source_records"] for item in objects])
    axis.scatter(
        [item["x"] for item in objects], [item["y"] for item in objects],
        s=sizes, marker="o", facecolor="none", edgecolor="#a50026",
        linewidth=0.95, alpha=0.94, zorder=7,
    )

    base.style_axis(axis, bounds)
    base.add_north_arrow(axis)
    base.add_scale_bar(axis, bounds)
    axis.set_title("（a）风险分段与结构失稳位置融合图", loc="left",
                   fontsize=13.5, fontweight="bold", pad=9)
    axis.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor="#a50026", markeredgewidth=1.1, markersize=7,
                   label=f"独立结构失稳对象（n={len(objects)}）"),
            Line2D([0], [0], color="#2995b3", linewidth=8, alpha=0.30,
                   label="结构失稳算法有效覆盖区"),
        ],
        loc="upper left", fontsize=8.5, frameon=True, framealpha=0.92,
    )
    axis.text(
        0.02, 0.79,
        "无蓝色晕带区段：未覆盖、结构状态未知\n"
        "显示顺序：覆盖晕带（底）→风险色线（中）→空心失稳点（顶）",
        transform=axis.transAxes, ha="left", va="top", fontsize=8.1,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2},
        zorder=10,
    )
    return collection, risks


def plot_two_panel_merged_figure(
    segments: list[dict], objects: list[dict], segment_status: dict[int, dict]
) -> None:
    figure = plt.figure(figsize=(15.8, 10.4))
    grid = figure.add_gridspec(
        1, 2, width_ratios=[0.82, 1.26],
        left=0.06, right=0.975, bottom=0.145, top=0.84, wspace=0.25,
    )
    map_axis = figure.add_subplot(grid[0, 0])
    density_axis = figure.add_subplot(grid[0, 1])

    collection, risks = plot_merged_risk_instability_map(
        map_axis, segments, objects, segment_status
    )
    plot_density_relation(density_axis)
    density_axis.set_title("（b）连续风险得分与结构失稳密度关联", loc="left",
                           fontsize=13.5, fontweight="bold", pad=9)
    base.add_risk_colorbar(
        figure, map_axis, collection, float(risks.min()), float(risks.max())
    )

    figure.suptitle("广武长城风险分段与结构失稳空间—统计关联", fontsize=19,
                    fontweight="bold", y=0.975)
    figure.text(
        0.5, 0.925,
        "基础风险得分保持连续显示  +  空心失稳点标示实际位置  →  采用连续回归与五组密度检验一致性",
        ha="center", va="center", fontsize=11.2, color="#6f1515", fontweight="bold",
        bbox={"boxstyle": "round,pad=0.42", "facecolor": "#fbf6ef",
              "edgecolor": "#d8b7a3", "alpha": 0.96},
    )
    figure.text(
        0.5, 0.89,
        "结构失稳仅作为独立验证数据；空心符号不填充、风险色线绘于覆盖晕带之上，避免遮挡分段得分",
        ha="center", va="center", fontsize=9.4, color="#555555",
    )

    output_png = OUT_DIR / "广武长城_风险分段与结构失稳位置融合_密度关联图.png"
    output_pdf = OUT_DIR / "广武长城_风险分段与结构失稳位置融合_密度关联图.pdf"
    figure.savefig(output_png, dpi=400, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)
    print(f"MERGED_PNG={output_png}")
    print(f"MERGED_PDF={output_pdf}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Microsoft YaHei",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    segments = base.read_risk_segments()
    objects, coverage, segment_status = base.read_validation_layers()

    plot_two_panel_merged_figure(segments, objects, segment_status)

    figure = plt.figure(figsize=(19.2, 10.8))
    grid = figure.add_gridspec(
        1, 3, width_ratios=[0.82, 0.86, 1.28],
        left=0.045, right=0.985, bottom=0.14, top=0.84, wspace=0.25,
    )
    risk_axis = figure.add_subplot(grid[0, 0])
    location_axis = figure.add_subplot(grid[0, 1])
    density_axis = figure.add_subplot(grid[0, 2])

    collection, risks = plot_full_risk_map(risk_axis, segments)
    plot_instability_location_map(location_axis, segments, objects, coverage, segment_status)
    plot_density_relation(density_axis)
    base.add_risk_colorbar(
        figure, risk_axis, collection, float(risks.min()), float(risks.max())
    )

    figure.suptitle("广武长城本体风险评价与结构失稳独立验证", fontsize=19,
                    fontweight="bold", y=0.975)
    figure.text(
        0.5, 0.925,
        "独立风险评价（不含结构失稳）  →  有效覆盖区内叠加失稳位置  →  连续回归与五组密度检验",
        ha="center", va="center", fontsize=11.5, color="#6f1515", fontweight="bold",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fbf6ef",
              "edgecolor": "#d8b7a3", "alpha": 0.96},
    )
    figure.text(
        0.5, 0.89,
        "50 m长城分段；风险评价范围为长城两侧100 m；结构失稳仅作为独立验证数据，未参与风险指标、权重或得分计算",
        ha="center", va="center", fontsize=9.4, color="#555555",
    )

    output_png = OUT_DIR / "广武长城_本体风险评价_结构失稳位置与密度关联综合图.png"
    output_pdf = OUT_DIR / "广武长城_本体风险评价_结构失稳位置与密度关联综合图.pdf"
    figure.savefig(output_png, dpi=400, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)
    print(f"PNG={output_png}")
    print(f"PDF={output_pdf}")


if __name__ == "__main__":
    main()
