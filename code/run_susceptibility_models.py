from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(r"D:\project\shuozhouinflucefactor")
sys.path.insert(0, str(ROOT / ".python_packages"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)


INPUT_CSV = ROOT / "process" / "insar" / "slope_unit_insar_stats.csv"
OUTPUT_DIR = ROOT / "process" / "instability_susceptibility"
RESULT_CSV = OUTPUT_DIR / "instability_results.csv"
IV_TABLE_CSV = OUTPUT_DIR / "information_value_table.csv"
FIRTH_COEF_CSV = OUTPUT_DIR / "firth_coefficients.csv"
RF_IMPORTANCE_CSV = OUTPUT_DIR / "rf_feature_importance.csv"
VALIDATION_CSV = OUTPUT_DIR / "validation_metrics.csv"
METADATA_JSON = OUTPUT_DIR / "model_metadata.json"

IV_CONTINUOUS = [
    "slope_mean",
    "relief_mean",
    "drain_dist_mean",
    "ROADdist_m",
    "FAULTdist_m",
]
IV_CATEGORICAL = ["LC_code", "GM_code"]
FIRTH_FEATURES = [
    "slope_mean",
    "drain_dist_mean",
    "ROADdist_m",
    "FAULTdist_m",
    "LC_grass",
]
RF_CONTINUOUS = [
    "slope_mean",
    "relief_mean",
    "drain_dist_mean",
    "ROADdist_m",
    "FAULTdist_m",
]
RF_CATEGORICAL = ["LC_code", "GM_code"]
INSTABILITY_NAMES = {
    0: "未评价",
    1: "极低易发",
    2: "低易发",
    3: "中易发",
    4: "高易发",
    5: "极高易发",
}
PRIORITY_NAMES = {
    0: "未评价",
    1: "低关注",
    2: "较低关注",
    3: "中等关注",
    4: "高关注",
    5: "重点监测",
}
RANDOM_SEED = 42


def prepare_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    data = pd.read_csv(INPUT_CSV)
    data["LC_grass"] = (data["LC_code"] == 11).astype(int)
    data["POS_event"] = ""
    data.loc[data["LS2"] == 1, "POS_event"] = "LS2"
    data.loc[data["LS3"] == 1, "POS_event"] = "LS3"
    data.loc[data["UNSTABLE"] == 1, "POS_event"] = "U1"
    positive = (data["model_keep"] == 1) & data["POS_event"].ne("")
    data["POS_label"] = positive.astype(int)
    stable_negative = (
        (data["model_keep"] == 1)
        & (data["INSAR_stable"] == 1)
        & ~positive
    )
    if positive.sum() < 2:
        raise RuntimeError("Too few positive slope units")
    if stable_negative.sum() < 20:
        raise RuntimeError("Too few InSAR-stable negative slope units")
    return data, positive.to_numpy(), stable_negative.to_numpy()


def iv_fit_predict(
    data: pd.DataFrame,
    train_indices: np.ndarray,
    prediction_indices: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    train = data.iloc[train_indices].copy()
    prediction = data.iloc[prediction_indices].copy()
    positive_total = int(train["target"].sum())
    negative_total = int((1 - train["target"]).sum())
    rows = []
    total_score = np.zeros(len(prediction), dtype="float64")

    for factor in IV_CONTINUOUS:
        _, raw_edges = pd.qcut(
            train[factor],
            q=4,
            duplicates="drop",
            retbins=True,
        )
        edges = np.unique(raw_edges)
        edges[0] = -np.inf
        edges[-1] = np.inf
        train_groups = pd.cut(train[factor], bins=edges, include_lowest=True)
        prediction_groups = pd.cut(
            prediction[factor], bins=edges, include_lowest=True
        )
        categories = train_groups.cat.categories
        group_count = len(categories)
        mapping = {}
        for category in categories:
            members = train_groups == category
            positive_count = int(train.loc[members, "target"].sum())
            negative_count = int(members.sum() - positive_count)
            positive_rate = (positive_count + 0.5) / (
                positive_total + 0.5 * group_count
            )
            negative_rate = (negative_count + 0.5) / (
                negative_total + 0.5 * group_count
            )
            value = float(np.log(positive_rate / negative_rate))
            mapping[category] = value
            rows.append(
                {
                    "factor": factor,
                    "group": str(category),
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "positive_rate_smoothed": positive_rate,
                    "negative_rate_smoothed": negative_rate,
                    "information_value": value,
                }
            )
        total_score += prediction_groups.map(mapping).astype(float).fillna(0).to_numpy()

    for factor in IV_CATEGORICAL:
        train_groups = train[factor].astype("Int64").astype(str)
        prediction_groups = prediction[factor].astype("Int64").astype(str)
        categories = sorted(train_groups.unique())
        group_count = len(categories)
        mapping = {}
        for category in categories:
            members = train_groups == category
            positive_count = int(train.loc[members, "target"].sum())
            negative_count = int(members.sum() - positive_count)
            positive_rate = (positive_count + 0.5) / (
                positive_total + 0.5 * group_count
            )
            negative_rate = (negative_count + 0.5) / (
                negative_total + 0.5 * group_count
            )
            value = float(np.log(positive_rate / negative_rate))
            mapping[category] = value
            rows.append(
                {
                    "factor": factor,
                    "group": category,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "positive_rate_smoothed": positive_rate,
                    "negative_rate_smoothed": negative_rate,
                    "information_value": value,
                }
            )
        total_score += prediction_groups.map(mapping).fillna(0).to_numpy()
    return total_score, pd.DataFrame(rows)


def standardize_matrix(
    train: pd.DataFrame,
    prediction: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_values = train[FIRTH_FEATURES].to_numpy(dtype="float64")
    prediction_values = prediction[FIRTH_FEATURES].to_numpy(dtype="float64")
    means = train_values.mean(axis=0)
    scales = train_values.std(axis=0, ddof=0)
    scales[scales == 0] = 1.0
    train_standard = (train_values - means) / scales
    prediction_standard = (prediction_values - means) / scales
    train_matrix = np.column_stack([np.ones(len(train)), train_standard])
    prediction_matrix = np.column_stack(
        [np.ones(len(prediction)), prediction_standard]
    )
    return train_matrix, prediction_matrix, means, scales


def firth_fit(
    matrix: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    ridge = 1e-9

    def penalized_log_likelihood(coefficients: np.ndarray) -> float:
        eta = np.clip(matrix @ coefficients, -35, 35)
        probability = expit(eta)
        log_likelihood = np.sum(
            target * np.log(probability + 1e-15)
            + (1 - target) * np.log(1 - probability + 1e-15)
        )
        weights = probability * (1 - probability)
        information = matrix.T @ (weights[:, None] * matrix)
        sign, log_determinant = np.linalg.slogdet(
            information + ridge * np.eye(information.shape[0])
        )
        if sign <= 0:
            return -np.inf
        return float(log_likelihood + 0.5 * log_determinant)

    coefficients = np.zeros(matrix.shape[1], dtype="float64")
    success = False
    message = "Maximum iterations reached"
    iteration = 0
    for iteration in range(1, 1001):
        eta = np.clip(matrix @ coefficients, -35, 35)
        probability = expit(eta)
        weights = np.clip(probability * (1 - probability), 1e-12, None)
        information = matrix.T @ (weights[:, None] * matrix)
        inverse_information = np.linalg.pinv(
            information + ridge * np.eye(information.shape[0])
        )
        leverage = weights * np.einsum(
            "ij,jk,ik->i", matrix, inverse_information, matrix
        )
        adjusted_score = matrix.T @ (
            target - probability + leverage * (0.5 - probability)
        )
        step = inverse_information @ adjusted_score
        maximum_step = float(np.max(np.abs(step)))
        if maximum_step > 5:
            step *= 5 / maximum_step

        current_value = penalized_log_likelihood(coefficients)
        candidate = coefficients + step
        candidate_value = penalized_log_likelihood(candidate)
        halvings = 0
        while candidate_value < current_value and halvings < 30:
            step *= 0.5
            candidate = coefficients + step
            candidate_value = penalized_log_likelihood(candidate)
            halvings += 1
        change = float(np.max(np.abs(candidate - coefficients)))
        coefficients = candidate
        if change < 1e-8:
            success = True
            message = "Converged by modified-score iteration"
            break

    probability = expit(np.clip(matrix @ coefficients, -35, 35))
    weights = probability * (1 - probability)
    information = matrix.T @ (weights[:, None] * matrix)
    covariance = np.linalg.pinv(information)
    standard_errors = np.sqrt(np.diag(covariance))
    diagnostics = {
        "success": success,
        "message": message,
        "iterations": iteration,
        "penalized_log_likelihood": penalized_log_likelihood(coefficients),
    }
    return coefficients, standard_errors, diagnostics


def make_rf_matrix(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    continuous = data[RF_CONTINUOUS].astype(float)
    categorical = pd.get_dummies(
        data[RF_CATEGORICAL].astype("Int64").astype(str),
        prefix=RF_CATEGORICAL,
        dtype=float,
    )
    matrix = pd.concat([continuous, categorical], axis=1)
    return matrix, matrix.columns.tolist()


def fit_random_forest(
    matrix: pd.DataFrame,
    target: np.ndarray,
    seed: int,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=4,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(matrix, target)
    return model


def normalized_rank(values: pd.Series, mask: np.ndarray) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    result.loc[mask] = values.loc[mask].rank(method="average", pct=True)
    return result


def evaluate_model(
    name: str,
    target: np.ndarray,
    scores: np.ndarray,
    scope: str,
    fold: str,
) -> dict:
    threshold = float(np.quantile(scores, 0.5))
    predicted = (scores >= threshold).astype(int)
    return {
        "model": name,
        "scope": scope,
        "fold": fold,
        "sample_count": len(target),
        "positive_count": int(target.sum()),
        "negative_count": int((1 - target).sum()),
        "roc_auc": float(roc_auc_score(target, scores)),
        "balanced_accuracy_at_median": float(
            balanced_accuracy_score(target, predicted)
        ),
        "note": (
            "训练样本内评价，偏乐观"
            if scope == "training"
            else "按不稳定事件组及邻近稳定负样本进行空间留出；样本极少"
        ),
    }


def event_spatial_validation(
    data: pd.DataFrame,
    rf_matrix: pd.DataFrame,
) -> list[dict]:
    records = []
    event_centers = {}
    events = ("LS2", "LS3", "U1")
    for event in events:
        positives = data[
            (data["model_keep"] == 1) & (data["POS_event"] == event)
        ]
        event_centers[event] = np.array(
            [positives["Xc"].mean(), positives["Yc"].mean()]
        )

    stable = data[
        (data["model_keep"] == 1) & (data["INSAR_stable"] == 1)
    ].copy()
    stable_coordinates = stable[["Xc", "Yc"]].to_numpy()
    test_negative_count = min(20, len(stable) // 3)

    for fold_number, held_event in enumerate(events, start=1):
        train_events = tuple(event for event in events if event != held_event)
        train_positive_indices = data.index[
            (data["model_keep"] == 1) & data["POS_event"].isin(train_events)
        ]
        test_positive_indices = data.index[
            (data["model_keep"] == 1) & (data["POS_event"] == held_event)
        ]
        held_distances = np.linalg.norm(
            stable_coordinates - event_centers[held_event], axis=1
        )
        nearest_positions = np.argsort(held_distances)[:test_negative_count]
        test_negative_indices = stable.index[nearest_positions]
        train_negative_indices = stable.index.difference(test_negative_indices)
        train_indices = np.concatenate(
            [train_positive_indices.to_numpy(), train_negative_indices.to_numpy()]
        )
        test_indices = np.concatenate(
            [test_positive_indices.to_numpy(), test_negative_indices.to_numpy()]
        )
        if (
            len(train_positive_indices) == 0
            or len(test_positive_indices) == 0
            or len(train_negative_indices) == 0
            or len(test_negative_indices) == 0
        ):
            continue
        train_target = data.loc[train_indices, "target"].to_numpy(dtype=int)
        test_target = data.loc[test_indices, "target"].to_numpy(dtype=int)

        iv_scores, _ = iv_fit_predict(data, train_indices, test_indices)
        records.append(
            evaluate_model(
                "Information value",
                test_target,
                iv_scores,
                "event_spatial_holdout",
                f"hold_{held_event}",
            )
        )

        train_frame = data.loc[train_indices]
        test_frame = data.loc[test_indices]
        train_matrix, test_matrix, _, _ = standardize_matrix(
            train_frame, test_frame
        )
        coefficients, _, diagnostics = firth_fit(train_matrix, train_target)
        firth_scores = expit(np.clip(test_matrix @ coefficients, -35, 35))
        record = evaluate_model(
            "Firth logistic regression",
            test_target,
            firth_scores,
            "event_spatial_holdout",
            f"hold_{held_event}",
        )
        record["note"] += f"; optimizer={diagnostics['message']}"
        records.append(record)

        rf = fit_random_forest(
            rf_matrix.loc[train_indices],
            train_target,
            RANDOM_SEED + fold_number,
        )
        rf_scores = rf.predict_proba(rf_matrix.loc[test_indices])[:, 1]
        records.append(
            evaluate_model(
                "Random forest",
                test_target,
                rf_scores,
                "event_spatial_holdout",
                f"hold_{held_event}",
            )
        )
        ensemble_scores = pd.DataFrame(
            {
                "iv": iv_scores,
                "firth": firth_scores,
                "rf": rf_scores,
            }
        ).rank(method="average", pct=True).mean(axis=1).to_numpy()
        records.append(
            evaluate_model(
                "Ensemble rank",
                test_target,
                ensemble_scores,
                "event_spatial_holdout",
                f"hold_{held_event}",
            )
        )
    return records


def make_plots(
    data: pd.DataFrame,
    training_indices: np.ndarray,
    rf_importance: pd.DataFrame,
    firth_coefficients: pd.DataFrame,
) -> None:
    target = data.loc[training_indices, "target"].to_numpy(dtype=int)
    fig, axis = plt.subplots(figsize=(7.5, 6), dpi=160)
    for field, name, color in [
        ("IV_score", "Information value", "#4C78A8"),
        ("FIRTH_prob", "Firth logistic", "#F58518"),
        ("RF_repeat_mean", "Random forest", "#54A24B"),
        ("ENS_score", "Ensemble rank", "#B279A2"),
    ]:
        scores = data.loc[training_indices, field].to_numpy(dtype=float)
        false_positive, true_positive, _ = roc_curve(target, scores)
        axis.plot(
            false_positive,
            true_positive,
            label=f"{name} (AUC={auc(false_positive, true_positive):.3f})",
            color=color,
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.5")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Training-sample ROC (optimistic; only 3 instability events)")
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_training_roc.png", bbox_inches="tight")
    plt.close(fig)

    top = rf_importance.head(12).sort_values("importance")
    fig, axis = plt.subplots(figsize=(8, 5.5), dpi=160)
    axis.barh(top["feature"], top["importance"], color="#54A24B")
    axis.set_xlabel("Random-forest importance")
    axis.set_title("Exploratory feature importance")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rf_feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    coefficients = firth_coefficients.iloc[1:].sort_values("coefficient")
    fig, axis = plt.subplots(figsize=(8, 5), dpi=160)
    axis.errorbar(
        coefficients["coefficient"],
        coefficients["term"],
        xerr=1.96 * coefficients["std_error"],
        fmt="o",
        color="#F58518",
        ecolor="0.4",
        capsize=3,
    )
    axis.axvline(0, color="0.5", linestyle="--")
    axis.set_xlabel("Coefficient per 1 SD (approximate 95% interval)")
    axis.set_title("Firth penalized logistic regression")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "firth_coefficients.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, positive_mask, stable_negative_mask = prepare_data()
    data["target"] = 0
    data.loc[positive_mask, "target"] = 1
    training_mask = positive_mask | stable_negative_mask
    training_indices = np.flatnonzero(training_mask)
    prediction_indices = np.arange(len(data))
    training_target = data.loc[training_indices, "target"].to_numpy(dtype=int)

    iv_score, iv_table = iv_fit_predict(
        data, training_indices, prediction_indices
    )
    data["IV_score"] = iv_score
    iv_table.to_csv(IV_TABLE_CSV, index=False, encoding="utf-8-sig")

    train_frame = data.loc[training_indices]
    prediction_frame = data.loc[prediction_indices]
    train_matrix, prediction_matrix, means, scales = standardize_matrix(
        train_frame, prediction_frame
    )
    coefficients, standard_errors, firth_diagnostics = firth_fit(
        train_matrix, training_target
    )
    data["FIRTH_prob"] = expit(
        np.clip(prediction_matrix @ coefficients, -35, 35)
    )
    terms = ["Intercept"] + FIRTH_FEATURES
    firth_table = pd.DataFrame(
        {
            "term": terms,
            "coefficient": coefficients,
            "std_error": standard_errors,
            "odds_ratio": np.exp(coefficients),
            "ci95_low": np.exp(coefficients - 1.96 * standard_errors),
            "ci95_high": np.exp(coefficients + 1.96 * standard_errors),
        }
    )
    firth_table.to_csv(FIRTH_COEF_CSV, index=False, encoding="utf-8-sig")

    rf_matrix, rf_feature_names = make_rf_matrix(data)
    rf_model = fit_random_forest(
        rf_matrix.loc[training_indices],
        training_target,
        RANDOM_SEED,
    )
    data["RF_prob"] = rf_model.predict_proba(rf_matrix)[:, 1]

    rng = np.random.default_rng(RANDOM_SEED)
    positive_indices = np.flatnonzero(positive_mask)
    negative_indices = np.flatnonzero(stable_negative_mask)
    repeat_predictions = []
    repeated_negative_count = min(len(negative_indices), len(positive_indices) * 3)
    for repeat in range(100):
        sampled_negative = rng.choice(
            negative_indices, size=repeated_negative_count, replace=False
        )
        sampled = np.concatenate([positive_indices, sampled_negative])
        repeated_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED + repeat + 100,
            n_jobs=1,
        )
        repeated_model.fit(
            rf_matrix.loc[sampled],
            data.loc[sampled, "target"].to_numpy(dtype=int),
        )
        repeat_predictions.append(
            repeated_model.predict_proba(rf_matrix)[:, 1]
        )
    repeat_cube = np.stack(repeat_predictions)
    data["RF_repeat_mean"] = repeat_cube.mean(axis=0)
    data["RF_repeat_std"] = repeat_cube.std(axis=0)

    rf_importance = pd.DataFrame(
        {
            "feature": rf_feature_names,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    rf_importance.to_csv(
        RF_IMPORTANCE_CSV, index=False, encoding="utf-8-sig"
    )

    model_scope = data["model_keep"].to_numpy() == 1
    data["IV_rank"] = normalized_rank(data["IV_score"], model_scope)
    data["FIRTH_rank"] = normalized_rank(data["FIRTH_prob"], model_scope)
    data["RF_rank"] = normalized_rank(data["RF_repeat_mean"], model_scope)
    data["ENS_score"] = data[["IV_rank", "FIRTH_rank", "RF_rank"]].mean(axis=1)
    data["INST_class"] = 0
    scoped_rank = data.loc[model_scope, "ENS_score"].rank(
        method="first", pct=True
    )
    data.loc[model_scope, "INST_class"] = np.ceil(scoped_rank * 5).clip(
        1, 5
    ).astype(int)
    data["INST_name"] = data["INST_class"].map(INSTABILITY_NAMES)

    activity_adjustment = data["INSAR_act"].map({0: 0, 1: 0, 2: 1, 3: 2})
    data["PRIORITY"] = 0
    data.loc[model_scope, "PRIORITY"] = (
        data.loc[model_scope, "INST_class"]
        + activity_adjustment.loc[model_scope]
    ).clip(upper=5)
    data["PRIORITY"] = data["PRIORITY"].astype(int)
    data["PRIORITY_name"] = data["PRIORITY"].map(PRIORITY_NAMES)

    validation_records = []
    for field, name in [
        ("IV_score", "Information value"),
        ("FIRTH_prob", "Firth logistic regression"),
        ("RF_repeat_mean", "Random forest"),
        ("ENS_score", "Ensemble rank"),
    ]:
        validation_records.append(
            evaluate_model(
                name,
                training_target,
                data.loc[training_indices, field].to_numpy(dtype=float),
                "training",
                "all",
            )
        )
    validation_records.extend(event_spatial_validation(data, rf_matrix))
    validation = pd.DataFrame(validation_records)
    validation.to_csv(VALIDATION_CSV, index=False, encoding="utf-8-sig")

    data.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    make_plots(data, training_indices, rf_importance, firth_table)

    high_units = data[
        model_scope & (data["INST_class"] >= 4)
    ]
    metadata = {
        "model_framework": [
            "Information value baseline",
            "Firth penalized logistic regression primary model",
            "Random forest exploratory nonlinear comparator",
            "Equal-weight rank ensemble for static susceptibility",
            "InSAR activity overlay for monitoring priority",
        ],
        "sample_definition": {
            "positive_slope_units": int(positive_mask.sum()),
            "independent_instability_events": 3,
            "positive_event_unit_counts": {
                event: int(
                    ((data["model_keep"] == 1) & (data["POS_event"] == event)).sum()
                )
                for event in ("LS2", "LS3", "U1")
            },
            "stable_negative_slope_units": int(stable_negative_mask.sum()),
            "training_sample_count": int(training_mask.sum()),
            "positive_definition": (
                "complete slope units intersecting 滑坡区域2, 滑坡区域3, "
                "or 非滑坡区域1 (three independent instability event groups)"
            ),
            "negative_definition": (
                "model_keep=1 and INSAR_stable=1; excludes mapped landslide "
                "and C-index-rejected unstable areas"
            ),
        },
        "features": {
            "information_value": IV_CONTINUOUS + IV_CATEGORICAL,
            "firth": FIRTH_FEATURES,
            "random_forest": RF_CONTINUOUS + RF_CATEGORICAL,
            "excluded": [
                "InSAR velocity as a static predictor (label leakage risk)",
                "SOILcode (constant across study domain)",
                "GEO_code (source/legend not verified)",
                "highly collinear DEM derivatives in the same model",
            ],
        },
        "firth_standardization": {
            factor: {"mean": float(mean), "std": float(scale)}
            for factor, mean, scale in zip(FIRTH_FEATURES, means, scales)
        },
        "firth_optimizer": firth_diagnostics,
        "instability_susceptibility_classes": INSTABILITY_NAMES,
        "monitoring_priority_rule": (
            "priority = min(5, static susceptibility class + 0 for stable/"
            "insufficient, +1 for weak/uncertain deformation, +2 for strong deformation)"
        ),
        "high_or_very_high_unit_count": len(high_units),
        "positive_units_in_high_or_very_high": int(
            high_units["POS_label"].sum()
        ),
        "limitations": [
            "Only three independent instability events are available; positive slope units within each event are spatially dependent.",
            "Training ROC-AUC is descriptive and optimistic, not an independent accuracy estimate.",
            "Three-fold leave-one-instability-event-out validation is reported but remains highly uncertain.",
            "Random forest results are exploratory and must not be presented as robust regional generalization.",
            "InSAR is used to support instability labels, stable-negative selection, and dynamic priority, not as a static predictor.",
            "U1 did not pass the original C-index landslide screen; it is treated as a user-confirmed unstable/deforming area, not as a confirmed landslide.",
        ],
    }
    METADATA_JSON.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
