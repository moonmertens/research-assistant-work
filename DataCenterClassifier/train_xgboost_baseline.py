from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


ID_COLUMNS = ["statefp10", "countyfp10", "tractce10", "county_id", "tract_id"]
TARGET_COLUMN = "everhyper"


# Command-line configuration.
# Parse command-line options and default locations for data and outputs.
def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    outputs_dir = project_dir / "outputs" / "baseline_xgboost"

    parser = argparse.ArgumentParser(
        description="Train a county-grouped baseline XGBoost model for hyperscale data center siting likelihood."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=data_dir / "everhyper_cleaned.csv",
        help="Path to the cleaned modeling dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=outputs_dir,
        help="Directory where model outputs will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Share of counties reserved for the held-out test set.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Share of training counties reserved for early-stopping validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for grouped splitting and model training.",
    )
    parser.add_argument(
        "--tuned-params-dir",
        type=Path,
        default=None,
        help="Directory containing best_params_constrained.json and best_params_unconstrained.json "
             "produced by tune_xgboost.py. When provided, these params override the hardcoded "
             "hyperparameters for the corresponding model variant.",
    )
    return parser.parse_args()


# Data loading and schema checks.
# Load the cleaned dataset and preserve identifiers exactly.
def read_modeling_data(input_path: Path) -> pd.DataFrame:
    dtype_map = {column: "string" for column in ID_COLUMNS}
    return pd.read_csv(input_path, dtype=dtype_map)


# Ensure the cleaned file contains the fields required for grouped modeling.
def validate_modeling_columns(dataframe: pd.DataFrame) -> None:
    required_columns = [*ID_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required modeling columns: {missing_columns}")


# Feature, target, and group construction.
# Split the cleaned data into predictors, target, and county groups.
def build_modeling_matrices(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    feature_columns = [column for column in dataframe.columns if column not in [*ID_COLUMNS, TARGET_COLUMN]]
    x = dataframe[feature_columns].copy()
    y = dataframe[TARGET_COLUMN].astype("int8")
    groups = dataframe["county_id"].astype("string")
    ids = dataframe[ID_COLUMNS + [TARGET_COLUMN]].copy()
    return x, y, groups, ids


# County-level summaries and grouped splitting.
# Aggregate tract outcomes to the county level so grouped splits can also be approximately stratified.
def build_group_summary(groups: pd.Series, target: pd.Series) -> pd.DataFrame:
    group_summary = pd.DataFrame({"county_id": groups, TARGET_COLUMN: target}).groupby("county_id", as_index=False).agg(
        tract_count=(TARGET_COLUMN, "size"),
        positive_count=(TARGET_COLUMN, "sum"),
    )
    group_summary["has_positive"] = (group_summary["positive_count"] > 0).astype(int)
    return group_summary


# Create a county-grouped split while stratifying on whether a county has any positive tracts.
def make_stratified_grouped_split(
    groups: pd.Series,
    target: pd.Series,
    holdout_size: float,
    random_state: int,
) -> tuple[pd.Index, pd.Index, dict[str, int | str | None]]:
    group_summary = build_group_summary(groups, target)
    stratify_labels = group_summary["has_positive"]

    if stratify_labels.nunique() < 2 or stratify_labels.value_counts().min() < 2:
        stratify_argument = None
        strategy = "grouped_only"
    else:
        stratify_argument = stratify_labels
        strategy = "grouped_plus_county_has_positive_stratification"

    train_groups, holdout_groups = train_test_split(
        group_summary["county_id"],
        test_size=holdout_size,
        random_state=random_state,
        stratify=stratify_argument,
    )

    train_index = pd.Index(groups.index[groups.isin(train_groups)])
    holdout_index = pd.Index(groups.index[groups.isin(holdout_groups)])
    split_metadata = {
        "strategy": strategy,
        "train_group_count": int(len(train_groups)),
        "holdout_group_count": int(len(holdout_groups)),
        "train_positive_group_count": int(group_summary.loc[group_summary["county_id"].isin(train_groups), "has_positive"].sum()),
        "holdout_positive_group_count": int(group_summary.loc[group_summary["county_id"].isin(holdout_groups), "has_positive"].sum()),
    }
    return train_index, holdout_index, split_metadata


# Class weighting and XGBoost configuration.
# Compute the class-imbalance weight from the fitting sample only.
def compute_scale_pos_weight(target: pd.Series) -> float:
    positive_count = int(target.sum())
    negative_count = int((1 - target).sum())
    if positive_count == 0:
        raise ValueError("Training split contains zero positive examples.")
    return negative_count / positive_count


# Monotonic constraints grounded in hyperscale data center site economics.
# +1: higher values strictly increase siting likelihood.
# -1: higher values strictly decrease siting likelihood.
MONOTONIC_CONSTRAINTS = {
    # Power infrastructure — more capacity is always strictly better.
    "substation_avg_voltage_kv20": 1,
    "transmission_hv_km25": 1,
    # Connectivity — fiber is a hard operational requirement.
    "fiber_fiber_km25": 1,
    # Operating cost — industrial electricity price directly compresses margin.
    "electricity_wave_price_industria": -1,
    # Operational risk — FEMA disaster declarations threaten five-nines uptime.
    "disaster_totaldeclare": -1,
    # Regulatory risk — EPA non-attainment blocks backup-generator air permits.
    "envirreg_nonatt_any": -1,
}


# Build a baseline XGBoost classifier tuned for sparse positive outcomes.
# When `params` is provided (e.g. from tune_xgboost.py) it overrides the hardcoded defaults.
def build_model(
    scale_pos_weight: float,
    random_state: int,
    use_constraints: bool = True,
    params: dict | None = None,
) -> XGBClassifier:
    base_params: dict = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.0,
    }
    if params is not None:
        base_params.update(params)
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric=["aucpr", "logloss"],
        tree_method="hist",
        missing=float("nan"),
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=50,
        monotone_constraints=MONOTONIC_CONSTRAINTS if use_constraints else {},
        **base_params,
    )


# Probability calibration helpers.
# Convert probabilities to logits for sigmoid calibration while avoiding infinities.
def probabilities_to_logits(probabilities: pd.Series | np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


# Choose a calibration method based on how many positive cases are available.
def choose_calibration_method(observed: pd.Series) -> str:
    positive_count = int(observed.sum())
    if positive_count >= 100:
        return "isotonic"
    return "sigmoid"


# Fit a probability calibrator on held-out predictions.
def fit_probability_calibrator(
    predicted_probability: pd.Series,
    observed: pd.Series,
) -> tuple[IsotonicRegression | LogisticRegression, str]:
    if observed.nunique() < 2:
        raise ValueError("Calibration split must contain both outcome classes.")

    calibration_method = choose_calibration_method(observed)
    if calibration_method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(predicted_probability.to_numpy(), observed.to_numpy())
        return calibrator, calibration_method

    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(probabilities_to_logits(predicted_probability).reshape(-1, 1), observed.to_numpy())
    return calibrator, calibration_method


# Apply the fitted calibration model to raw predicted probabilities.
def apply_probability_calibrator(
    calibrator: IsotonicRegression | LogisticRegression,
    calibration_method: str,
    predicted_probability: pd.Series,
) -> pd.Series:
    if calibration_method == "isotonic":
        calibrated = calibrator.predict(predicted_probability.to_numpy())
    else:
        calibrated = calibrator.predict_proba(probabilities_to_logits(predicted_probability).reshape(-1, 1))[:, 1]

    return pd.Series(calibrated, index=predicted_probability.index, name="calibrated_probability")


# Model fitting with an inner grouped validation split.
# Fit the model with a grouped validation set used for early stopping and calibration.
def fit_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    train_groups: pd.Series,
    validation_size: float,
    random_state: int,
    use_constraints: bool = True,
    params: dict | None = None,
) -> tuple[XGBClassifier, dict[str, pd.DataFrame | pd.Series | float | int | dict[str, int | str | None]]]:
    fit_index, validation_index, validation_split_metadata = make_stratified_grouped_split(
        train_groups,
        y_train,
        holdout_size=validation_size,
        random_state=random_state,
    )

    x_fit = x_train.loc[fit_index]
    y_fit = y_train.loc[fit_index]
    x_validation = x_train.loc[validation_index]
    y_validation = y_train.loc[validation_index]

    model = build_model(compute_scale_pos_weight(y_fit), random_state=random_state, use_constraints=use_constraints, params=params)
    model.fit(
        x_fit,
        y_fit,
        eval_set=[(x_validation, y_validation)],
        verbose=False,
    )

    split_details: dict[str, pd.DataFrame | pd.Series | float | int] = {
        "x_fit": x_fit,
        "y_fit": y_fit,
        "x_validation": x_validation,
        "y_validation": y_validation,
        "scale_pos_weight": compute_scale_pos_weight(y_fit),
        "best_iteration": int(model.best_iteration),
        "validation_split_metadata": validation_split_metadata,
    }
    return model, split_details


# Metric computation and calibration diagnostics.
# Compute headline binary-probability metrics for a set of predicted probabilities.
def compute_probability_metrics(observed: pd.Series, predicted_probability: pd.Series) -> dict[str, float]:
    metrics = {
        "roc_auc": float(roc_auc_score(observed, predicted_probability)),
        "pr_auc": float(average_precision_score(observed, predicted_probability)),
        "log_loss": float(log_loss(observed, predicted_probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(observed, predicted_probability)),
        "positive_rate": float(observed.mean()),
        "row_count": float(len(observed)),
        "positive_count": float(observed.sum()),
    }
    return metrics


# Score a partition with the uncalibrated XGBoost model.
def score_uncalibrated_partition(model: XGBClassifier, x: pd.DataFrame, y: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    predicted_probability = pd.Series(model.predict_proba(x)[:, 1], index=x.index, name="uncalibrated_probability")
    return predicted_probability, compute_probability_metrics(y, predicted_probability)


# Build a decile table to compare predicted and observed event rates.
def build_decile_summary(predicted_probability: pd.Series, observed: pd.Series) -> list[dict[str, float | int]]:
    ranked = pd.DataFrame({
        "predicted_probability": predicted_probability,
        "observed": observed,
    }).sort_values("predicted_probability")
    ranked["decile"] = pd.qcut(ranked["predicted_probability"], q=10, labels=False, duplicates="drop")

    summary: list[dict[str, float | int]] = []
    for decile, group in ranked.groupby("decile", observed=False):
        summary.append(
            {
                "decile": int(decile),
                "row_count": int(len(group)),
                "mean_predicted_probability": float(group["predicted_probability"].mean()),
                "observed_rate": float(group["observed"].mean()),
                "positive_count": int(group["observed"].sum()),
            }
        )
    return summary


# Output writing for models, metadata, and scored partitions.
# Save the model, metrics, feature list, calibration object, and partition predictions for downstream review.
def save_outputs(
    output_dir: Path,
    model: XGBClassifier,
    calibrator: IsotonicRegression | LogisticRegression,
    calibration_method: str,
    feature_columns: list[str],
    metrics_payload: dict,
    test_predictions: pd.DataFrame,
    validation_predictions: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(output_dir / "xgboost_baseline_model.json")
    dump(calibrator, output_dir / "probability_calibrator.joblib")
    (output_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    (output_dir / "calibration_metadata.json").write_text(
        json.dumps({"calibration_method": calibration_method}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "baseline_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)


# Train one model variant and save all outputs to the given directory.
def run_variant(
    label: str,
    use_constraints: bool,
    output_dir: Path,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    groups_test: pd.Series,
    ids_train: pd.DataFrame,
    ids_test: pd.DataFrame,
    validation_size: float,
    random_state: int,
    test_split_metadata: dict,
    input_path: Path,
    params: dict | None = None,
) -> None:
    print(f"\n--- Training {label} model ---")

    model, split_details = fit_model(
        x_train,
        y_train,
        groups_train,
        validation_size=validation_size,
        random_state=random_state,
        use_constraints=use_constraints,
        params=params,
    )

    validation_uncalibrated_probabilities, validation_uncalibrated_metrics = score_uncalibrated_partition(
        model,
        split_details["x_validation"],
        split_details["y_validation"],
    )
    calibrator, calibration_method = fit_probability_calibrator(
        validation_uncalibrated_probabilities,
        split_details["y_validation"],
    )

    validation_calibrated_probabilities = apply_probability_calibrator(
        calibrator,
        calibration_method,
        validation_uncalibrated_probabilities,
    )
    test_uncalibrated_probabilities, test_uncalibrated_metrics = score_uncalibrated_partition(model, x_test, y_test)
    test_calibrated_probabilities = apply_probability_calibrator(
        calibrator,
        calibration_method,
        test_uncalibrated_probabilities,
    )

    validation_calibrated_metrics = compute_probability_metrics(
        split_details["y_validation"],
        validation_calibrated_probabilities,
    )
    test_calibrated_metrics = compute_probability_metrics(y_test, test_calibrated_probabilities)

    validation_predictions = ids_train.loc[split_details["x_validation"].index].copy()
    validation_predictions["uncalibrated_probability"] = validation_uncalibrated_probabilities.values
    validation_predictions["calibrated_probability"] = validation_calibrated_probabilities.values

    test_predictions = ids_test.copy()
    test_predictions["uncalibrated_probability"] = test_uncalibrated_probabilities.values
    test_predictions["calibrated_probability"] = test_calibrated_probabilities.values

    metrics_payload = {
        "input_path": str(input_path),
        "monotonic_constraints_applied": use_constraints,
        "monotonic_constraints": MONOTONIC_CONSTRAINTS if use_constraints else {},
        "tuned_params": params,
        "feature_count": len(x_train.columns),
        "train_row_count": len(x_train),
        "validation_row_count": len(split_details["x_validation"]),
        "test_row_count": len(x_test),
        "test_split_metadata": test_split_metadata,
        "validation_split_metadata": split_details["validation_split_metadata"],
        "train_county_count": int(groups_train.nunique()),
        "validation_county_count": int(groups_train.loc[split_details["x_validation"].index].nunique()),
        "test_county_count": int(groups_test.nunique()),
        "scale_pos_weight": float(split_details["scale_pos_weight"]),
        "best_iteration": int(split_details["best_iteration"]),
        "calibration_method": calibration_method,
        "validation_metrics_uncalibrated": validation_uncalibrated_metrics,
        "validation_metrics_calibrated": validation_calibrated_metrics,
        "test_metrics_uncalibrated": test_uncalibrated_metrics,
        "test_metrics_calibrated": test_calibrated_metrics,
        "test_deciles_uncalibrated": build_decile_summary(test_uncalibrated_probabilities, y_test),
        "test_deciles_calibrated": build_decile_summary(test_calibrated_probabilities, y_test),
    }

    save_outputs(
        output_dir=output_dir,
        model=model,
        calibrator=calibrator,
        calibration_method=calibration_method,
        feature_columns=x_train.columns.tolist(),
        metrics_payload=metrics_payload,
        test_predictions=test_predictions,
        validation_predictions=validation_predictions,
    )

    print(f"Saved outputs to: {output_dir}")
    print(f"Calibration method: {calibration_method}")
    print(f"Test PR AUC (uncalibrated):  {test_uncalibrated_metrics['pr_auc']:.6f}")
    print(f"Test ROC AUC (uncalibrated): {test_uncalibrated_metrics['roc_auc']:.6f}")
    print(f"Test Brier score (calibrated): {test_calibrated_metrics['brier_score']:.6f}")


# End-to-end training workflow.
# Run constrained and unconstrained variants and persist both sets of outputs.
def main() -> None:
    args = parse_args()
    base_dir = args.output_dir.parent

    # Load the prepared tract-level data and assemble modeling inputs.
    dataframe = read_modeling_data(args.input)
    validate_modeling_columns(dataframe)
    x, y, groups, ids = build_modeling_matrices(dataframe)

    print(f"Features used: {len(x.columns)}")

    # Create the held-out county test split once, shared by both variants.
    train_index, test_index, test_split_metadata = make_stratified_grouped_split(
        groups,
        y,
        holdout_size=args.test_size,
        random_state=args.random_state,
    )

    x_train = x.loc[train_index]
    y_train = y.loc[train_index]
    groups_train = groups.loc[train_index]
    ids_train = ids.loc[train_index]

    x_test = x.loc[test_index]
    y_test = y.loc[test_index]
    groups_test = groups.loc[test_index]
    ids_test = ids.loc[test_index]

    print(f"Train rows: {len(x_train):,}  |  Test rows: {len(x_test):,}")

    shared = dict(
        x_train=x_train,
        y_train=y_train,
        groups_train=groups_train,
        x_test=x_test,
        y_test=y_test,
        groups_test=groups_test,
        ids_train=ids_train,
        ids_test=ids_test,
        validation_size=args.validation_size,
        random_state=args.random_state,
        test_split_metadata=test_split_metadata,
        input_path=args.input,
    )

    # Load Optuna-tuned hyperparameters if a tuning output directory is provided.
    tuned_params: dict[str, dict | None] = {"constrained": None, "unconstrained": None}
    if args.tuned_params_dir is not None:
        for _label in ("constrained", "unconstrained"):
            _params_path = args.tuned_params_dir / f"best_params_{_label}.json"
            if _params_path.exists():
                _result = json.loads(_params_path.read_text(encoding="utf-8"))
                tuned_params[_label] = _result["best_params"]
                print(
                    f"Loaded tuned params for {_label} "
                    f"(validation PR AUC {_result['best_validation_pr_auc']:.6f})"
                )
            else:
                print(f"Warning: tuned params not found for {_label} at {_params_path}; using defaults.")

    run_variant("constrained",   use_constraints=True,  output_dir=base_dir / "constrained",   params=tuned_params["constrained"],   **shared)
    run_variant("unconstrained", use_constraints=False, output_dir=base_dir / "unconstrained", params=tuned_params["unconstrained"], **shared)


if __name__ == "__main__":
    main()