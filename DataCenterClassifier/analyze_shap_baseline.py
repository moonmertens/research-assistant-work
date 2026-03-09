from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from xgboost import Booster, DMatrix


ID_COLUMNS = ["statefp10", "countyfp10", "tractce10", "county_id", "tract_id"]
TARGET_COLUMN = "everhyper"


# Command-line argument setup.
# Command-line configuration for SHAP analysis inputs and outputs.
def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    model_dir = project_dir / "outputs" / "baseline_xgboost"
    shap_dir = model_dir / "shap"

    parser = argparse.ArgumentParser(
        description="Compute Tree SHAP explanations for the baseline XGBoost siting model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=data_dir / "everhyper_cleaned.csv",
        help="Path to the cleaned modeling dataset.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=model_dir / "xgboost_baseline_model.json",
        help="Path to the trained XGBoost model JSON.",
    )
    parser.add_argument(
        "--feature-columns-path",
        type=Path,
        default=model_dir / "feature_columns.json",
        help="Path to the JSON file listing model feature columns.",
    )
    parser.add_argument(
        "--calibrator-path",
        type=Path,
        default=model_dir / "probability_calibrator.joblib",
        help="Optional path to the fitted probability calibrator.",
    )
    parser.add_argument(
        "--calibration-metadata-path",
        type=Path,
        default=model_dir / "calibration_metadata.json",
        help="Optional path to calibration metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=shap_dir,
        help="Directory where SHAP outputs will be saved.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of tracts to explain. Use 0 or a value >= row count to explain all rows.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=25,
        help="Number of top features to retain in the compact row-level contribution file.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used when sampling rows for SHAP analysis.",
    )
    return parser.parse_args()


# Data loading and feature list validation.
# Load the cleaned data while preserving tract identifiers as strings.
def read_modeling_data(input_path: Path) -> pd.DataFrame:
    dtype_map = {column: "string" for column in ID_COLUMNS}
    return pd.read_csv(input_path, dtype=dtype_map)


# Read the ordered feature list used during training.
def load_feature_columns(feature_columns_path: Path) -> list[str]:
    return json.loads(feature_columns_path.read_text(encoding="utf-8"))


# Check that the cleaned dataset contains all features expected by the model.
def validate_feature_columns(dataframe: pd.DataFrame, feature_columns: list[str]) -> None:
    missing_columns = [column for column in feature_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing model feature columns: {missing_columns}")


# Sampling and model loading.
# Take a reproducible sample of tracts to keep SHAP outputs manageable.
def sample_rows(dataframe: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(dataframe):
        return dataframe.copy()
    return dataframe.sample(n=sample_size, random_state=random_state).sort_index()


# Load the saved XGBoost booster directly for Tree SHAP contribution calculations.
def load_model(model_path: Path) -> Booster:
    booster = Booster()
    booster.load_model(model_path)
    return booster


# Probability transformation helpers.
# Convert raw margins to probabilities for easier interpretation.
def margin_to_probability(raw_margin: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-raw_margin))


# Convert probabilities to logits for sigmoid calibration.
def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


# Optional calibration loading and application.
# Load the optional calibration objects used in the training pipeline.
def load_calibration_artifacts(
    calibrator_path: Path,
    calibration_metadata_path: Path,
) -> tuple[IsotonicRegression | LogisticRegression | None, str | None]:
    if not calibrator_path.exists() or not calibration_metadata_path.exists():
        return None, None

    calibrator = load(calibrator_path)
    calibration_metadata = json.loads(calibration_metadata_path.read_text(encoding="utf-8"))
    return calibrator, calibration_metadata.get("calibration_method")


# Apply the saved calibrator when available so row-level outputs include both probability scales.
def apply_probability_calibrator(
    calibrator: IsotonicRegression | LogisticRegression | None,
    calibration_method: str | None,
    uncalibrated_probability: np.ndarray,
) -> np.ndarray | None:
    if calibrator is None or calibration_method is None:
        return None

    if calibration_method == "isotonic":
        return calibrator.predict(uncalibrated_probability)

    return calibrator.predict_proba(probabilities_to_logits(uncalibrated_probability).reshape(-1, 1))[:, 1]


# SHAP computation and summarization.
# Compute Tree SHAP values from the trained booster.
def compute_shap_values(booster: Booster, x: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    dmatrix = DMatrix(x, feature_names=x.columns.tolist(), missing=np.nan)
    contributions = booster.predict(dmatrix, pred_contribs=True, validate_features=True)
    shap_values = pd.DataFrame(contributions[:, :-1], index=x.index, columns=x.columns)
    base_value = contributions[:, -1]
    return shap_values, base_value


# Summarize global importance using average absolute SHAP magnitude.
def build_global_importance(shap_values: pd.DataFrame) -> pd.DataFrame:
    global_importance = pd.DataFrame(
        {
            "feature": shap_values.columns,
            "mean_abs_shap": shap_values.abs().mean().to_numpy(),
            "mean_shap": shap_values.mean().to_numpy(),
        }
    )
    return global_importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# Row-level explanation table construction.
# Build a compact row-level table containing only the strongest feature contributions per tract.
def build_top_contributions(
    sampled_rows: pd.DataFrame,
    shap_values: pd.DataFrame,
    base_value: np.ndarray,
    top_features: int,
    calibrated_probability: np.ndarray | None,
) -> pd.DataFrame:
    top_feature_names = shap_values.abs().mean().sort_values(ascending=False).head(top_features).index.tolist()
    row_level = sampled_rows[ID_COLUMNS + [TARGET_COLUMN]].copy()
    row_level["base_value_margin"] = base_value
    row_level["raw_margin_prediction"] = base_value + shap_values.sum(axis=1).to_numpy()
    row_level["uncalibrated_probability"] = margin_to_probability(row_level["raw_margin_prediction"].to_numpy())
    if calibrated_probability is not None:
        row_level["calibrated_probability"] = calibrated_probability

    for feature_name in top_feature_names:
        row_level[f"shap_{feature_name}"] = shap_values[feature_name].to_numpy()
        row_level[f"value_{feature_name}"] = sampled_rows[feature_name].to_numpy()

    return row_level


# Output writing for SHAP tables and metadata.
# Persist the SHAP summary, global importance, and row-level contribution outputs.
def save_outputs(
    output_dir: Path,
    metadata: dict,
    global_importance: pd.DataFrame,
    row_level_contributions: pd.DataFrame,
    shap_values: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shap_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    global_importance.to_csv(output_dir / "shap_global_importance.csv", index=False)
    row_level_contributions.to_csv(output_dir / "shap_top_contributions.csv", index=False)
    shap_values.to_csv(output_dir / "shap_values_full_sample.csv", index=True, index_label="row_index")


# Plot generation for the global SHAP ranking.
# Render a PNG bar chart of the top features ranked by mean absolute SHAP.
def save_global_importance_plot(
    output_dir: Path,
    global_importance: pd.DataFrame,
    top_features: int,
) -> None:
    plot_data = global_importance.head(top_features).iloc[::-1]

    figure_height = max(6, 0.35 * len(plot_data) + 1.5)
    fig, ax = plt.subplots(figsize=(10, figure_height))
    ax.barh(plot_data["feature"], plot_data["mean_abs_shap"], color="#1f77b4")
    ax.set_title("Top Features by Mean Absolute SHAP")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "shap_global_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# End-to-end SHAP workflow.
# Run the full SHAP analysis workflow from model loading through saved explanation outputs.
def main() -> None:
    args = parse_args()

    # Load the cleaned data and verify it matches the trained feature set.
    dataframe = read_modeling_data(args.input)
    feature_columns = load_feature_columns(args.feature_columns_path)
    validate_feature_columns(dataframe, feature_columns)
    sampled_rows = sample_rows(dataframe, args.sample_size, args.random_state)

    # Load the trained model and any saved probability calibration artifacts.
    x_sample = sampled_rows[feature_columns].copy()
    booster = load_model(args.model_path)
    calibrator, calibration_method = load_calibration_artifacts(
        args.calibrator_path,
        args.calibration_metadata_path,
    )

    # Compute SHAP values and aggregate them into a global importance table.
    shap_values, base_value = compute_shap_values(booster, x_sample)
    global_importance = build_global_importance(shap_values)

    # Recover model probabilities so row-level outputs include prediction context.
    raw_margin_prediction = base_value + shap_values.sum(axis=1).to_numpy()
    uncalibrated_probability = margin_to_probability(raw_margin_prediction)
    calibrated_probability = apply_probability_calibrator(
        calibrator,
        calibration_method,
        uncalibrated_probability,
    )

    # Build the compact row-level contribution table for the top SHAP drivers.
    row_level_contributions = build_top_contributions(
        sampled_rows=sampled_rows,
        shap_values=shap_values,
        base_value=base_value,
        top_features=args.top_features,
        calibrated_probability=calibrated_probability,
    )

    # Record run metadata and the top global features for quick reference.
    metadata = {
        "input_path": str(args.input),
        "model_path": str(args.model_path),
        "feature_columns_path": str(args.feature_columns_path),
        "sample_row_count": int(len(sampled_rows)),
        "full_row_count": int(len(dataframe)),
        "feature_count": int(len(feature_columns)),
        "top_features_saved_per_row": int(args.top_features),
        "calibration_method": calibration_method,
        "global_top_features": global_importance.head(args.top_features).to_dict(orient="records"),
    }

    # Save tabular outputs and render the PNG importance plot.
    save_outputs(
        output_dir=args.output_dir,
        metadata=metadata,
        global_importance=global_importance,
        row_level_contributions=row_level_contributions,
        shap_values=shap_values,
    )
    save_global_importance_plot(args.output_dir, global_importance, args.top_features)

    print(f"Saved SHAP outputs to: {args.output_dir}")
    print(f"Rows explained: {len(sampled_rows):,}")
    print(f"Features explained: {len(feature_columns)}")
    print(f"Calibration method available: {calibration_method or 'none'}")
    print("Top 5 features by mean absolute SHAP:")
    for _, row in global_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['mean_abs_shap']:.6f}")


if __name__ == "__main__":
    main()