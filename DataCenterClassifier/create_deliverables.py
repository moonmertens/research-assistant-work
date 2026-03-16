from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from joblib import load
from xgboost import DMatrix
from xgboost import XGBClassifier

# Allow importing shared utilities from the training script regardless of working directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from train_xgboost_baseline import (  # noqa: E402
    apply_probability_calibrator,
    read_modeling_data,
    validate_modeling_columns,
)


# Parse input/output locations for scored deliverable generation.
def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    outputs_dir = project_dir / "outputs"

    parser = argparse.ArgumentParser(
        description="Create final tract-level deliverables using the saved constrained and unconstrained models."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=data_dir / "everhyper_cleaned.csv",
        help="Path to the cleaned tract-level dataset to score.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=outputs_dir,
        help="Directory containing the constrained/ and unconstrained/ model artifact folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=outputs_dir / "deliverables",
        help="Directory where the final deliverable folders and scored datasets will be written.",
    )
    parser.add_argument(
        "--prediction-column",
        type=str,
        default="predicted_probability",
        help="Name of the appended final prediction column.",
    )
    parser.add_argument(
        "--shap-top-k",
        type=int,
        default=10,
        help="Number of top features to include in SHAP plots.",
    )
    parser.add_argument(
        "--shap-sample-rows",
        type=int,
        default=None,
        help="Optional cap on rows used for SHAP computation (predictions still use all rows).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used when sampling rows for SHAP.",
    )
    return parser.parse_args()


# Load the trained model artifacts needed to reproduce final calibrated scores.
def load_variant_artifacts(variant_dir: Path) -> tuple[XGBClassifier, object, str, list[str]]:
    model = XGBClassifier()
    model.load_model(variant_dir / "xgboost_baseline_model.json")

    calibrator = load(variant_dir / "probability_calibrator.joblib")
    calibration_metadata = json.loads((variant_dir / "calibration_metadata.json").read_text(encoding="utf-8"))
    feature_columns = json.loads((variant_dir / "feature_columns.json").read_text(encoding="utf-8"))
    return model, calibrator, calibration_metadata["calibration_method"], feature_columns


# Fail fast if the scored dataset no longer matches the trained feature schema.
def validate_feature_columns(dataframe: pd.DataFrame, feature_columns: list[str], variant: str) -> None:
    missing_columns = [column for column in feature_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            f"Input dataset is missing {len(missing_columns)} features required by the {variant} model: {missing_columns[:10]}"
        )


# Compute tree SHAP values from the fitted XGBoost model using pred_contribs.
def compute_shap_values(model: XGBClassifier, model_input: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    raw_contribs = model.get_booster().predict(DMatrix(model_input), pred_contribs=True)
    shap_values = pd.DataFrame(raw_contribs[:, :-1], columns=model_input.columns, index=model_input.index)
    expected_value = pd.Series(raw_contribs[:, -1], index=model_input.index, name="expected_value")
    return shap_values, expected_value


# Save a bar plot for top-k features ranked by mean absolute SHAP value.
def save_topk_bar_plot(mean_abs_shap: pd.Series, output_path: Path, top_k: int, variant: str) -> None:
    top_features = mean_abs_shap.head(top_k).sort_values(ascending=True)
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.barh(top_features.index, top_features.values)
    axis.set_xlabel("Mean |SHAP value|")
    axis.set_ylabel("Feature")
    axis.set_title(f"{variant.capitalize()} model: Top {top_k} features by mean |SHAP|")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# Save a distribution plot (boxplot) for signed SHAP values of top-k features.
def save_topk_distribution_plot(
    shap_values: pd.DataFrame,
    top_features: list[str],
    output_path: Path,
    variant: str,
) -> None:
    distributions = [shap_values[feature].to_numpy() for feature in top_features]
    fig, axis = plt.subplots(figsize=(11, 7))
    axis.boxplot(distributions, vert=False, tick_labels=top_features, showfliers=False)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_xlabel("SHAP value (impact on model output)")
    axis.set_title(f"{variant.capitalize()} model: SHAP value distributions (top features)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# Score one variant end to end and write a tract-level deliverable CSV plus metadata.
def score_variant(
    dataframe: pd.DataFrame,
    shap_dataframe: pd.DataFrame,
    variant: str,
    variant_dir: Path,
    output_dir: Path,
    prediction_column: str,
    input_path: Path,
    top_k: int,
) -> Path:
    model, calibrator, calibration_method, feature_columns = load_variant_artifacts(variant_dir)
    validate_feature_columns(dataframe, feature_columns, variant)

    # Preserve the training feature order exactly when scoring.
    model_input = dataframe.loc[:, feature_columns]
    uncalibrated_probability = pd.Series(
        model.predict_proba(model_input)[:, 1],
        index=dataframe.index,
        name="uncalibrated_probability",
    )
    calibrated_probability = apply_probability_calibrator(
        calibrator,
        calibration_method,
        uncalibrated_probability,
    )

    # Append the final calibrated score back onto the full tract dataset.
    deliverable = dataframe.copy()
    deliverable[prediction_column] = calibrated_probability.to_numpy()

    variant_output_dir = output_dir / variant
    variant_output_dir.mkdir(parents=True, exist_ok=True)

    output_file = variant_output_dir / f"{input_path.stem}_deliverable.csv"
    deliverable.to_csv(output_file, index=False)

    # Compute and persist SHAP outputs for the same model inside the deliverables variant folder.
    shap_model_input = shap_dataframe.loc[:, feature_columns]
    shap_values, expected_value = compute_shap_values(model, shap_model_input)
    mean_abs_shap = shap_values.abs().mean().sort_values(ascending=False)
    top_features = mean_abs_shap.head(top_k).index.tolist()

    shap_values_output = variant_output_dir / "shap_values.csv"
    shap_values.to_csv(shap_values_output, index=False)

    expected_output = variant_output_dir / "expected_value.csv"
    expected_value.to_frame().to_csv(expected_output, index=False)

    importance_output = variant_output_dir / "shap_feature_importance.csv"
    mean_abs_shap.rename("mean_abs_shap").to_frame().to_csv(importance_output, index=True)

    bar_plot_output = variant_output_dir / "shap_top10_bar.png"
    save_topk_bar_plot(mean_abs_shap, bar_plot_output, top_k=top_k, variant=variant)

    distribution_plot_output = variant_output_dir / "shap_top10_distribution.png"
    save_topk_distribution_plot(
        shap_values=shap_values,
        top_features=top_features,
        output_path=distribution_plot_output,
        variant=variant,
    )

    metadata = {
        "variant": variant,
        "input_path": str(input_path),
        "row_count": len(deliverable),
        "feature_count": len(feature_columns),
        "prediction_column": prediction_column,
        "calibration_method": calibration_method,
        "output_file": str(output_file),
        "shap": {
            "row_count": int(len(shap_dataframe)),
            "top_k": int(top_k),
            "top_features": top_features,
            "shap_values": str(shap_values_output),
            "expected_value": str(expected_output),
            "feature_importance": str(importance_output),
            "topk_bar_plot": str(bar_plot_output),
            "topk_distribution_plot": str(distribution_plot_output),
        },
    }
    (variant_output_dir / "deliverable_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_file


# Generate both constrained and unconstrained deliverables from the saved artifacts.
def main() -> None:
    args = parse_args()

    dataframe = read_modeling_data(args.input)
    validate_modeling_columns(dataframe)

    shap_dataframe = dataframe
    if args.shap_sample_rows is not None and args.shap_sample_rows < len(dataframe):
        shap_dataframe = dataframe.sample(n=args.shap_sample_rows, random_state=args.random_state).sort_index()
        print(f"Using SHAP sample rows: {len(shap_dataframe):,}")
    else:
        print(f"Using SHAP rows: {len(shap_dataframe):,}")

    output_paths = []
    for variant in ("constrained", "unconstrained"):
        variant_dir = args.models_dir / variant
        if not variant_dir.exists():
            raise FileNotFoundError(f"Model artifact directory not found: {variant_dir}")

        output_path = score_variant(
            dataframe=dataframe,
            shap_dataframe=shap_dataframe,
            variant=variant,
            variant_dir=variant_dir,
            output_dir=args.output_dir,
            prediction_column=args.prediction_column,
            input_path=args.input,
            top_k=args.shap_top_k,
        )
        output_paths.append(output_path)
        print(f"Saved {variant} deliverable to: {output_path}")

    print(f"Created {len(output_paths)} deliverable files in: {args.output_dir}")


if __name__ == "__main__":
    main()