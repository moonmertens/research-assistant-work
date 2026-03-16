from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from joblib import load
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


# Score one variant end to end and write a tract-level deliverable CSV plus metadata.
def score_variant(
    dataframe: pd.DataFrame,
    variant: str,
    variant_dir: Path,
    output_dir: Path,
    prediction_column: str,
    input_path: Path,
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

    metadata = {
        "variant": variant,
        "input_path": str(input_path),
        "row_count": len(deliverable),
        "feature_count": len(feature_columns),
        "prediction_column": prediction_column,
        "calibration_method": calibration_method,
        "output_file": str(output_file),
    }
    (variant_output_dir / "deliverable_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_file


# Generate both constrained and unconstrained deliverables from the saved artifacts.
def main() -> None:
    args = parse_args()

    dataframe = read_modeling_data(args.input)
    validate_modeling_columns(dataframe)

    output_paths = []
    for variant in ("constrained", "unconstrained"):
        variant_dir = args.models_dir / variant
        if not variant_dir.exists():
            raise FileNotFoundError(f"Model artifact directory not found: {variant_dir}")

        output_path = score_variant(
            dataframe=dataframe,
            variant=variant,
            variant_dir=variant_dir,
            output_dir=args.output_dir,
            prediction_column=args.prediction_column,
            input_path=args.input,
        )
        output_paths.append(output_path)
        print(f"Saved {variant} deliverable to: {output_path}")

    print(f"Created {len(output_paths)} deliverable files in: {args.output_dir}")


if __name__ == "__main__":
    main()