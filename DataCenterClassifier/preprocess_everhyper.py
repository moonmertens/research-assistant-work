from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ID_COLUMNS = ["statefp10", "countyfp10", "tractce10"]
TARGET_COLUMN = "everhyper"
OUTPUT_DATA_NAME = "everhyper_cleaned.csv"
OUTPUT_REPORT_NAME = "everhyper_validation_summary.json"


# Parse command-line inputs and default file locations.
def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"

    parser = argparse.ArgumentParser(
        description="Clean and validate everhyper.csv for downstream XGBoost modeling."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=data_dir / "everhyper.csv",
        help="Path to the raw input CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=data_dir / OUTPUT_DATA_NAME,
        help="Path to the cleaned output CSV.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=data_dir / OUTPUT_REPORT_NAME,
        help="Path to the validation report JSON.",
    )
    return parser.parse_args()


# Standardize raw column names and fail if cleaning creates duplicates.
def clean_column_names(columns: pd.Index) -> list[str]:
    cleaned = columns.astype(str).str.strip().str.replace(" ", "", regex=False)
    duplicated = cleaned[cleaned.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Duplicate column names after cleaning: {duplicated}")
    return cleaned.tolist()


# Load the raw CSV while preserving geography IDs as strings.
def read_raw_data(input_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(input_path, dtype={column: "string" for column in ID_COLUMNS})
    dataframe.columns = clean_column_names(dataframe.columns)
    return dataframe


# Ensure the input data has the identifiers and target we need.
def validate_required_columns(dataframe: pd.DataFrame) -> None:
    missing_columns = [column for column in [*ID_COLUMNS, TARGET_COLUMN] if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


# Normalize geography codes and create derived county and tract IDs.
def normalize_geography_codes(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.copy()
    normalized["statefp10"] = normalized["statefp10"].astype("string").str.strip().str.zfill(2)
    normalized["countyfp10"] = normalized["countyfp10"].astype("string").str.strip().str.zfill(3)
    normalized["tractce10"] = normalized["tractce10"].astype("string").str.strip().str.zfill(6)
    normalized["county_id"] = normalized["statefp10"] + normalized["countyfp10"]
    normalized["tract_id"] = normalized["county_id"] + normalized["tractce10"]
    return normalized


# Convert the target to a strict binary numeric column.
def coerce_target(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.copy()
    normalized[TARGET_COLUMN] = pd.to_numeric(normalized[TARGET_COLUMN], errors="coerce")
    if normalized[TARGET_COLUMN].isna().any():
        raise ValueError("everhyper contains missing or non-numeric values.")

    invalid_values = sorted(
        value for value in normalized[TARGET_COLUMN].dropna().unique().tolist() if value not in (0, 1)
    )
    if invalid_values:
        raise ValueError(f"everhyper must be binary with values in {{0, 1}}. Found: {invalid_values}")

    normalized[TARGET_COLUMN] = normalized[TARGET_COLUMN].astype("int8")
    return normalized


# Convert all predictor columns to numeric and record invalid text values.
def coerce_numeric_features(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    normalized = dataframe.copy()
    feature_columns = [column for column in normalized.columns if column not in [*ID_COLUMNS, "county_id", "tract_id"]]
    coercion_failures: list[str] = []

    for column in feature_columns:
        if column == TARGET_COLUMN:
            continue
        original = normalized[column]
        coerced = pd.to_numeric(original, errors="coerce")
        invalid_mask = coerced.isna() & original.notna() & (original.astype("string").str.strip() != "")
        if invalid_mask.any():
            coercion_failures.append(column)
        normalized[column] = coerced

    return normalized, coercion_failures


# Identify repeated tract IDs so they can be surfaced as a hard validation error.
def detect_duplicate_tracts(dataframe: pd.DataFrame) -> pd.DataFrame:
    duplicate_mask = dataframe.duplicated(subset=["tract_id"], keep=False)
    return dataframe.loc[duplicate_mask, ["statefp10", "countyfp10", "tractce10", "tract_id"]].sort_values(
        by=["tract_id"]
    )


# Summarize key dataset checks for later review in a JSON report.
def build_validation_summary(
    dataframe: pd.DataFrame,
    coercion_failures: list[str],
) -> dict:
    feature_columns = [
        column
        for column in dataframe.columns
        if column not in [*ID_COLUMNS, "county_id", "tract_id", TARGET_COLUMN]
    ]
    missing_counts = dataframe[feature_columns].isna().sum().sort_values(ascending=False)
    constant_columns = [
        column for column in feature_columns if dataframe[column].nunique(dropna=False) <= 1
    ]
    duplicated_rows = int(dataframe.duplicated().sum())
    duplicated_tracts = int(dataframe["tract_id"].duplicated().sum())

    return {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "feature_count": int(len(feature_columns)),
        "positive_count": int(dataframe[TARGET_COLUMN].sum()),
        "positive_rate": float(dataframe[TARGET_COLUMN].mean()),
        "county_count": int(dataframe["county_id"].nunique()),
        "state_count": int(dataframe["statefp10"].nunique()),
        "duplicated_full_rows": duplicated_rows,
        "duplicated_tract_ids": duplicated_tracts,
        "numeric_coercion_failures": coercion_failures,
        "constant_feature_columns": constant_columns,
        "top_missing_features": [
            {
                "column": column,
                "missing_count": int(count),
                "missing_rate": float(count / len(dataframe)),
            }
            for column, count in missing_counts.head(25).items()
            if count > 0
        ],
    }


# Stop execution when duplicate tracts or invalid numeric fields are detected.
def validate_data(dataframe: pd.DataFrame, coercion_failures: list[str]) -> None:
    duplicated_tracts = detect_duplicate_tracts(dataframe)
    if not duplicated_tracts.empty:
        sample = duplicated_tracts.head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate tract IDs found. Sample: {sample}")

    if coercion_failures:
        raise ValueError(
            "Some feature columns contain non-numeric values after cleaning. "
            f"Review these columns: {coercion_failures}"
        )


# Write the cleaned dataset and validation summary to disk.
def save_outputs(dataframe: pd.DataFrame, report: dict, output_path: Path, report_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


# Run the full preprocessing pipeline from raw input to saved outputs.
def main() -> None:
    args = parse_args()

    dataframe = read_raw_data(args.input)
    validate_required_columns(dataframe)
    dataframe = normalize_geography_codes(dataframe)
    dataframe = coerce_target(dataframe)
    dataframe, coercion_failures = coerce_numeric_features(dataframe)
    validation_summary = build_validation_summary(dataframe, coercion_failures)
    validate_data(dataframe, coercion_failures)
    save_outputs(dataframe, validation_summary, args.output, args.report)

    print(f"Saved cleaned data to: {args.output}")
    print(f"Saved validation summary to: {args.report}")
    print(f"Rows: {validation_summary['row_count']:,}")
    print(f"Positive rate: {validation_summary['positive_rate']:.6f}")
    print(f"Counties: {validation_summary['county_count']:,}")
    print(f"Constant feature columns: {len(validation_summary['constant_feature_columns'])}")


if __name__ == "__main__":
    main()