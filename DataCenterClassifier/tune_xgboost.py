from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import optuna
except ImportError as exc:
    raise SystemExit(
        "optuna is required for hyperparameter tuning.\n"
        "Install it with:  pip install optuna"
    ) from exc

import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

# Allow importing shared utilities from the training script regardless of working directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from train_xgboost_baseline import (  # noqa: E402
    MONOTONIC_CONSTRAINTS,
    build_modeling_matrices,
    compute_scale_pos_weight,
    make_stratified_grouped_split,
    read_modeling_data,
    validate_modeling_columns,
)


# Command-line configuration.
def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    outputs_dir = project_dir / "outputs" / "tuning"

    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for the XGBoost data-center classifier."
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
        help="Directory where tuning results will be saved.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials per variant.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional hard time limit in seconds per variant. Stops early if reached.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Share of counties reserved for test (must match --test-size in train_xgboost_baseline.py).",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Share of training counties used as the inner validation fold.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splits, TPE sampler, and XGBoost (must match train_xgboost_baseline.py).",
    )
    return parser.parse_args()


# Optuna single-trial objective.
# Trains one XGBoost model with the suggested hyperparameters and returns validation PR AUC.
def _trial_objective(
    trial: optuna.Trial,
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    use_constraints: bool,
    random_state: int,
) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
    }

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric=["aucpr", "logloss"],
        n_estimators=2000,
        tree_method="hist",
        missing=float("nan"),
        scale_pos_weight=compute_scale_pos_weight(y_fit),
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=50,
        monotone_constraints=MONOTONIC_CONSTRAINTS if use_constraints else {},
        **params,
    )
    model.fit(x_fit, y_fit, eval_set=[(x_validation, y_validation)], verbose=False)

    probs = pd.Series(model.predict_proba(x_validation)[:, 1], index=x_validation.index)
    pr_auc = float(average_precision_score(y_validation, probs))
    trial.set_user_attr("best_iteration", int(model.best_iteration))
    return pr_auc


# Run an Optuna study for one model variant and persist results.
def tune_variant(
    label: str,
    use_constraints: bool,
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    n_trials: int,
    timeout: float | None,
    random_state: int,
    output_dir: Path,
) -> None:
    print(f"\n--- Tuning {label} model ({n_trials} trials) ---")

    # Silence per-trial INFO logs; show only warnings and errors.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(
        study_name=f"xgboost_{label}",
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(
        lambda trial: _trial_objective(
            trial, x_fit, y_fit, x_validation, y_validation, use_constraints, random_state
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    best_params = study.best_params
    print(f"Best validation PR AUC ({label}): {study.best_value:.6f}")
    print(f"Best params ({label}): {json.dumps(best_params, indent=2)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Full trial history for post-hoc analysis.
    study.trials_dataframe().to_csv(output_dir / f"trials_{label}.csv", index=False)

    # Best params + metadata consumed by train_xgboost_baseline.py via --tuned-params-dir.
    result = {
        "variant": label,
        "use_constraints": use_constraints,
        "best_validation_pr_auc": study.best_value,
        "best_iteration": study.best_trial.user_attrs["best_iteration"],
        "n_trials_completed": len(study.trials),
        "best_params": best_params,
    }
    (output_dir / f"best_params_{label}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"Saved tuning outputs to: {output_dir}")


# End-to-end tuning workflow.
# Builds the same deterministic splits as train_xgboost_baseline.py, then
# runs independent Optuna studies for the constrained and unconstrained variants.
def main() -> None:
    args = parse_args()

    dataframe = read_modeling_data(args.input)
    validate_modeling_columns(dataframe)
    x, y, groups, ids = build_modeling_matrices(dataframe)
    print(f"Loaded {len(x):,} tracts with {len(x.columns)} features.")

    # Outer train/test split — random_state must match train_xgboost_baseline.py.
    train_index, _, _ = make_stratified_grouped_split(
        groups, y, holdout_size=args.test_size, random_state=args.random_state
    )
    x_train = x.loc[train_index]
    y_train = y.loc[train_index]
    groups_train = groups.loc[train_index]

    # Inner fit/validation split — used for early stopping and objective evaluation.
    fit_index, validation_index, _ = make_stratified_grouped_split(
        groups_train, y_train, holdout_size=args.validation_size, random_state=args.random_state
    )
    x_fit = x_train.loc[fit_index]
    y_fit = y_train.loc[fit_index]
    x_validation = x_train.loc[validation_index]
    y_validation = y_train.loc[validation_index]

    print(
        f"Fit rows: {len(x_fit):,}  |  Validation rows: {len(x_validation):,}\n"
        f"Positive rate — fit: {float(y_fit.mean()):.4%}  |  validation: {float(y_validation.mean()):.4%}"
    )

    shared = dict(
        x_fit=x_fit,
        y_fit=y_fit,
        x_validation=x_validation,
        y_validation=y_validation,
        n_trials=args.n_trials,
        timeout=args.timeout,
        random_state=args.random_state,
        output_dir=args.output_dir,
    )

    tune_variant("constrained",   use_constraints=True,  **shared)
    tune_variant("unconstrained", use_constraints=False, **shared)


if __name__ == "__main__":
    main()
