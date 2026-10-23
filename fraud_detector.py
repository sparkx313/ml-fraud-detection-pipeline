"""Train, evaluate, save, and apply an offline transaction fraud classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_FORMAT_VERSION = 1


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"CSV file does not exist: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"CSV file contains no data rows: {path}")
    return frame


def validate_binary_target(frame: pd.DataFrame, target: str) -> pd.Series:
    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' is missing from the input CSV.")
    labels = pd.to_numeric(frame[target], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(
            f"Target '{target}' must contain only numeric 0 (legitimate) and 1 (fraud) labels."
        )
    labels = labels.astype("int8")
    counts = labels.value_counts()
    if set(counts.index.tolist()) != {0, 1} or int(counts.min()) < 2:
        raise ValueError(
            "Training needs at least two legitimate and two fraud rows for a stratified holdout."
        )
    return labels


def clean_features(features: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    cleaned = features.copy()
    numeric_set = set(numeric_columns)
    for column in numeric_columns:
        values = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = values.replace([np.inf, -np.inf], np.nan)
    for column in cleaned.columns:
        if column not in numeric_set:
            values = cleaned[column].astype("string").astype(object)
            cleaned[column] = values.where(pd.notna(values), np.nan)
    return cleaned


def build_pipeline(
    numeric_columns: list[str], categorical_columns: list[str]
) -> Pipeline:
    transforms: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transforms.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transforms.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    preprocessor = ColumnTransformer(transforms, remainder="drop")
    return Pipeline(
        [
            ("features", preprocessor),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1000),
            ),
        ]
    )


def validate_threshold(threshold: float) -> None:
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("The fraud threshold must be greater than 0 and less than 1.")


def ensure_distinct_paths(input_path: Path, output_paths: list[Path]) -> None:
    resolved = [path.resolve() for path in output_paths]
    if input_path.resolve() in resolved:
        raise ValueError("An output path must not overwrite the input CSV.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("Predictions, metrics, and model output paths must be different.")


def train(args: argparse.Namespace) -> int:
    validate_threshold(args.threshold)
    output_paths = [args.predictions, args.metrics, args.model]
    ensure_distinct_paths(args.data, output_paths)

    frame = read_csv(args.data)
    labels = validate_binary_target(frame, args.target)
    id_column = args.id_column
    if id_column == args.target:
        raise ValueError("The ID column and target column must be different.")
    if id_column and id_column not in frame.columns:
        raise ValueError(f"ID column '{id_column}' is missing from the input CSV.")
    excluded = [args.target] + ([id_column] if id_column else [])
    features = frame.drop(columns=excluded).dropna(axis=1, how="all")
    if features.empty:
        raise ValueError("No usable feature columns remain after excluding the target and ID.")

    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    features = clean_features(features, numeric_columns)

    try:
        x_train, x_test, y_train, y_test, _row_train, row_test = train_test_split(
            features,
            labels,
            frame.index,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=labels,
        )
    except ValueError as exc:
        raise ValueError(f"Could not create the stratified holdout: {exc}") from exc

    usable_columns = [column for column in x_train.columns if x_train[column].notna().any()]
    if not usable_columns:
        raise ValueError("No feature has values in the training split.")
    x_train = x_train.loc[:, usable_columns]
    x_test = x_test.loc[:, usable_columns]
    features = features.loc[:, usable_columns]
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]
    model = build_pipeline(numeric_columns, categorical_columns)

    model.fit(x_train, y_train)
    fraud_probabilities = model.predict_proba(x_test)[:, 1]
    predicted_labels = (fraud_probabilities >= args.threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels, labels=[0, 1]).ravel()

    metrics: dict[str, Any] = {
        "model": "class-weighted logistic regression",
        "target_column": args.target,
        "rows_total": int(len(frame)),
        "rows_train": int(len(x_train)),
        "rows_test": int(len(x_test)),
        "fraud_rate": float(labels.mean()),
        "threshold": float(args.threshold),
        "seed": int(args.seed),
        "metrics": {
            "precision": float(precision_score(y_test, predicted_labels, zero_division=0)),
            "recall": float(recall_score(y_test, predicted_labels, zero_division=0)),
            "f1": float(f1_score(y_test, predicted_labels, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted_labels)),
            "average_precision": float(average_precision_score(y_test, fraud_probabilities)),
            "roc_auc": float(roc_auc_score(y_test, fraud_probabilities)),
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "feature_columns": features.columns.tolist(),
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
    }

    prediction_rows = pd.DataFrame(
        {
            "source_row": np.asarray(row_test) + 2,
            "actual_label": y_test.to_numpy(),
            "fraud_probability": fraud_probabilities,
            "predicted_label": predicted_labels,
        }
    )
    if id_column:
        prediction_rows.insert(1, id_column, frame.loc[row_test, id_column].to_numpy())

    model_bundle = {
        "format_version": MODEL_FORMAT_VERSION,
        "pipeline": model,
        "feature_columns": features.columns.tolist(),
        "numeric_columns": numeric_columns,
        "id_column": id_column,
        "target_column": args.target,
        "threshold": float(args.threshold),
    }
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_rows.sort_values("source_row").to_csv(args.predictions, index=False)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    joblib.dump(model_bundle, args.model)

    print(f"Holdout average precision: {metrics['metrics']['average_precision']:.4f}")
    print(f"Holdout precision / recall: {metrics['metrics']['precision']:.4f} / {metrics['metrics']['recall']:.4f}")
    print(f"Saved predictions to {args.predictions}")
    print(f"Saved metrics to {args.metrics}")
    print(f"Saved fitted model to {args.model}")
    return 0


def predict(args: argparse.Namespace) -> int:
    if args.threshold is not None:
        validate_threshold(args.threshold)
    ensure_distinct_paths(args.input, [args.output, args.model])
    frame = read_csv(args.input)
    bundle = joblib.load(args.model)
    if not isinstance(bundle, dict) or bundle.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported model artifact format.")

    feature_columns = bundle["feature_columns"]
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError("Prediction CSV is missing model feature(s): " + ", ".join(missing))
    features = frame.loc[:, feature_columns].copy()
    features = clean_features(features, bundle["numeric_columns"])
    probabilities = bundle["pipeline"].predict_proba(features)[:, 1]
    threshold = args.threshold if args.threshold is not None else bundle["threshold"]
    results = pd.DataFrame(
        {
            "source_row": frame.index.to_numpy() + 2,
            "fraud_probability": probabilities,
            "predicted_label": (probabilities >= threshold).astype("int8"),
        }
    )
    id_column = bundle.get("id_column")
    if id_column and id_column in frame.columns:
        results.insert(1, id_column, frame[id_column].to_numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Scored {len(results)} transaction(s) to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and apply a local, imbalanced transaction fraud classifier."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train, evaluate, and save a model.")
    train_parser.add_argument("--data", required=True, type=Path, help="Labeled transaction CSV.")
    train_parser.add_argument("--target", default="is_fraud", help="Binary 0/1 target column (default: is_fraud).")
    train_parser.add_argument("--id-column", help="Optional transaction identifier to exclude from training and include in output.")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Stratified holdout fraction (default: 0.2).")
    train_parser.add_argument("--threshold", type=float, default=0.5, help="Fraud probability decision threshold (default: 0.5).")
    train_parser.add_argument("--seed", type=int, default=42, help="Reproducible split seed (default: 42).")
    train_parser.add_argument("--predictions", type=Path, default=Path("fraud_predictions.csv"))
    train_parser.add_argument("--metrics", type=Path, default=Path("fraud_metrics.json"))
    train_parser.add_argument("--model", type=Path, default=Path("fraud_model.joblib"))
    train_parser.set_defaults(run=train)

    predict_parser = commands.add_parser("predict", help="Score an unlabeled transaction CSV with a saved model.")
    predict_parser.add_argument("--model", required=True, type=Path, help="Model artifact created by the train command.")
    predict_parser.add_argument("--input", required=True, type=Path, help="Transactions CSV with the model feature columns.")
    predict_parser.add_argument("--output", type=Path, default=Path("fraud_scores.csv"))
    predict_parser.add_argument("--threshold", type=float, help="Override the decision threshold saved with the model.")
    predict_parser.set_defaults(run=predict)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.run(args)
    except (OSError, ValueError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
