"""Create reusable error-analysis tables from a prediction CSV without loading a model."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluation import (
    compute_binary_metrics,
    grouped_metrics,
    parse_group_columns,
    save_group_metrics,
    validate_binary_column,
)

LENGTH_BINS = [0, 50, 100, 200, 500, float("inf")]
LENGTH_LABELS = ["0-50", "51-100", "101-200", "201-500", "500+"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze errors in an existing binary prediction CSV.")
    parser.add_argument("--input", required=True, help="Prediction CSV created by a batch predictor.")
    parser.add_argument("--output-dir", required=True, help="Directory for analysis CSV files.")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--prediction-column", default="pred_label")
    parser.add_argument("--probability-column", default="prob_ai")
    parser.add_argument("--text-column", default="answer")
    parser.add_argument("--group-columns", default="generator,source")
    return parser.parse_args()


def validate_input(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    required = [args.label_column, args.prediction_column, args.probability_column, args.text_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Input CSV contains no rows.")
    validate_binary_column(frame, args.label_column, "label")
    validate_binary_column(frame, args.prediction_column, "prediction")
    probabilities = pd.to_numeric(frame[args.probability_column], errors="coerce")
    if probabilities.isna().any():
        raise ValueError(f"Probability column {args.probability_column!r} contains non-numeric values.")


def add_length_bucket(frame: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Return a copy with fixed inclusive character-length buckets."""
    result = frame.copy()
    lengths = result[text_column].fillna("").astype(str).str.len()
    result["length_bucket"] = pd.cut(
        lengths,
        bins=LENGTH_BINS,
        labels=LENGTH_LABELS,
        include_lowest=True,
    )
    return result


def write_analysis(frame: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> list[Path]:
    """Write overall, grouped, length, and confidence-sorted error CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    overall_path = output_dir / "overall_metrics.csv"
    pd.DataFrame([compute_binary_metrics(frame, args.label_column, args.prediction_column, args.probability_column)]).to_csv(
        overall_path, index=False, encoding="utf-8-sig"
    )
    outputs.append(overall_path)

    grouped_base = output_dir / "metrics.csv"
    outputs.extend(
        save_group_metrics(
            frame,
            grouped_base,
            parse_group_columns(args.group_columns),
            args.label_column,
            args.prediction_column,
            args.probability_column,
        )
    )

    with_length = add_length_bucket(frame, args.text_column)
    length_path = output_dir / "metrics_by_length.csv"
    grouped_metrics(
        with_length,
        "length_bucket",
        args.label_column,
        args.prediction_column,
        args.probability_column,
    ).to_csv(length_path, index=False, encoding="utf-8-sig")
    outputs.append(length_path)

    labels = pd.to_numeric(frame[args.label_column])
    predictions = pd.to_numeric(frame[args.prediction_column])
    false_positives = frame.loc[(labels == 0) & (predictions == 1)].sort_values(
        args.probability_column, ascending=False
    )
    false_positive_path = output_dir / "false_positives.csv"
    false_positives.to_csv(false_positive_path, index=False, encoding="utf-8-sig")
    outputs.append(false_positive_path)

    false_negatives = frame.loc[(labels == 1) & (predictions == 0)].sort_values(
        args.probability_column, ascending=True
    )
    false_negative_path = output_dir / "false_negatives.csv"
    false_negatives.to_csv(false_negative_path, index=False, encoding="utf-8-sig")
    outputs.append(false_negative_path)
    return outputs


def main() -> int:
    args = parse_args()
    try:
        frame = pd.read_csv(args.input)
        validate_input(frame, args)
        outputs = write_analysis(frame, Path(args.output_dir), args)
    except (OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
        print(f"error: {exc}")
        return 2

    print("analysis outputs:")
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
