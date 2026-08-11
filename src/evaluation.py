"""Shared metrics and grouped summaries for binary AIGC classification."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


METRIC_COLUMNS = [
    "total",
    "human_total",
    "ai_total",
    "pred_human",
    "pred_ai",
    "correct",
    "accuracy",
    "human_false_ai",
    "human_false_positive_rate",
    "ai_detected",
    "ai_recall",
    "ai_missed",
    "ai_miss_rate",
    "avg_prob_ai",
]


def parse_group_columns(value: str) -> list[str]:
    """Parse a comma-separated CLI value while retaining caller order."""
    columns: list[str] = []
    for column in value.split(","):
        column = column.strip()
        if column and column not in columns:
            columns.append(column)
    return columns


def validate_binary_column(frame: pd.DataFrame, column: str, kind: str) -> None:
    """Raise a concise error when a label or prediction column is not binary."""
    if column not in frame.columns:
        raise ValueError(f"Missing {kind} column: {column}")

    values = pd.to_numeric(frame[column], errors="coerce")
    invalid = values.isna() | ~values.isin([0, 1])
    if invalid.any():
        examples = frame.loc[invalid, column].head(5).tolist()
        raise ValueError(f"{kind.capitalize()} column {column!r} must contain only 0 or 1; invalid values: {examples}")


def compute_binary_metrics(
    frame: pd.DataFrame,
    label_column: str = "label",
    prediction_column: str = "pred_label",
    probability_column: str = "prob_ai",
) -> dict[str, int | float | None]:
    """Compute project-standard metrics for ``0=human`` and ``1=AI`` labels."""
    validate_binary_column(frame, label_column, "label")
    validate_binary_column(frame, prediction_column, "prediction")

    total = len(frame)
    if total == 0:
        return dict.fromkeys(METRIC_COLUMNS, None) | {
            "total": 0,
            "human_total": 0,
            "ai_total": 0,
            "pred_human": 0,
            "pred_ai": 0,
            "correct": 0,
            "human_false_ai": 0,
            "ai_detected": 0,
            "ai_missed": 0,
        }

    labels = pd.to_numeric(frame[label_column])
    predictions = pd.to_numeric(frame[prediction_column])
    human_total = int((labels == 0).sum())
    ai_total = int((labels == 1).sum())
    human_false_ai = int(((labels == 0) & (predictions == 1)).sum())
    ai_detected = int(((labels == 1) & (predictions == 1)).sum())
    ai_missed = int(((labels == 1) & (predictions == 0)).sum())
    correct = int((labels == predictions).sum())

    avg_prob_ai: float | None = None
    if probability_column in frame.columns:
        probabilities = pd.to_numeric(frame[probability_column], errors="coerce")
        if probabilities.notna().any():
            avg_prob_ai = float(probabilities.mean())

    return {
        "total": total,
        "human_total": human_total,
        "ai_total": ai_total,
        "pred_human": int((predictions == 0).sum()),
        "pred_ai": int((predictions == 1).sum()),
        "correct": correct,
        "accuracy": correct / total,
        "human_false_ai": human_false_ai,
        "human_false_positive_rate": human_false_ai / human_total if human_total else None,
        "ai_detected": ai_detected,
        "ai_recall": ai_detected / ai_total if ai_total else None,
        "ai_missed": ai_missed,
        "ai_miss_rate": ai_missed / ai_total if ai_total else None,
        "avg_prob_ai": avg_prob_ai,
    }


def grouped_metrics(
    frame: pd.DataFrame,
    group_column: str,
    label_column: str = "label",
    prediction_column: str = "pred_label",
    probability_column: str = "prob_ai",
) -> pd.DataFrame:
    """Return standard metrics per group, including missing values as a group."""
    if group_column not in frame.columns:
        raise ValueError(f"Missing group column: {group_column}")

    rows = []
    for group_name, group in frame.groupby(group_column, dropna=False, observed=False):
        rows.append(
            {group_column: group_name}
            | compute_binary_metrics(group, label_column, prediction_column, probability_column)
        )
    return pd.DataFrame(rows, columns=[group_column, *METRIC_COLUMNS])


def available_group_columns(frame: pd.DataFrame, requested: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return available group columns and requested columns that are absent."""
    available = [column for column in requested if column in frame.columns]
    missing = [column for column in requested if column not in frame.columns]
    return available, missing


def group_output_path(output_path: Path, group_column: str) -> Path:
    """Construct the legacy-compatible grouped output name using pathlib."""
    return output_path.with_name(f"{output_path.stem}_by_{group_column}.csv")


def save_group_metrics(
    frame: pd.DataFrame,
    output_path: Path,
    requested_groups: Iterable[str],
    label_column: str = "label",
    prediction_column: str = "pred_label",
    probability_column: str = "prob_ai",
) -> list[Path]:
    """Save summaries for available grouping columns and return their paths."""
    saved: list[Path] = []
    available, missing = available_group_columns(frame, requested_groups)
    for column in missing:
        print(f"group column not found: {column}; grouped metrics were skipped.")
    for column in available:
        summary = grouped_metrics(frame, column, label_column, prediction_column, probability_column)
        path = group_output_path(output_path, column)
        summary.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\nmetrics by {column}:")
        print(summary.to_string(index=False))
        print("group metrics output:", path)
        saved.append(path)
    return saved


def print_overall_metrics(metrics: dict[str, int | float | None]) -> None:
    """Print the concise legacy console summary used by both predictors."""
    if metrics["accuracy"] is not None:
        print("accuracy:", f"{metrics['accuracy']:.6f}")
    if metrics["human_total"]:
        print(
            "human false positive rate:",
            f"{metrics['human_false_positive_rate']:.6f}",
            f"({metrics['human_false_ai']}/{metrics['human_total']})",
        )
    if metrics["ai_total"]:
        print("ai recall:", f"{metrics['ai_recall']:.6f}", f"({metrics['ai_detected']}/{metrics['ai_total']})")
