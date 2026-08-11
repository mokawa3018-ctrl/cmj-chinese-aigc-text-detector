from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import compute_binary_metrics, grouped_metrics, parse_group_columns, validate_binary_column  # noqa: E402


def test_binary_metrics_match_project_label_definition():
    frame = pd.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "pred_label": [0, 1, 1, 0],
            "prob_ai": [0.1, 0.9, 0.8, 0.2],
        }
    )
    metrics = compute_binary_metrics(frame)
    assert metrics["total"] == 4
    assert metrics["correct"] == 2
    assert metrics["human_false_ai"] == 1
    assert metrics["ai_detected"] == 1
    assert metrics["ai_missed"] == 1
    assert metrics["accuracy"] == 0.5


def test_empty_frame_has_zero_counts_and_undefined_rates():
    frame = pd.DataFrame(columns=["label", "pred_label", "prob_ai"])
    metrics = compute_binary_metrics(frame)
    assert metrics["total"] == 0
    assert metrics["correct"] == 0
    assert metrics["accuracy"] is None
    assert metrics["ai_recall"] is None


@pytest.mark.parametrize(
    ("labels", "expected_rate"),
    [([0, 0], None), ([1, 1], None)],
)
def test_single_class_data_has_undefined_absent_class_rate(labels, expected_rate):
    frame = pd.DataFrame({"label": labels, "pred_label": labels, "prob_ai": [0.1] * len(labels)})
    metrics = compute_binary_metrics(frame)
    if labels[0] == 0:
        assert metrics["ai_recall"] is expected_rate
    else:
        assert metrics["human_false_positive_rate"] is expected_rate


def test_invalid_labels_raise_clear_error():
    frame = pd.DataFrame({"label": [0, 2], "pred_label": [0, 1]})
    with pytest.raises(ValueError, match="must contain only 0 or 1"):
        validate_binary_column(frame, "label", "label")


def test_group_metrics_and_missing_group_handling():
    frame = pd.DataFrame(
        {"label": [0, 1], "pred_label": [0, 1], "prob_ai": [0.1, 0.9], "generator": ["human", "model"]}
    )
    summary = grouped_metrics(frame, "generator")
    assert list(summary["generator"]) == ["human", "model"]
    assert parse_group_columns("generator, source,generator") == ["generator", "source"]
    with pytest.raises(ValueError, match="Missing group column"):
        grouped_metrics(frame, "source")
