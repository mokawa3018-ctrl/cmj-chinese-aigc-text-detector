from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import analyze_errors  # noqa: E402


def test_error_analysis_writes_expected_files_and_sorts_by_confidence(tmp_path):
    frame = pd.DataFrame(
        {
            "answer": ["人类A", "人类B", "AI A", "AI B"],
            "label": [0, 0, 1, 1],
            "pred_label": [1, 1, 0, 0],
            "prob_ai": [0.7, 0.95, 0.4, 0.05],
            "generator": ["human", "human", "model-a", "model-b"],
            "source": ["law", "law", "baike", "baike"],
        }
    )
    args = SimpleNamespace(
        label_column="label",
        prediction_column="pred_label",
        probability_column="prob_ai",
        text_column="answer",
        group_columns="generator,source,missing_group",
    )
    output_dir = tmp_path / "analysis"

    outputs = analyze_errors.write_analysis(frame, output_dir, args)

    expected = {
        "overall_metrics.csv",
        "metrics_by_generator.csv",
        "metrics_by_source.csv",
        "metrics_by_length.csv",
        "false_positives.csv",
        "false_negatives.csv",
    }
    assert {path.name for path in outputs} == expected
    assert list(pd.read_csv(output_dir / "false_positives.csv")["prob_ai"]) == [0.95, 0.7]
    assert list(pd.read_csv(output_dir / "false_negatives.csv")["prob_ai"]) == [0.05, 0.4]


def test_error_analysis_rejects_invalid_prediction_labels():
    frame = pd.DataFrame({"answer": ["x"], "label": [0], "pred_label": [2], "prob_ai": [0.2]})
    args = SimpleNamespace(label_column="label", prediction_column="pred_label", probability_column="prob_ai", text_column="answer")
    try:
        analyze_errors.validate_input(frame, args)
    except ValueError as error:
        assert "must contain only 0 or 1" in str(error)
    else:
        raise AssertionError("invalid predictions should be rejected")
