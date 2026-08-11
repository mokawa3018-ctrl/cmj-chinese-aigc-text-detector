"""Batch CSV inference for the exported ONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from transformers import BertTokenizer

from evaluation import (
    compute_binary_metrics,
    parse_group_columns,
    print_overall_metrics,
    save_group_metrics,
    validate_binary_column,
)


def softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(values)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict AIGC labels for a CSV file with an ONNX model.")
    parser.add_argument("--onnx-dir", required=True, help="Directory containing model.onnx and tokenizer files.")
    parser.add_argument("--input", required=True, help="Input CSV path. Must contain a text column.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--text-column", default="answer", help="Text column name.")
    parser.add_argument("--label-column", default="label", help="Optional label column name.")
    parser.add_argument("--group-columns", default="generator,source", help="Comma-separated grouped metric columns.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def validate_request(frame: pd.DataFrame, args: argparse.Namespace) -> bool:
    if frame.empty:
        raise ValueError("Input CSV contains no rows.")
    if args.text_column not in frame.columns:
        raise ValueError(f"Input CSV is missing text column: {args.text_column}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if args.max_length <= 0:
        raise ValueError("--max-length must be a positive integer.")
    has_label = args.label_column in frame.columns
    if has_label:
        validate_binary_column(frame, args.label_column, "label")
    return has_label


def predict_batches(session, tokenizer, texts: list[str], batch_size: int, max_length: int):
    """Run existing ONNX inference while preserving ``prob_ai`` semantics."""
    pred_labels: list[int] = []
    prob_human: list[float] = []
    prob_ai: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        logits = session.run(
            ["logits"],
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "token_type_ids": encoded["token_type_ids"],
            },
        )[0]
        probabilities = softmax(logits)
        predictions = np.argmax(probabilities, axis=-1)
        pred_labels.extend(predictions.tolist())
        prob_human.extend(probabilities[:, 0].tolist())
        prob_ai.extend(probabilities[:, 1].tolist())
        print(f"completed batch {min(start + batch_size, len(texts))}/{len(texts)}")
    return pred_labels, prob_human, prob_ai


def main() -> int:
    args = parse_args()
    try:
        frame = pd.read_csv(args.input)
        has_label = validate_request(frame, args)
        onnx_dir = Path(args.onnx_dir)
        tokenizer = BertTokenizer.from_pretrained(onnx_dir)
        session = ort.InferenceSession(str(onnx_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        predictions, human_probabilities, ai_probabilities = predict_batches(
            session,
            tokenizer,
            frame[args.text_column].fillna("").astype(str).tolist(),
            args.batch_size,
            args.max_length,
        )
    except (OSError, RuntimeError, ValueError, pd.errors.ParserError) as exc:
        print(f"error: {exc}")
        return 2

    frame["pred_label"] = predictions
    frame["prob_human"] = human_probabilities
    frame["prob_ai"] = ai_probabilities
    if has_label:
        frame["correct"] = frame["pred_label"] == pd.to_numeric(frame[args.label_column])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("input:", args.input)
    print("output:", output_path)
    print("rows:", len(frame))
    print("pred counts:")
    print(frame["pred_label"].value_counts(dropna=False).sort_index())

    if has_label:
        print("label counts:")
        print(frame[args.label_column].value_counts(dropna=False).sort_index())
        metrics = compute_binary_metrics(frame, args.label_column)
        print_overall_metrics(metrics)
        save_group_metrics(frame, output_path, parse_group_columns(args.group_columns), args.label_column)
    else:
        print(f"label column not found: {args.label_column}; supervised metrics were skipped.")

    preview_columns = [
        column
        for column in ["sample_id", args.label_column, "generator", "source", "pred_label", "prob_ai", "correct"]
        if column in frame.columns
    ]
    print(frame[preview_columns].head(20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
