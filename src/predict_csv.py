"""Batch CSV inference for Hugging Face PyTorch sequence-classification models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer

from evaluation import (
    compute_binary_metrics,
    parse_group_columns,
    print_overall_metrics,
    save_group_metrics,
    validate_binary_column,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict AIGC labels for a CSV file with a PyTorch model.")
    parser.add_argument("--model-dir", required=True, help="Local HuggingFace model directory.")
    parser.add_argument("--input", required=True, help="Input CSV path. Must contain a text column.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--text-column", default="answer", help="Text column name.")
    parser.add_argument("--label-column", default="label", help="Optional label column name.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--group-columns",
        default="generator,source",
        help="Comma-separated columns for grouped metric CSVs. Use an empty string to disable.",
    )
    return parser.parse_args()


def validate_request(frame: pd.DataFrame, args: argparse.Namespace) -> bool:
    """Validate non-model inputs and return whether supervised metrics are available."""
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


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return "cpu"
    return requested


def predict_batches(model, tokenizer, texts: list[str], batch_size: int, max_length: int, device: str):
    """Run the existing batched inference logic and return labels plus probabilities."""
    pred_labels: list[int] = []
    prob_human: list[float] = []
    prob_ai: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

        pred_labels.extend(predictions.detach().cpu().tolist())
        prob_human.extend(probabilities[:, 0].detach().cpu().tolist())
        prob_ai.extend(probabilities[:, 1].detach().cpu().tolist())
        print(f"completed batch {min(start + batch_size, len(texts))}/{len(texts)}")
    return pred_labels, prob_human, prob_ai


def main() -> int:
    args = parse_args()
    try:
        frame = pd.read_csv(args.input)
        has_label = validate_request(frame, args)
        device = resolve_device(args.device)
        tokenizer = BertTokenizer.from_pretrained(args.model_dir)
        model = BertForSequenceClassification.from_pretrained(args.model_dir).to(device)
        model.eval()
        predictions, human_probabilities, ai_probabilities = predict_batches(
            model,
            tokenizer,
            frame[args.text_column].fillna("").astype(str).tolist(),
            args.batch_size,
            args.max_length,
            device,
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
