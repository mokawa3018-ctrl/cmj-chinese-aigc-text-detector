"""Predict a single Chinese text with a Hugging Face sequence classifier."""
from __future__ import annotations

import argparse
import sys


LABELS = {0: "human", 1: "AI"}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict whether one text is human-written or AI-generated.")
    parser.add_argument("--model-path", required=True, help="Local path or Hugging Face model id.")
    parser.add_argument("--text", required=True, help="Text to classify.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum tokenized length.")
    return parser.parse_args()


def choose_device(torch, requested_device):
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available. Use --device cpu or --device auto.")
    return requested_device


def load_model(model_path):
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Missing inference dependencies. Install torch and transformers before running predict_text.py."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {model_path!r}: {exc}") from exc

    return torch, tokenizer, model


def predict(torch, tokenizer, model, text, device, max_length):
    model.to(device)
    model.eval()

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()

    pred_label = int(max(range(len(probs)), key=lambda index: probs[index]))
    return pred_label, probs


def main():
    args = parse_args()
    text = args.text.strip()
    if not text:
        print("ERROR: --text must not be empty.", file=sys.stderr)
        return 2
    if args.max_length <= 0:
        print("ERROR: --max-length must be a positive integer.", file=sys.stderr)
        return 2

    try:
        torch, tokenizer, model = load_model(args.model_path)
        device = choose_device(torch, args.device)
        pred_label, probs = predict(torch, tokenizer, model, text, device, args.max_length)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"predicted_label: {pred_label} ({LABELS.get(pred_label, 'unknown')})")
    print(f"human_probability: {probs[0]:.6f}")
    print(f"ai_probability: {probs[1]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
