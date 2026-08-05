import argparse
import os

import numpy as np
import onnxruntime as ort
import pandas as pd
from transformers import BertTokenizer


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def add_binary_metrics(frame, label_column):
    total = len(frame)
    human_total = int((frame[label_column] == 0).sum())
    ai_total = int((frame[label_column] == 1).sum())
    pred_human = int((frame["pred_label"] == 0).sum())
    pred_ai = int((frame["pred_label"] == 1).sum())
    correct = int(frame["correct"].sum())
    human_false_ai = int(((frame[label_column] == 0) & (frame["pred_label"] == 1)).sum())
    ai_detected = int(((frame[label_column] == 1) & (frame["pred_label"] == 1)).sum())
    ai_missed = int(((frame[label_column] == 1) & (frame["pred_label"] == 0)).sum())

    return {
        "total": total,
        "human_total": human_total,
        "ai_total": ai_total,
        "pred_human": pred_human,
        "pred_ai": pred_ai,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "human_false_ai": human_false_ai,
        "human_false_positive_rate": human_false_ai / human_total if human_total else None,
        "ai_detected": ai_detected,
        "ai_recall": ai_detected / ai_total if ai_total else None,
        "ai_missed": ai_missed,
        "ai_miss_rate": ai_missed / ai_total if ai_total else None,
        "avg_prob_ai": float(frame["prob_ai"].mean()) if total else None,
    }


def group_metrics(df, group_col, label_column):
    rows = []
    for name, group in df.groupby(group_col, dropna=False):
        metrics = add_binary_metrics(group, label_column)
        metrics[group_col] = name
        rows.append(metrics)
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict AIGC labels for a CSV file with an ONNX model.")
    parser.add_argument("--onnx-dir", required=True, help="Directory containing model.onnx and tokenizer files.")
    parser.add_argument("--input", required=True, help="Input CSV path. Must contain an answer column.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--label-column", default="label", help="Label column name, if present.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input)
    if "answer" not in df.columns:
        raise ValueError("Input CSV must contain answer column.")
    has_label = args.label_column in df.columns

    tokenizer = BertTokenizer.from_pretrained(args.onnx_dir)
    session = ort.InferenceSession(
        os.path.join(args.onnx_dir, "model.onnx"),
        providers=["CPUExecutionProvider"],
    )

    pred_labels = []
    prob_human = []
    prob_ai = []

    texts = df["answer"].fillna("").astype(str).tolist()

    for start in range(0, len(texts), args.batch_size):
        batch_texts = texts[start : start + args.batch_size]
        encoded = tokenizer(
            batch_texts,
            max_length=args.max_length,
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

        probs = softmax(logits)
        pred = np.argmax(probs, axis=-1)

        pred_labels.extend(pred.tolist())
        prob_human.extend(probs[:, 0].tolist())
        prob_ai.extend(probs[:, 1].tolist())

        done = min(start + args.batch_size, len(texts))
        print(f"completed batch {done}/{len(texts)}")

    df["pred_label"] = pred_labels
    df["prob_human"] = prob_human
    df["prob_ai"] = prob_ai
    if has_label:
        df["correct"] = df["pred_label"] == df[args.label_column]

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("input:", args.input)
    print("output:", args.output)
    print("rows:", len(df))
    print("pred counts:")
    print(df["pred_label"].value_counts(dropna=False))

    if has_label:
        print("label counts:")
        print(df[args.label_column].value_counts(dropna=False))

        correct = int(df["correct"].sum())
        print("accuracy:", f"{correct / len(df):.6f}")

        metrics = add_binary_metrics(df, args.label_column)
        if metrics["human_total"]:
            print(
                "human false positive rate:",
                f"{metrics['human_false_positive_rate']:.6f}",
                f"({metrics['human_false_ai']}/{metrics['human_total']})",
            )
        if metrics["ai_total"]:
            print("ai recall:", f"{metrics['ai_recall']:.6f}", f"({metrics['ai_detected']}/{metrics['ai_total']})")

        for group_col in ["generator", "source"]:
            if group_col in df.columns:
                summary = group_metrics(df, group_col, args.label_column)
                summary_path = args.output.replace(".csv", f"_by_{group_col}.csv")
                summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
                print(f"\nmetrics by {group_col}:")
                print(summary.to_string(index=False))
                print("group metrics output:", summary_path)
    else:
        print(f"label column not found: {args.label_column}; supervised metrics were skipped.")

    preview_cols = [
        c
        for c in ["sample_id", args.label_column, "generator", "source", "pred_label", "prob_ai", "correct"]
        if c in df.columns
    ]
    print(df[preview_cols].head(20))


if __name__ == "__main__":
    main()
