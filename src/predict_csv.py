import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Predict AIGC labels for a CSV file with a PyTorch model.")
    parser.add_argument("--model-dir", required=True, help="Local HuggingFace model directory.")
    parser.add_argument("--input", required=True, help="Input CSV path. Must contain a text column.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--text-column", default="answer", help="Text column name.")
    parser.add_argument("--label-column", default="label", help="Label column name, if present.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--group-columns",
        default="generator,source",
        help="Comma-separated columns for grouped metric CSVs. Use empty string to disable.",
    )
    return parser.parse_args()


def add_binary_metrics(frame, label_column):
    total = len(frame)
    human_total = int((frame[label_column] == 0).sum())
    ai_total = int((frame[label_column] == 1).sum())
    pred_human = int((frame["pred_label"] == 0).sum())
    pred_ai = int((frame["pred_label"] == 1).sum())
    correct = int((frame["pred_label"] == frame[label_column]).sum())
    false_ai = int(((frame[label_column] == 0) & (frame["pred_label"] == 1)).sum())
    missed_ai = int(((frame[label_column] == 1) & (frame["pred_label"] == 0)).sum())
    ai_detected = ai_total - missed_ai

    return {
        "total": total,
        "human_total": human_total,
        "ai_total": ai_total,
        "pred_human": pred_human,
        "pred_ai": pred_ai,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "human_false_ai": false_ai,
        "human_false_positive_rate": false_ai / human_total if human_total else None,
        "ai_detected": ai_detected,
        "ai_recall": ai_detected / ai_total if ai_total else None,
        "ai_missed": missed_ai,
        "ai_miss_rate": missed_ai / ai_total if ai_total else None,
        "avg_prob_ai": float(frame["prob_ai"].mean()) if total else None,
    }


def save_group_metrics(df, output_path, label_column, group_columns):
    saved_paths = []
    for column in group_columns:
        if column not in df.columns:
            continue

        rows = []
        for value, group in df.groupby(column, dropna=False):
            metrics = add_binary_metrics(group, label_column)
            metrics[column] = value
            rows.append(metrics)

        summary = pd.DataFrame(rows)
        ordered_columns = [column] + [col for col in summary.columns if col != column]
        summary = summary[ordered_columns]
        sort_columns = [col for col in ["ai_miss_rate", "human_false_positive_rate", "total"] if col in summary.columns]
        if sort_columns:
            summary = summary.sort_values(sort_columns, ascending=[False] * len(sort_columns))

        summary_path = output_path.with_name(f"{output_path.stem}_by_{column}.csv")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        saved_paths.append(summary_path)

        print(f"\nmetrics by {column}:")
        print(summary.to_string(index=False))
        print(f"group metrics output: {summary_path}")

    return saved_paths


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if args.text_column not in df.columns:
        raise ValueError(f"Input CSV must contain column: {args.text_column}")

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    texts = df[args.text_column].fillna("").astype(str).tolist()
    pred_labels = []
    prob_human = []
    prob_ai = []

    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch_texts = texts[start : start + args.batch_size]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu()

            pred_labels.extend(torch.argmax(probs, dim=-1).tolist())
            prob_human.extend(probs[:, 0].tolist())
            prob_ai.extend(probs[:, 1].tolist())

    df["pred_label"] = pred_labels
    df["prob_human"] = prob_human
    df["prob_ai"] = prob_ai

    has_label = args.label_column in df.columns
    if has_label:
        df["correct"] = df["pred_label"] == df[args.label_column]

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"rows: {len(df)}")
    print("pred counts:")
    print(df["pred_label"].value_counts(dropna=False).sort_index())

    if has_label:
        print("label counts:")
        print(df[args.label_column].value_counts(dropna=False).sort_index())
        print(f"accuracy: {float(df['correct'].mean()):.6f}")
        if set(df[args.label_column].dropna().unique()).issubset({0, 1}):
            metrics = add_binary_metrics(df, args.label_column)
            if metrics["human_total"]:
                print(
                    "human false positive rate: "
                    f"{metrics['human_false_positive_rate']:.6f} "
                    f"({metrics['human_false_ai']}/{metrics['human_total']})"
                )
            if metrics["ai_total"]:
                print(f"ai recall: {metrics['ai_recall']:.6f} ({metrics['ai_detected']}/{metrics['ai_total']})")

            group_columns = [column.strip() for column in args.group_columns.split(",") if column.strip()]
            if group_columns:
                save_group_metrics(df, output_path, args.label_column, group_columns)

    preview_columns = [
        column
        for column in ["sample_id", args.label_column, "generator", "source", "pred_label", "prob_ai", "correct"]
        if column in df.columns
    ]
    print(df[preview_columns].head(20))


if __name__ == "__main__":
    main()
