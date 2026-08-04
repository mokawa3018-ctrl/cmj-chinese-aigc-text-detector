import argparse
import sys

import pandas as pd


REQUIRED_COLUMNS = [
    "pair_id",
    "sample_id",
    "source_id",
    "question",
    "answer",
    "label",
    "generator",
    "source",
    "target_char_count",
    "dataset_role",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Validate sample train and validation CSV files.")
    parser.add_argument("--train", required=True, help="Path to the training CSV file.")
    parser.add_argument("--validation", required=True, help="Path to the validation CSV file.")
    return parser.parse_args()


def load_csv(path, role):
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"{role}: failed to read CSV: {exc}") from exc


def check_frame(df, role):
    errors = []

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        errors.append(f"{role}: missing required columns: {', '.join(missing)}")
        return errors

    empty_answer = df["answer"].fillna("").astype(str).str.strip().eq("")
    if empty_answer.any():
        errors.append(f"{role}: empty answer rows: {int(empty_answer.sum())}")

    labels = set(df["label"].dropna().tolist())
    if not labels.issubset({0, 1}):
        errors.append(f"{role}: label must only contain 0 or 1, got: {sorted(labels)}")

    duplicate_sample_id = df["sample_id"].duplicated()
    if duplicate_sample_id.any():
        errors.append(f"{role}: duplicate sample_id rows: {int(duplicate_sample_id.sum())}")

    duplicate_pair_label = df.duplicated(subset=["pair_id", "label"])
    if duplicate_pair_label.any():
        errors.append(f"{role}: duplicate pair_id + label rows: {int(duplicate_pair_label.sum())}")

    actual_lengths = df["answer"].fillna("").astype(str).str.len()
    expected_lengths = pd.to_numeric(df["target_char_count"], errors="coerce")
    bad_lengths = expected_lengths.isna() | (actual_lengths != expected_lengths)
    if bad_lengths.any():
        bad_rows = df.loc[bad_lengths, ["sample_id", "target_char_count"]].head(10).to_dict("records")
        errors.append(f"{role}: target_char_count mismatch rows: {int(bad_lengths.sum())}; examples: {bad_rows}")

    bad_role = df["dataset_role"].fillna("").astype(str) != role
    if bad_role.any():
        errors.append(f"{role}: dataset_role mismatch rows: {int(bad_role.sum())}")

    return errors


def print_summary(df, role):
    print(f"\n== {role} ==")
    print(f"rows: {len(df)}")
    print("label counts:")
    print(df["label"].value_counts(dropna=False).sort_index().to_string())
    print("generator counts:")
    print(df["generator"].value_counts(dropna=False).to_string())
    print("source counts:")
    print(df["source"].value_counts(dropna=False).to_string())
    print(f"pair_ids: {df['pair_id'].nunique()}")


def main():
    args = parse_args()
    train_df = load_csv(args.train, "train")
    validation_df = load_csv(args.validation, "validation")

    errors = []
    errors.extend(check_frame(train_df, "train"))
    errors.extend(check_frame(validation_df, "validation"))

    if "pair_id" in train_df.columns and "pair_id" in validation_df.columns:
        overlap = sorted(set(train_df["pair_id"]) & set(validation_df["pair_id"]))
        if overlap:
            errors.append(f"train/validation: overlapping pair_id values: {overlap[:10]}")

    if errors:
        print("Data check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Data check passed.")
    print_summary(train_df, "train")
    print_summary(validation_df, "validation")
    print("\ntrain/validation pair_id overlap: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
