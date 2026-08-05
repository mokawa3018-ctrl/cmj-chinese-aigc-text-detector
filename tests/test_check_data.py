import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "check_data.py"
FIELDS = [
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


def make_row(pair_id, sample_id, label, role, answer=None):
    text = answer if answer is not None else f"{role} answer {sample_id}"
    return {
        "pair_id": pair_id,
        "sample_id": sample_id,
        "source_id": f"SRC-{sample_id}",
        "question": f"question {pair_id}",
        "answer": text,
        "label": label,
        "generator": "synthetic_human" if label == 0 else "synthetic_ai",
        "source": "open_questions",
        "target_char_count": len(text),
        "dataset_role": role,
    }


def valid_rows(role, prefix):
    return [
        make_row(f"{prefix}-001", f"{prefix}-H-001", 0, role),
        make_row(f"{prefix}-001", f"{prefix}-A-001", 1, role),
        make_row(f"{prefix}-002", f"{prefix}-H-002", 0, role),
        make_row(f"{prefix}-002", f"{prefix}-A-002", 1, role),
    ]


def write_csv(path, rows, fields=FIELDS):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_check(train_path, validation_path):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train",
            str(train_path),
            "--validation",
            str(validation_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def write_valid_pair(tmp_path):
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    write_csv(train_path, valid_rows("train", "TRAIN"))
    write_csv(validation_path, valid_rows("validation", "VAL"))
    return train_path, validation_path


def test_valid_train_validation_data_returns_success(tmp_path):
    train_path, validation_path = write_valid_pair(tmp_path)

    result = run_check(train_path, validation_path)

    assert result.returncode == 0
    assert "Data check passed." in result.stdout
    assert "train/validation pair_id overlap: 0" in result.stdout


def test_missing_required_field_returns_failure(tmp_path):
    train_path, validation_path = write_valid_pair(tmp_path)
    fields = [field for field in FIELDS if field != "source_id"]
    write_csv(train_path, valid_rows("train", "TRAIN"), fields=fields)

    result = run_check(train_path, validation_path)

    assert result.returncode != 0
    assert "missing required columns" in result.stderr


def test_empty_answer_returns_failure(tmp_path):
    train_path, validation_path = write_valid_pair(tmp_path)
    rows = valid_rows("train", "TRAIN")
    rows[0]["answer"] = ""
    rows[0]["target_char_count"] = 0
    write_csv(train_path, rows)

    result = run_check(train_path, validation_path)

    assert result.returncode != 0
    assert "empty answer" in result.stderr


def test_invalid_label_returns_failure(tmp_path):
    train_path, validation_path = write_valid_pair(tmp_path)
    rows = valid_rows("train", "TRAIN")
    rows[0]["label"] = 2
    write_csv(train_path, rows)

    result = run_check(train_path, validation_path)

    assert result.returncode != 0
    assert "label must only contain 0 or 1" in result.stderr


def test_duplicate_sample_id_returns_failure(tmp_path):
    train_path, validation_path = write_valid_pair(tmp_path)
    rows = valid_rows("train", "TRAIN")
    rows[1]["sample_id"] = rows[0]["sample_id"]
    write_csv(train_path, rows)

    result = run_check(train_path, validation_path)

    assert result.returncode != 0
    assert "duplicate sample_id" in result.stderr


def test_duplicate_pair_id_and_label_returns_failure(tmp_path):
    train_path, validation_path = write_valid_pair(tmp_path)
    rows = valid_rows("train", "TRAIN")
    rows[1]["label"] = rows[0]["label"]
    write_csv(train_path, rows)

    result = run_check(train_path, validation_path)

    assert result.returncode != 0
    assert "duplicate pair_id + label" in result.stderr


def test_target_char_count_mismatch_returns_failure(tmp_path):
    train_path, validation_path = write_valid_pair(tmp_path)
    rows = valid_rows("train", "TRAIN")
    rows[0]["target_char_count"] = rows[0]["target_char_count"] + 1
    write_csv(train_path, rows)

    result = run_check(train_path, validation_path)

    assert result.returncode != 0
    assert "target_char_count mismatch" in result.stderr


def test_dataset_role_mismatch_returns_failure(tmp_path):
    train_path, validation_path = write_valid_pair(tmp_path)
    rows = valid_rows("train", "TRAIN")
    rows[0]["dataset_role"] = "validation"
    write_csv(train_path, rows)

    result = run_check(train_path, validation_path)

    assert result.returncode != 0
    assert "dataset_role mismatch" in result.stderr


def test_train_validation_pair_id_overlap_returns_failure(tmp_path):
    train_path, validation_path = write_valid_pair(tmp_path)
    rows = valid_rows("validation", "VAL")
    rows[0]["pair_id"] = "TRAIN-001"
    write_csv(validation_path, rows)

    result = run_check(train_path, validation_path)

    assert result.returncode != 0
    assert "overlapping pair_id" in result.stderr
