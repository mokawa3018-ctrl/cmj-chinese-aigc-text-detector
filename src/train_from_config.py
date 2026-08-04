"""Launch the project-used training entry from a JSON experiment config."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REQUIRED_FIELDS = {
    "base_model",
    "train_file",
    "validation_file",
    "output_dir",
    "text_column",
    "label_column",
    "num_labels",
    "epochs",
    "batch_size",
    "validation_batch_size",
    "max_length",
    "learning_rate",
    "weight_decay",
    "seed",
    "device",
    "mode",
    "clean",
    "quick_validation",
    "augmentation_min_length",
    "pu_loss_weight",
    "pu_type",
    "prior",
    "length_threshold",
}

OPTIONAL_FIELDS = {"note", "data_name"}
KNOWN_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def validate_contract(config: dict) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(config))
    if missing:
        errors.append("missing required fields: " + ", ".join(missing))
        return errors

    if config.get("text_column") != "answer":
        errors.append("text_column must be 'answer' because src/training/dataset.py reads answer")
    if config.get("label_column") != "label":
        errors.append("label_column must be 'label' because src/training/dataset.py reads label")
    if config.get("num_labels") != 2:
        errors.append("num_labels must be 2 for the human/AI classifier")

    for field in ("base_model", "train_file", "validation_file", "output_dir", "device", "mode", "pu_type"):
        if not isinstance(config.get(field), str) or not config[field]:
            errors.append(f"{field} must be a non-empty string")
    if "data_name" in config and (not isinstance(config["data_name"], str) or not config["data_name"]):
        errors.append("data_name must be a non-empty string when provided")

    for field in ("epochs", "batch_size", "validation_batch_size", "max_length"):
        if not is_int(config.get(field)) or config[field] <= 0:
            errors.append(f"{field} must be a positive integer")

    if not is_number(config.get("learning_rate")) or config["learning_rate"] <= 0:
        errors.append("learning_rate must be a positive number")
    if not is_number(config.get("weight_decay")) or config["weight_decay"] < 0:
        errors.append("weight_decay must be a non-negative number")
    if not is_int(config.get("seed")):
        errors.append("seed must be an integer")

    for field in ("clean", "quick_validation", "augmentation_min_length", "length_threshold"):
        if not is_int(config.get(field)):
            errors.append(f"{field} must be an integer")

    for field in ("pu_loss_weight", "prior"):
        if not is_number(config.get(field)):
            errors.append(f"{field} must be a number")

    return errors


def is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_config(config_path: Path) -> tuple[dict | None, list[str]]:
    try:
        with config_path.open("r", encoding="utf-8-sig") as f:
            config = json.load(f)
    except FileNotFoundError:
        return None, [f"config file not found: {config_path}"]
    except PermissionError:
        return None, [f"config file is not readable: {config_path}"]
    except UnicodeDecodeError as exc:
        return None, [f"config file must be UTF-8 text: {config_path} ({exc.reason})"]
    except json.JSONDecodeError as exc:
        return None, [f"invalid JSON in config file: {config_path} (line {exc.lineno}, column {exc.colno}: {exc.msg})"]
    except OSError as exc:
        return None, [f"could not read config file: {config_path} ({exc})"]

    if not isinstance(config, dict):
        return None, ["config top level must be a JSON object"]
    return config, []


def build_command(root: Path, config: dict) -> tuple[list[str], list[tuple[str, Path, bool]]]:
    train_py = root / "src" / "training" / "train.py"
    base_model = resolve_path(root, config["base_model"])
    train_file = resolve_path(root, config["train_file"])
    validation_file = resolve_path(root, config["validation_file"])
    output_dir = resolve_path(root, config["output_dir"])

    model_name = base_model.name
    local_model = str(base_model.parent)

    command = [
        sys.executable,
        str(train_py),
        "--device",
        str(config["device"]),
        "--max-epochs",
        str(config["epochs"]),
        "--batch-size",
        str(config["batch_size"]),
        "--val-batch-size",
        str(config["validation_batch_size"]),
        "--max-sequence-length",
        str(config["max_length"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--weight-decay",
        str(config["weight_decay"]),
        "--seed",
        str(config["seed"]),
        "--local-model",
        local_model,
        "--model-name",
        model_name,
        "--local-data",
        ".",
        "--train-data-file",
        str(train_file),
        "--val-data-file",
        str(validation_file),
        "--val_file1",
        str(validation_file),
        "--data-name",
        str(config.get("data_name", "save")),
        "--mode",
        str(config["mode"]),
        "--aug_min_length",
        str(config["augmentation_min_length"]),
        "--lamb",
        str(config["pu_loss_weight"]),
        "--pu_type",
        str(config["pu_type"]),
        "--prior",
        str(config["prior"]),
        "--len_thres",
        str(config["length_threshold"]),
        "--clean",
        str(config["clean"]),
        "--quick_val",
        str(config["quick_validation"]),
        "--log-dir",
        str(output_dir),
    ]

    checked_paths = [
        ("base_model", base_model, base_model.is_dir()),
        ("train_file", train_file, train_file.is_file()),
        ("validation_file", validation_file, validation_file.is_file()),
    ]
    return command, checked_paths


def print_summary(config_path: Path, command: list[str], checked_paths: list[tuple[str, Path, bool]]) -> None:
    print("config:", config_path)
    print("\nPath checks:")
    for name, path, exists in checked_paths:
        status = "OK" if exists else "MISSING"
        print(f"  {name}: {status} - {path}")

    print("\nTraining command:")
    for item in command:
        print(item)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run src/training/train.py from a JSON config.")
    parser.add_argument("--config", required=True, help="Path to a training JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without training.")
    args = parser.parse_args()

    root = project_root()
    config_path = resolve_path(root, args.config)
    config, load_errors = load_config(config_path)
    if load_errors:
        for error in load_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    assert config is not None

    unknown_fields = sorted(set(config) - KNOWN_FIELDS)
    for field in unknown_fields:
        print(f"WARNING: unknown config field will be ignored: {field}", file=sys.stderr)

    errors = validate_contract(config)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2

    command, checked_paths = build_command(root, config)
    print_summary(config_path, command, checked_paths)

    missing_paths = [(name, path) for name, path, exists in checked_paths if not exists]
    if args.dry_run:
        if missing_paths:
            print("\nDry run only: missing paths are reported above, but training was not started.")
        return 0

    if missing_paths:
        print("\nERROR: required training paths are missing; refusing to start training.", file=sys.stderr)
        for name, path in missing_paths:
            print(f"  {name}: {path}", file=sys.stderr)
        return 2

    completed = subprocess.run(command, cwd=str(root), check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
