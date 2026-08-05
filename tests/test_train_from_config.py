import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "train_from_config.py"


def valid_config(tmp_path):
    base_model = tmp_path / "models" / "base" / "AIGC_detector_zhv3"
    train_file = tmp_path / "data" / "train.csv"
    validation_file = tmp_path / "data" / "validation.csv"
    output_dir = tmp_path / "outputs" / "run"
    base_model.mkdir(parents=True)
    train_file.parent.mkdir(parents=True)
    train_file.write_text("answer,label\nhello,0\n", encoding="utf-8")
    validation_file.write_text("answer,label\nhello,0\n", encoding="utf-8")
    return {
        "base_model": str(base_model),
        "train_file": str(train_file),
        "validation_file": str(validation_file),
        "output_dir": str(output_dir),
        "text_column": "answer",
        "label_column": "label",
        "num_labels": 2,
        "epochs": 1,
        "batch_size": 16,
        "validation_batch_size": 8,
        "max_length": 512,
        "learning_rate": 0.00002,
        "weight_decay": 0.01,
        "seed": 0,
        "device": "cuda",
        "mode": "original_single",
        "clean": 0,
        "quick_validation": 0,
        "augmentation_min_length": 0,
        "pu_loss_weight": 0.4,
        "pu_type": "dual_softmax_dyn_dtrun",
        "prior": 0.2,
        "length_threshold": 55,
    }


def write_config(path, config):
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")


def run_launcher(config_path, *extra_args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(config_path), *extra_args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_valid_config_dry_run_succeeds_without_training(tmp_path):
    config_path = tmp_path / "config.json"
    write_config(config_path, valid_config(tmp_path))

    result = run_launcher(config_path, "--dry-run")

    assert result.returncode == 0
    assert "Training command:" in result.stdout
    assert "src" in result.stdout
    assert "training" in result.stdout
    assert "train.py" in result.stdout
    assert "Dry run only" not in result.stdout


def test_invalid_json_returns_code_2_without_traceback(tmp_path):
    config_path = tmp_path / "broken.json"
    config_path.write_text("{not valid json", encoding="utf-8")

    result = run_launcher(config_path, "--dry-run")

    assert result.returncode == 2
    assert "ERROR:" in result.stderr
    assert "invalid JSON" in result.stderr
    assert "Traceback" not in result.stderr


def test_json_top_level_array_returns_code_2(tmp_path):
    config_path = tmp_path / "array.json"
    config_path.write_text("[]", encoding="utf-8")

    result = run_launcher(config_path, "--dry-run")

    assert result.returncode == 2
    assert "top level must be a JSON object" in result.stderr
    assert "Traceback" not in result.stderr


def test_missing_field_returns_code_2(tmp_path):
    config = valid_config(tmp_path)
    config.pop("epochs")
    config_path = tmp_path / "missing.json"
    write_config(config_path, config)

    result = run_launcher(config_path, "--dry-run")

    assert result.returncode == 2
    assert "missing required fields" in result.stderr
    assert "Traceback" not in result.stderr


def test_field_type_error_returns_code_2(tmp_path):
    config = valid_config(tmp_path)
    config["batch_size"] = "sixteen"
    config_path = tmp_path / "bad-type.json"
    write_config(config_path, config)

    result = run_launcher(config_path, "--dry-run")

    assert result.returncode == 2
    assert "batch_size must be a positive integer" in result.stderr
    assert "Traceback" not in result.stderr


def test_missing_model_or_data_paths_block_non_dry_run(tmp_path):
    config = valid_config(tmp_path)
    config["base_model"] = str(tmp_path / "missing-model")
    config["train_file"] = str(tmp_path / "missing-train.csv")
    config_path = tmp_path / "missing-paths.json"
    write_config(config_path, config)

    result = run_launcher(config_path)

    assert result.returncode == 2
    assert "required training paths are missing" in result.stderr
    assert "missing-model" in result.stderr
    assert "missing-train.csv" in result.stderr
    assert "Traceback" not in result.stderr


def test_unknown_field_outputs_warning_without_error(tmp_path):
    config = valid_config(tmp_path)
    config["unexpected_field"] = "ignored"
    config_path = tmp_path / "unknown.json"
    write_config(config_path, config)

    result = run_launcher(config_path, "--dry-run")

    assert result.returncode == 0
    assert "WARNING: unknown config field will be ignored: unexpected_field" in result.stderr
    assert "Training command:" in result.stdout
