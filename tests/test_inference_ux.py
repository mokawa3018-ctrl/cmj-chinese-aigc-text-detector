import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FakeTensor:
    def __init__(self, data):
        self.data = data

    def to(self, _device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.data

    def __getitem__(self, item):
        if isinstance(item, tuple):
            row_selector, column_selector = item
            rows = self.data if isinstance(row_selector, slice) else [self.data[row_selector]]
            if isinstance(column_selector, int):
                return FakeTensor([row[column_selector] for row in rows])
        if isinstance(item, int):
            row = self.data[item]
            return FakeTensor(row) if isinstance(row, list) else row
        return FakeTensor(self.data[item])


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


class FakeTorch:
    cuda = SimpleNamespace(is_available=lambda: False)

    @staticmethod
    def device(name):
        return name

    @staticmethod
    def no_grad():
        return FakeNoGrad()

    @staticmethod
    def softmax(logits, dim=-1):
        _ = dim
        if logits.data and isinstance(logits.data[0], float):
            total = sum(logits.data)
            return FakeTensor([value / total for value in logits.data])

        rows = []
        for row in logits.data:
            total = sum(row)
            rows.append([value / total for value in row])
        return FakeTensor(rows)

    @staticmethod
    def argmax(tensor, dim=-1):
        _ = dim
        return FakeTensor([max(range(len(row)), key=lambda index: row[index]) for row in tensor.data])


class FakeTokenizer:
    @classmethod
    def from_pretrained(cls, _model_dir):
        return cls()

    def __call__(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {
            "input_ids": FakeTensor([[1, 2]] * len(texts)),
            "attention_mask": FakeTensor([[1, 1]] * len(texts)),
            "token_type_ids": FakeTensor([[0, 0]] * len(texts)),
        }


class FakeModel:
    @classmethod
    def from_pretrained(cls, _model_dir):
        return cls()

    def to(self, _device):
        return self

    def eval(self):
        return None

    def __call__(self, **kwargs):
        batch_size = len(kwargs["input_ids"].data)
        logits = [[2.0, 1.0], [1.0, 3.0]][:batch_size]
        return SimpleNamespace(logits=FakeTensor(logits))


class FakeSession:
    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, _outputs, inputs):
        batch_size = len(inputs["input_ids"].data)
        return [[[2.0, 1.0], [1.0, 3.0]][:batch_size]]


def load_module(module_name, relative_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForSequenceClassification=FakeModel,
            AutoTokenizer=FakeTokenizer,
            BertForSequenceClassification=FakeModel,
            BertTokenizer=FakeTokenizer,
        ),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=FakeSession))
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_predict_text_outputs_probabilities_with_mocked_model(monkeypatch, capsys):
    module = load_module("predict_text_under_test", "src/predict_text.py", monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["predict_text.py", "--model-path", "mock-model", "--text", "这是一段测试文本。", "--device", "cpu"],
    )

    assert module.main() == 0

    output = capsys.readouterr().out
    assert "predicted_label: 0 (human)" in output
    assert "human_probability:" in output
    assert "ai_probability:" in output


def test_predict_text_rejects_empty_text(monkeypatch, capsys):
    module = load_module("predict_text_empty_under_test", "src/predict_text.py", monkeypatch)
    monkeypatch.setattr(sys, "argv", ["predict_text.py", "--model-path", "mock-model", "--text", "   "])

    assert module.main() == 2
    assert "--text must not be empty" in capsys.readouterr().err


def test_predict_csv_without_label_writes_predictions_only(tmp_path, monkeypatch, capsys):
    module = load_module("predict_csv_under_test", "src/predict_csv.py", monkeypatch)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame({"answer": ["人工文本", "AI文本"]}).to_csv(input_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict_csv.py",
            "--model-dir",
            "mock-model",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--device",
            "cpu",
        ],
    )

    module.main()

    output = pd.read_csv(output_path)
    assert list(output["pred_label"]) == [0, 1]
    assert "prob_human" in output.columns
    assert "prob_ai" in output.columns
    assert "correct" not in output.columns
    assert "supervised metrics were skipped" in capsys.readouterr().out


def test_predict_csv_with_label_and_missing_groups_skips_group_files(tmp_path, monkeypatch):
    module = load_module("predict_csv_label_under_test", "src/predict_csv.py", monkeypatch)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame({"answer": ["人工文本", "AI文本"], "label": [0, 1]}).to_csv(input_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict_csv.py",
            "--model-dir",
            "mock-model",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--device",
            "cpu",
        ],
    )

    module.main()

    output = pd.read_csv(output_path)
    assert list(output["correct"]) == [True, True]
    assert not output_path.with_name("output_by_generator.csv").exists()
    assert not output_path.with_name("output_by_source.csv").exists()


def test_predict_csv_onnx_without_label_writes_predictions_only(tmp_path, monkeypatch, capsys):
    module = load_module("predict_csv_onnx_under_test", "src/predict_csv_onnx.py", monkeypatch)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame({"answer": ["人工文本", "AI文本"]}).to_csv(input_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict_csv_onnx.py",
            "--onnx-dir",
            "mock-model",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    module.main()

    output = pd.read_csv(output_path)
    assert list(output["pred_label"]) == [0, 1]
    assert "correct" not in output.columns
    assert "supervised metrics were skipped" in capsys.readouterr().out


def test_predict_csv_onnx_supports_custom_text_column(tmp_path, monkeypatch):
    module = load_module("predict_csv_onnx_custom_text", "src/predict_csv_onnx.py", monkeypatch)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame({"content": ["人工", "AI"]}).to_csv(input_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict_csv_onnx.py",
            "--onnx-dir",
            "mock-model",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--text-column",
            "content",
        ],
    )

    assert module.main() == 0
    assert list(pd.read_csv(output_path)["pred_label"]) == [0, 1]


def test_predict_csv_rejects_missing_text_column(tmp_path, monkeypatch, capsys):
    module = load_module("predict_csv_missing_text", "src/predict_csv.py", monkeypatch)
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"content": ["文本"]}).to_csv(input_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["predict_csv.py", "--model-dir", "mock-model", "--input", str(input_path), "--output", str(tmp_path / "out.csv"), "--device", "cpu"],
    )

    assert module.main() == 2
    assert "missing text column" in capsys.readouterr().out


def test_predict_csv_preserves_cuda_to_cpu_fallback(monkeypatch, capsys):
    module = load_module("predict_csv_device_fallback", "src/predict_csv.py", monkeypatch)

    assert module.resolve_device("cuda") == "cpu"
    assert "falling back to CPU" in capsys.readouterr().out


def test_predict_csv_onnx_rejects_invalid_batch_or_max_length(tmp_path, monkeypatch, capsys):
    module = load_module("predict_csv_onnx_invalid_values", "src/predict_csv_onnx.py", monkeypatch)
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"answer": ["文本"]}).to_csv(input_path, index=False)
    for option, value in [("--batch-size", "0"), ("--max-length", "0")]:
        monkeypatch.setattr(
            sys,
            "argv",
            ["predict_csv_onnx.py", "--onnx-dir", "mock-model", "--input", str(input_path), "--output", str(tmp_path / "out.csv"), option, value],
        )
        assert module.main() == 2
        assert "positive integer" in capsys.readouterr().out
