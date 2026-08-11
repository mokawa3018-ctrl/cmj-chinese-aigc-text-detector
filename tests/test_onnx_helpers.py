import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path, monkeypatch, modules):
    for module_name, module in modules.items():
        monkeypatch.setitem(sys.modules, module_name, module)
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_onnx_uses_public_io_names_dynamic_axes_and_opset(monkeypatch):
    calls = {}

    def export(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs

    fake_torch = SimpleNamespace(onnx=SimpleNamespace(export=export))
    module = load_module(
        "export_onnx_under_test",
        "src/export_onnx.py",
        monkeypatch,
        {"torch": fake_torch, "transformers": SimpleNamespace(BertTokenizer=object, BertForSequenceClassification=object)},
    )
    encoded = {"input_ids": "ids", "attention_mask": "mask", "token_type_ids": "types"}
    module.export_model("model", encoded, "model.onnx")

    assert calls["kwargs"]["input_names"] == ["input_ids", "attention_mask", "token_type_ids"]
    assert calls["kwargs"]["output_names"] == ["logits"]
    assert calls["kwargs"]["opset_version"] == 14
    assert calls["kwargs"]["dynamic_axes"]["input_ids"][1] == "sequence_length"


def test_check_onnx_reports_numerical_difference_and_label_match(monkeypatch):
    module = load_module(
        "check_onnx_under_test",
        "src/check_onnx.py",
        monkeypatch,
        {
            "onnxruntime": SimpleNamespace(),
            "torch": SimpleNamespace(),
            "transformers": SimpleNamespace(BertTokenizer=object, BertForSequenceClassification=object),
        },
    )
    comparison = module.compare_outputs(
        np.array([[2.0, 1.0], [1.0, 3.0]]),
        np.array([[2.0, 1.000001], [1.0, 3.0]]),
    )
    assert comparison["labels_match"] is True
    assert comparison["max_abs_logits_diff"] < 1.1e-6
