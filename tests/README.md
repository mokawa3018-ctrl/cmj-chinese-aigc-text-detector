# Tests

This directory contains lightweight automated tests for the public helper scripts.

The test suite covers:

- `src/check_data.py` data validation behavior.
- `src/train_from_config.py` JSON config validation and dry-run behavior.
- Shared binary metrics, grouped summaries, edge cases, and invalid labels.
- CSV inference input validation, unlabeled CSV output, custom text columns, and absent groups.
- Error-analysis CSV generation and confidence ordering for false positives and false negatives.
- ONNX export parameter construction and PyTorch/ONNX comparison logic through mocks.
- CLI execution through the active Python interpreter where no model dependency is needed.

It deliberately does not install deep learning dependencies, download models, load real weights, run training, execute real model inference, export a real ONNX file, or inspect a production ONNX file. Model-facing tests use small fake objects only.

Run locally with:

```bash
python -m pip install -r requirements-ci.txt
python -m compileall src tests
ruff check src/check_data.py src/train_from_config.py src/evaluation.py src/analyze_errors.py src/predict_csv.py src/predict_csv_onnx.py src/predict_text.py src/export_onnx.py src/check_onnx.py tests
python -m pytest -q
```
