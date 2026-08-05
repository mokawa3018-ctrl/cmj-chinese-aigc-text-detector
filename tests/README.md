# Tests

This directory contains lightweight automated tests for the public helper scripts.

The test suite covers:

- `src/check_data.py` data validation behavior.
- `src/train_from_config.py` JSON config validation and dry-run behavior.
- CLI execution through the active Python interpreter.

It deliberately does not install deep learning dependencies, download models, load weights, run training, run inference, or inspect ONNX files.

Run locally with:

```bash
python -m pip install -r requirements-ci.txt
python -m compileall src tests
ruff check src/check_data.py src/train_from_config.py tests
python -m pytest -q
```
