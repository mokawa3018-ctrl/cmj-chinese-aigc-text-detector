# Models

Model weights are intentionally excluded from ordinary Git commits.

The local `models/release/` directory is ignored by `.gitignore` and is used only for local release-candidate packaging. It should not be pushed to the public repository as regular Git files.

The published PyTorch and ONNX model files are available on Hugging Face:

https://huggingface.co/mokawa3018/cmj-chinese-aigc-text-detector

Download the full model:

```bash
hf download mokawa3018/cmj-chinese-aigc-text-detector \
  --local-dir models/cmj-chinese-aigc-text-detector
```

Layout:

- PyTorch / Transformers files are at the Hugging Face repository root.
- ONNX files are under `onnx/`, with the model at `onnx/model.onnx`.
- Labels are `0 = human` and `1 = AI`.

Keep large model files out of normal Git commits. Review the Hugging Face model card for license, data-source, limitation, and usage-boundary notes.
