import argparse
import os
import shutil

import torch
from transformers import BertForSequenceClassification, BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Export a local BertForSequenceClassification model to ONNX.")
    parser.add_argument("--model-dir", required=True, help="Local HuggingFace PyTorch model directory.")
    parser.add_argument("--output-dir", required=True, help="Directory where model.onnx and tokenizer files will be written.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum token length used for the export sample.")
    return parser.parse_args()


def export_model(model, encoded, onnx_path):
    """Export with the exact public input/output names and dynamic axes."""
    torch.onnx.export(
        model,
        (
            encoded["input_ids"],
            encoded["attention_mask"],
            encoded["token_type_ids"],
        ),
        onnx_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    sample_text = "This is a short sample sentence for ONNX export."
    encoded = tokenizer(
        sample_text,
        max_length=args.max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    onnx_path = os.path.join(args.output_dir, "model.onnx")

    with torch.no_grad():
        export_model(model, encoded, onnx_path)

    for name in [
        "config.json",
        "vocab.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]:
        src = os.path.join(args.model_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, name))

    print("saved:", onnx_path)
    print("output_dir:", args.output_dir)


if __name__ == "__main__":
    main()
