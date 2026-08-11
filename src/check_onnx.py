import argparse

import numpy as np
import onnxruntime as ort
import torch
from transformers import BertForSequenceClassification, BertTokenizer


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def compare_outputs(pt_logits, onnx_logits):
    """Return numerical and label-agreement checks for matching model outputs."""
    pt_probs = softmax(pt_logits)
    onnx_probs = softmax(onnx_logits)
    pt_labels = np.argmax(pt_probs, axis=-1)
    onnx_labels = np.argmax(onnx_probs, axis=-1)
    return {
        "max_abs_logits_diff": float(np.max(np.abs(pt_logits - onnx_logits))),
        "max_abs_probs_diff": float(np.max(np.abs(pt_probs - onnx_probs))),
        "labels_match": bool(np.array_equal(pt_labels, onnx_labels)),
        "pt_probs": pt_probs,
        "onnx_probs": onnx_probs,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX outputs for the same model.")
    parser.add_argument("--pt-model-dir", required=True, help="Local HuggingFace PyTorch model directory.")
    parser.add_argument("--onnx-dir", required=True, help="Directory containing model.onnx and tokenizer files.")
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()

    texts = [
        "This is a manually written sample sentence.",
        "Artificial intelligence systems can generate fluent text for many scenarios.",
        "Short practical answers may be classified differently from long formal answers.",
    ]

    tokenizer = BertTokenizer.from_pretrained(args.pt_model_dir)
    pt_model = BertForSequenceClassification.from_pretrained(args.pt_model_dir)
    pt_model.eval()

    encoded = tokenizer(
        texts,
        max_length=args.max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    with torch.no_grad():
        pt_logits = pt_model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            token_type_ids=encoded["token_type_ids"],
        ).logits.numpy()

    session = ort.InferenceSession(
        f"{args.onnx_dir}/model.onnx",
        providers=["CPUExecutionProvider"],
    )

    onnx_logits = session.run(
        ["logits"],
        {
            "input_ids": encoded["input_ids"].numpy(),
            "attention_mask": encoded["attention_mask"].numpy(),
            "token_type_ids": encoded["token_type_ids"].numpy(),
        },
    )[0]

    comparison = compare_outputs(pt_logits, onnx_logits)
    pt_probs = comparison["pt_probs"]
    onnx_probs = comparison["onnx_probs"]

    print("max_abs_logits_diff:", comparison["max_abs_logits_diff"])
    print("max_abs_probs_diff:", comparison["max_abs_probs_diff"])
    print("prediction_labels_match:", comparison["labels_match"])

    for i, text in enumerate(texts):
        print("\ntext:", text)
        print("pt_pred:", int(np.argmax(pt_probs[i])), "pt_prob_ai:", float(pt_probs[i][1]))
        print("onnx_pred:", int(np.argmax(onnx_probs[i])), "onnx_prob_ai:", float(onnx_probs[i][1]))


if __name__ == "__main__":
    main()
