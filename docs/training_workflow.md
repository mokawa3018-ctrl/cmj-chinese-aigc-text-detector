# Training Workflow

This document records the confirmed training workflow used in the original local experiment. The public repository does not include model weights, full datasets, prediction details, or the original official training code.

## Task And Labels

The task is Chinese AIGC text detection as binary text classification.

Confirmed label definition:

- `0`: human-written text
- `1`: AI-generated text

The main text field used by the local CSV datasets was `answer`. The original local datasets also contained metadata columns such as `question`, `generator`, `source`, and `pair_id`.

## Source Training Code

Confirmed fact:

The original experiment used the training entry from `YuchuanTian/AIGC_text_detector`, specifically its `train.py`, with local adaptations.

Confirmed local adaptations visible in the original project:

- Local model directory loading through command-line parameters.
- Local CSV train and validation file paths.
- CSV reading in `original_single` mode using `answer` and `label`.
- Additional validation metrics including TP, FN, TN, FP, precision, recall, and F1.
- Hugging Face `save_pretrained` output for complete model checkpoints.
- `best-model.pt` checkpoint saving with model state, optimizer state, epoch, and args.
- TensorBoard event writing under the output directory.

Confirmed fact:

This public repository currently does not copy the official training code. It only records the experiment workflow and includes independent evaluation and ONNX utility scripts.

## Base Model

Confirmed fact:

The fine-tuning experiments used the official Chinese zh-v3 detector as the base model:

```text
models/base/AIGC_detector_zhv3
```

The path above is a public-repository placeholder. The original local path was server-specific and is intentionally not included here.

## Experiment Groups

Confirmed experiment groups:

- GLM single-model fine-tuning.
- Multi-model v2 fine-tuning with human:AI ratio of 1:5.
- Multi-model v2 balance fine-tuning with human:AI ratio of 1:1.

Confirmed final selection:

The final selected model was the multi-model v2 balance model. It was selected because, on the frozen evaluation sets used in the local experiment, it reduced the human false positive rate compared with the 1:5 multi-model model while retaining better AI recall than the original zh-v3 baseline.

## Final Balanced Training Data

Confirmed facts for the final 1:1 balanced experiment:

- Training set: 1,700 rows.
- Validation set: 300 rows.
- Training labels: 850 human and 850 AI.
- Validation labels: 150 human and 150 AI.
- AI sources in training: `GPT初始`, `GPT-5`, `Qwen3`, `DeepSeek`, `GLM`.
- AI sources in training were balanced: 170 rows per AI source.
- AI sources in validation were balanced: 30 rows per AI source.
- Empty `answer` count: 0.
- `pair_id` overlap between train and validation: 0.

Confirmed workflow rule:

The frozen test sets were not used for training.

Inference from available files:

The `pair_id` split was intended to avoid question-level leakage between training and validation.

## Actual Training Flow

Confirmed data flow:

1. Load the local zh-v3 base model with Hugging Face Transformers.
2. Read CSV train and validation files.
3. Split rows by `label`: `0` as human, `1` as AI.
4. Tokenize text from the `answer` column.
5. Fine-tune a `BertForSequenceClassification` model.
6. Validate after the epoch.
7. Save both `best-model.pt` and a Hugging Face-format `complete-1` directory.
8. Export ONNX from the final balanced Hugging Face-format model.

## Confirmed Training Parameters

The final balanced experiment used these confirmed parameters:

```text
base_model: models/base/AIGC_detector_zhv3
train_file: data/private/train.csv
validation_file: data/private/validation.csv
output_dir: outputs/multimodel_v2_balance
text_column: answer
label_column: label
num_labels: 2
epochs: 1
batch_size: 16
validation_batch_size: 8
max_length: 512
learning_rate: 0.00002
weight_decay: 0.01
seed: 0
device: cuda
mode: original_single
clean: 0
quick_validation: 0
augmentation_min_length: 0
pu_loss_weight: 0.4
pu_type: dual_softmax_dyn_dtrun
prior: 0.2
length_threshold: 55
```

Confirmed optimizer:

```text
AdamW
```

Confirmed scheduler:

No explicit learning-rate scheduler was found in the training code used for the original experiment.

Confirmed loss behavior:

The training loss combined the Hugging Face classification loss with PU loss:

```text
classification loss + 0.4 * PU loss
```

The PU loss type was:

```text
dual_softmax_dyn_dtrun
```

## Checkpoint Outputs

Confirmed checkpoint outputs:

- `best-model.pt`: PyTorch checkpoint containing epoch, model state, optimizer state, and args.
- `complete-1/`: Hugging Face-format model directory.
- `sentence_lengths.pkl`: sentence length record.
- `events.out.tfevents.*`: TensorBoard event files.

Confirmed Hugging Face-format files:

- `config.json`
- `pytorch_model.bin`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `vocab.txt`

## ONNX Export

Confirmed fact:

ONNX was exported from the final multi-model v2 balance Hugging Face-format checkpoint.

Final public model name used in the local experiment:

```text
AIGC_text_detector_zhv3_cv2_onnx
```

Confirmed ONNX consistency check:

The local experiment compared PyTorch and ONNX outputs and found very small numerical differences between logits and probabilities.

## Unable To Confirm

The following items could not be fully confirmed from local files alone:

- The exact `CUDA_VISIBLE_DEVICES` value used for the final successful balanced training run.
- Whether the second balanced training run overwrote the first balanced run's checkpoint files.
- The complete original-vs-local code diff against the upstream repository, because the copied official code directory is not a Git repository.
- The full training command for the GLM single-model experiment.
- Full TensorBoard scalar history beyond the presence of event files.

## Reproducibility Note

The public repository records the real experiment parameters in `configs/train_multimodel_v2_balance.example.json`. The helper `src/train_from_config.py` reads that JSON, validates the expected data contract, and converts it into the real `src/training/train.py` command-line arguments.

The example config still uses public placeholder paths. Users must provide a compatible base model and private train/validation CSV files before running non-dry-run training.
