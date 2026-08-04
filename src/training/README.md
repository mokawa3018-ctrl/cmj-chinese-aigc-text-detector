# Training Module

The files in this directory are the project-used training chain based on `YuchuanTian/AIGC_text_detector`.

The upstream training code is licensed under Apache License 2.0. This repository keeps the Apache-2.0 license text at the repository root and documents upstream attribution in `THIRD_PARTY_NOTICES.md`.

## Scope

This module is not a from-scratch training framework. It preserves the training path used in the local experiments, including CSV loading, classification loss, PU loss, validation metrics, TensorBoard logging, `best-model.pt` saving, and Hugging Face `save_pretrained` checkpoints.

## Project-Used Changes

The project-used version supports:

- local base model directories through `--local-model` and `--model-name`
- local CSV train and validation files through `--local-data`, `--train-data-file`, `--val-data-file`, and `--val_file1`
- CSV fields `answer`, `label`, and `question`
- validation confusion-matrix logging with TP, FN, TN, and FP
- F1, precision, recall, and accuracy reporting
- PU loss configuration through `--lamb`, `--pu_type`, `--prior`, and `--len_thres`
- Hugging Face-format checkpoint export in `complete-{epoch}/`

## Direct Command-Line Training

```bash
python src/training/train.py \
  --device cuda \
  --max-epochs 1 \
  --batch-size 16 \
  --val-batch-size 8 \
  --max-sequence-length 512 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --seed 0 \
  --local-model models/base \
  --model-name AIGC_detector_zhv3 \
  --local-data . \
  --train-data-file data/private/train.csv \
  --val-data-file data/private/validation.csv \
  --val_file1 data/private/validation.csv \
  --data-name save \
  --mode original_single \
  --aug_min_length 0 \
  --lamb 0.4 \
  --pu_type dual_softmax_dyn_dtrun \
  --prior 0.2 \
  --len_thres 55 \
  --clean 0 \
  --quick_val 0 \
  --log-dir outputs/multimodel_v2_balance
```

The paths above are placeholders. A compatible base model and private training data must be prepared by the user.

## JSON Config Launcher

The repository also provides a small launcher that reads a JSON config and converts it to the real command-line arguments used by `train.py`:

```bash
python src/train_from_config.py --config configs/train_multimodel_v2_balance.example.json --dry-run
```

Remove `--dry-run` only after the base model, training file, and validation file paths exist.

## JSON Field Mapping

| JSON field | Training argument |
| --- | --- |
| `base_model` | split into `--local-model` parent directory and `--model-name` final directory name |
| `train_file` | `--train-data-file` |
| `validation_file` | `--val-data-file` and `--val_file1` |
| `output_dir` | `--log-dir` |
| `epochs` | `--max-epochs` |
| `batch_size` | `--batch-size` |
| `validation_batch_size` | `--val-batch-size` |
| `max_length` | `--max-sequence-length` |
| `learning_rate` | `--learning-rate` |
| `weight_decay` | `--weight-decay` |
| `seed` | `--seed` |
| `device` | `--device` |
| `mode` | `--mode` |
| `clean` | `--clean` |
| `quick_validation` | `--quick_val` |
| `augmentation_min_length` | `--aug_min_length` |
| `pu_loss_weight` | `--lamb` |
| `pu_type` | `--pu_type` |
| `prior` | `--prior` |
| `length_threshold` | `--len_thres` |
| `text_column` | validated as `answer`; not passed as an argument |
| `label_column` | validated as `label`; not passed as an argument |
| `num_labels` | validated as `2`; not passed as an argument |

## Data Requirements

Training CSV files must include at least:

```text
question
answer
label
```

The public sample CSV files are synthetic and only demonstrate structure. They do not reproduce the reported experiment metrics.
