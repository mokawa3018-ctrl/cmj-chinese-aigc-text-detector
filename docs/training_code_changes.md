# Training Code Changes

The training module in `src/training/` is based on `YuchuanTian/AIGC_text_detector` and preserves the Apache License 2.0 license text at the repository root.

This document records changes that are visible in the project-used version. When authorship of a specific line cannot be independently confirmed, it is described as "the project-used version" rather than attributed to a specific person.

## Included Training Files

The public training module includes the files needed by the actual training chain:

```text
src/training/train.py
src/training/dataset.py
src/training/option.py
src/training/utils.py
src/training/pu_loss_mod.py
src/training/multiscale_kit.py
src/training/corpus_cleaning_kit.py
src/training/prior_kit.py
```

`prior_kit.py` is included because `pu_loss_mod.py` imports `expected_log_beta` from it.

The following upstream files were not copied:

```text
imgs/
__pycache__/
README.md
requirements.txt
```

## Confirmed Project-Used Modifications

The project-used training code includes support for local model paths:

```text
--local-model
--model-name
```

The project-used training code includes support for local train and validation CSV paths:

```text
--local-data
--train-data-file
--val-data-file
--val_file1
```

The project-used dataset loader supports CSV training in `original_single` mode and reads:

```text
answer
label
question
```

The project-used training loop records validation confusion-matrix components:

```text
TP
FN
TN
FP
```

It also computes:

```text
accuracy
precision
recall
F1
best F1
```

The project-used training loop saves:

```text
best-model.pt
complete-{epoch}/
sentence_lengths.pkl
TensorBoard events
```

The Hugging Face-format `complete-{epoch}/` directory is saved with `save_pretrained`.

The project-used loss combines the Hugging Face classification loss with PU loss when `--lamb` is greater than zero:

```text
classification loss + lamb * PU loss
```

The final balanced experiment used:

```text
lamb = 0.4
pu_type = dual_softmax_dyn_dtrun
prior = 0.2
len_thres = 55
```

The project-used training code includes `quick_val` handling.

The project-used training code writes TensorBoard summaries through `SummaryWriter`.

## Behavior Kept Unchanged During Public整理

No training algorithm rewrite was performed when moving the project-used files into this public repository.

The following behaviors were kept as they appeared in the project-used version:

```text
CSV data reading logic
classification loss
PU loss logic
AdamW optimizer
validation metric calculation
best-model.pt saving
Hugging Face save_pretrained saving
TensorBoard logging
```

## JSON Config Status

The example config at `configs/train_multimodel_v2_balance.example.json` records the final balanced experiment parameters with public placeholder paths.

The copied training entry currently does not read JSON config files directly. It uses command-line arguments.

Equivalent public-path example command:

```bash
python src/training/train.py \
  --device cuda \
  --max-epochs 1 \
  --batch-size 16 \
  --val-batch-size 8 \
  --max-sequence-length 512 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --local-model models/base \
  --model-name AIGC_detector_zhv3 \
  --local-data data/private \
  --train-data-file train.csv \
  --val-data-file validation.csv \
  --val_file1 validation.csv \
  --data-name save \
  --mode original_single \
  --aug_min_length 0 \
  --clean 0 \
  --quick_val 0 \
  --log-dir outputs/multimodel_v2_balance
```

This command is an example using public placeholder paths. It requires the user to provide a compatible base model and private train/validation CSV files.

## Notes On Reproducibility

The training module can represent the real local experiment more closely than a newly written simplified script, because it preserves the project-used loss, validation, and checkpoint behavior.

Full reproduction still requires:

```text
compatible Python dependencies
the same or equivalent base model
the same or equivalent train/validation data
GPU or CPU compute environment
confirmed licenses for models and data
```
