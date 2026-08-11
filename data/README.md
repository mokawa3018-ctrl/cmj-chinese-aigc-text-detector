# Dataset Files

This directory contains the public CSV files used for training and evaluation examples in this repository.

## Train

- `train/train.csv`: balanced multi-model fine-tuning training set.
- `train/validation.csv`: validation set split by `pair_id`.

Label definition:

- `0`: human-written text
- `1`: AI-generated text

The training split contains 1,700 rows:

- human: 850
- GPT初始: 170
- GPT-5: 170
- Qwen3: 170
- DeepSeek: 170
- GLM: 170

The validation split contains 300 rows:

- human: 150
- GPT初始: 30
- GPT-5: 30
- Qwen3: 30
- DeepSeek: 30
- GLM: 30

## Test

- `test/human_1000.csv`: human-only test set, 1,000 rows.
- `test/ai_balanced_1000.csv`: AI-only balanced test set, 1,000 rows.
- `test/mixed_2000.csv`: mixed human/AI test set, 2,000 rows.
- `test/zh_test.csv`: Chinese HC3-style test set, 7,696 rows.
- `test/non_qa_200.csv`: supplementary non-QA test set, 200 rows.

## Columns

Common columns include:

- `question`: source question or prompt.
- `answer`: text to classify.
- `label`: ground-truth label.
- `generator`: text source or model family.
- `source`: topic/source category.
- `sample_id` or `id`: sample identifier.

Some files include extra metadata columns such as `pair_id`, `dataset_role`, `target_char_count`, or `mixed_test_id`.

## Notes

These CSV files are intended for reproducible evaluation and fine-tuning experiments. Prediction result files are not included here and must not be used as training data.

Before redistributing or reusing the data in another project, verify that the dataset license and source-data permissions fit your use case.
