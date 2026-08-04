# Experiment Summary

## Goal

This project reproduced and fine-tuned a Chinese AIGC text detection model, evaluated it on fixed test sets, and exported the final selected model to ONNX.

The binary label definition is:

- `0`: human-written or human-style text.
- `1`: AI-generated or AI-style text.

## Fixed Test Sets

The local experiment used fixed test sets to compare model versions:

- Human-only fixed test set.
- AI-only fixed test set.
- Mixed fixed test set.

These fixed test sets were not used for training.

Validation metrics and external test metrics should not be mixed as if they measure the same thing. Validation metrics are used during model selection, while fixed test sets and larger external tests are used to evaluate generalization under the chosen data split.

## Original zh-v3 Baseline

On the fixed test sets, the original zh-v3 model achieved:

- Human text accuracy: 89.10%.
- Human false positive rate: 10.90%.
- AI text recall: 90.50%.
- Mixed test accuracy: 89.80%.

These results establish the baseline for the local fine-tuning experiments.

## Fine-Tuning Experiments

Three fine-tuning experiments were tracked:

- GLM single-model fine-tuning.
- Multi-model 1:5 non-balanced fine-tuning.
- Multi-model 1:1 balanced fine-tuning.

The GLM single-model fine-tuning experiment has one confirmed metric:

- Validation F1: 91.16%.

The multi-model 1:5 experiment used more AI samples than human samples. Its validation metrics were strong:

- Accuracy: 95.00%.
- Precision: 95.48%.
- Recall: 98.67%.
- F1: 97.05%.

However, on the external fixed tests, the 1:5 model raised the human false positive rate to 15.50%. This means the model became more biased toward predicting the AI class. Although AI recall improved to 97.30%, the higher false positive rate made it less suitable as the final selected model.

The final multi-model 1:1 balanced model used balanced human and AI training labels:

- Training set: 1,700 rows.
- Validation set: 300 rows.
- Training labels: 850 human and 850 AI.
- Validation labels: 150 human and 150 AI.
- Five AI generation sources were balanced within the AI samples.
- Data was split by `pair_id` to reduce question-level leakage between training and validation.

Its validation metrics were:

- Accuracy: 89.33%.
- Precision: 86.42%.
- Recall: 93.33%.
- F1: 89.74%.

Its fixed test metrics were:

- Human text accuracy: 91.60%.
- Human false positive rate: 8.40%.
- AI text recall: 94.10%.
- Mixed test accuracy: 92.85%.

## Why The 1:1 Model Was Selected

The 1:5 model had a higher validation F1, but it also pushed the model toward the AI class and increased the human false positive rate to 15.50% on the fixed human test set.

The final 1:1 model was selected because it provided a better balance among:

- Lower human false positive rate.
- Improved AI recall compared with the original zh-v3 baseline.
- Higher mixed test accuracy compared with both the original baseline and the 1:5 model.

This choice reflects the practical cost of false positives in AIGC detection. A model that over-detects AI text can be risky when results are used to review human-written content.

## HC3 Chinese 7696-Row Evaluation

The larger HC3 Chinese evaluation contained:

- Human text: 4,481 rows.
- AI text: 3,215 rows.
- Total: 7,696 rows.

The original model achieved:

- Overall accuracy: 89.54%.
- AI detection rate: 91.14%.
- Human false positive rate: 11.60%.

The final 1:1 balanced model achieved:

- Overall accuracy: 91.84%.
- AI detection rate: 92.78%.
- Human false positive rate: 8.84%.

The measured changes were:

- Overall accuracy: +2.30 percentage points.
- AI detection rate: +1.64 percentage points.
- Human false positive rate: -2.76 percentage points.
- AI correct detections increased by 53 rows.
- Human false positives decreased by approximately 124 rows.

The human false positive reduction is marked as approximate because it is inferred from rounded percentage metrics and the 4,481-row human subset.

## ONNX Consistency

The final balanced PyTorch model was exported to ONNX.

PyTorch and ONNX consistency validation was completed in the original server environment. The local release-packaging environment did not reload the ONNX graph or repeat inference.

The server-side PyTorch and ONNX consistency check found:

- Maximum logits absolute difference: 1.43e-6.
- Maximum probability absolute difference: 2.31e-7.
- Tested sample predicted labels were fully consistent.

The actual ONNX export script configured:

- Inputs: `input_ids`, `attention_mask`, `token_type_ids`.
- Output: `logits`.
- Opset: 14.
- Dynamic batch: enabled in the export configuration.
- Dynamic sequence length: enabled in the export configuration.

These ONNX input names, dynamic axes, and opset details come from the export script configuration. They were not re-derived from a local `onnx.checker` run in the release-packaging environment.

ONNX fixed test metrics matched the final model summary:

- Human text accuracy: 91.60%.
- Human false positive rate: 8.40%.
- AI text recall: 94.10%.
- Mixed test accuracy: 92.85%.

## Error Analysis

The `nlpcc_dbqa` source category was the clearest weak area in the available error analysis:

- Overall accuracy: about 76.55%.
- Human false positive rate: about 32.38%.
- AI recall: about 81.83%.

Legal and finance categories were also weaker than medicine, encyclopedia, psychology, and open-ended QA categories, but this document does not list additional exact category metrics because they were not part of the current public summary request.

Likely follow-up directions:

- Add more high-quality paired samples for `nlpcc_dbqa`, legal, and finance categories.
- Analyze the effect of text length.
- Analyze the effect of different generation models.
- Analyze data source distribution bias.
- Validate generalization with an independent out-of-domain test set.

## Limitations

These metrics represent specific datasets, splits, and model versions. They should not be interpreted as universal performance on all Chinese text.

AIGC detection should be treated as an auxiliary signal. It should not be used alone for punishment, academic misconduct judgment, identity judgment, or other high-impact decisions.

Percentages are kept at the precision reported in the local experiment. Counts inferred from rounded percentages are marked as approximate rather than treated as exact confusion-matrix values.
