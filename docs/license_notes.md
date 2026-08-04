# License Notes

This document records licensing information that was visible in the local project files. It is not legal advice.

## Official Training Code

Confirmed local source:

```text
YuchuanTian/AIGC_text_detector
```

The local `official_aigc_text_detector/LICENSE` file contains Apache License 2.0.

The original local experiment used this official training code with adaptations for:

- Local model paths.
- Local CSV train and validation files.
- CSV reading for the local `answer` and `label` format.
- Additional metric reporting.
- Hugging Face-format model saving.
- `best-model.pt` checkpoint saving.

Current public-repository status:

This repository does not currently copy the official training code.

If official training code is copied later, the public repository should preserve:

- The original Apache License 2.0 text.
- Original copyright notices.
- Clear attribution to `YuchuanTian/AIGC_text_detector`.
- A short description of local modifications.

## Base Models And Weights

The local official zh-v3 model directory contained a README with:

```text
license: apache-2.0
```

The local Chinese RoBERTa WWM base model directory also contained a README with:

```text
license: apache-2.0
```

However, local files alone are not enough to make a final redistribution decision for model weights.

Before publishing any model weights, verify the official model pages and their complete model cards:

- Official zh-v3 detector source.
- Chinese RoBERTa WWM source.
- Any hosting platform terms that apply to the downloaded files.

## Fine-Tuned Weights

The final fine-tuned weights may be affected by:

- The license of the official zh-v3 base model.
- The license of the underlying Chinese RoBERTa WWM model.
- The license and authorization status of all training data.
- Any privacy restrictions in the training data.

This repository should not include fine-tuned weights until all relevant model and data permissions are confirmed.

## Training And Evaluation Data

The original local experiment used curated training and evaluation CSV files. Full datasets are not included in this public repository at this stage.

Before publishing full datasets, confirm:

- The source dataset licenses.
- Whether generated model responses can be redistributed.
- Whether the data contains personal information.
- Whether company-internal or private data is present.
- Whether the final publication scope is approved.

## Current Publication Boundary

At this stage, the repository should include only:

- Public utility scripts.
- Public configuration examples.
- Documentation of the workflow.
- Small artificial or clearly publishable sample data.
- Aggregated and sanitized metrics after review.

At this stage, the repository should not include:

- Unconfirmed model weights.
- Full private datasets.
- Full prediction-detail CSV files.
- Server logs.
- Interface scripts containing private endpoints, accounts, tokens, or company-internal information.

Users who want to reproduce the training must obtain suitable base models and datasets themselves under the applicable licenses.
