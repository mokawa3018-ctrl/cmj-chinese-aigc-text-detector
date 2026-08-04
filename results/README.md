# Results

This directory publishes aggregate metrics only.

It does not include row-level prediction files, full HC3 data, private datasets, model weights, logs, or checkpoints.

The HC3 Chinese evaluation data is not redistributed in this repository. Users should obtain datasets from their official sources and follow the applicable licenses and terms.

`summary_metrics.csv` uses a long-table format:

```text
evaluation_set, model, metric, value, unit, notes
```

Metrics from different `evaluation_set` values should not be compared blindly. Validation metrics are used for training selection, while fixed test sets and the HC3 Chinese 7696-row test are used for external evaluation.

Some count changes are marked as approximate when they are inferred from rounded percentages.
