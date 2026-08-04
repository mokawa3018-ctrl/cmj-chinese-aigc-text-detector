# Sample Data

This directory contains small, newly written synthetic CSV files:

- `sample_train.csv`
- `sample_validation.csv`

These files are only for demonstrating the expected field structure, data validation, and a minimal smoke-test flow. They are not real collected human data, not the original training data, and not sufficient for reproducing experiment metrics.

The full training and test datasets are not distributed in this repository because their licenses, privacy status, and publication scope must be confirmed separately.

The real experiment split data by `pair_id` to avoid question-level leakage between training and validation.
