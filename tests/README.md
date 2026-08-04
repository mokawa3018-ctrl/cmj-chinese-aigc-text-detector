# Tests

No formal automated test suite has been added yet.

Current validation is performed through script-level checks such as:

```bash
python src/check_data.py --train data/samples/sample_train.csv --validation data/samples/sample_validation.csv
```

Future tests can cover metric calculation, data validation edge cases, and command-line argument parsing.
