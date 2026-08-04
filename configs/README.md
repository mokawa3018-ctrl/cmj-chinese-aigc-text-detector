# Configs

This directory stores public example configuration files.

`train_multimodel_v2_balance.example.json` records the final balanced fine-tuning experiment parameters with placeholder relative paths. It can be read by `src/train_from_config.py`, which validates the JSON and converts it into the real `src/training/train.py` command-line arguments.

Preview the resolved command without starting training:

```bash
python src/train_from_config.py --config configs/train_multimodel_v2_balance.example.json --dry-run
```

The placeholder model and private data paths must exist before running without `--dry-run`.

Do not store server addresses, account names, tokens, private paths, or private dataset locations here.
