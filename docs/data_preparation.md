# Data Preparation

## Modes
- `dunnhumby`: validates required raw files.
- `synthetic`: auto-generates schema-compatible demo data.

## Command
```bash
python -m src.data.prepare --help
```

## Artifacts
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/all.csv`
- `data/processed/metadata.json`
