from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.models.xgboost_module import XGBoostModule
from src.utils.logging_utils import configure_logging, log_event


def train_model(train_path: Path, model_path: Path) -> dict[str, str]:
    logger = configure_logging()
    df = pd.read_csv(train_path)
    model = XGBoostModule()
    model.fit(df)
    model.save(model_path)
    out = {"model_path": str(model_path), "rows": str(len(df))}
    log_event(logger, "xgboost_trained", **out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--train-path", default="data/processed/train.csv")
    parser.add_argument("--model-path", default="outputs/models/xgboost.joblib")
    args = parser.parse_args()
    result = train_model(Path(args.train_path), Path(args.model_path))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
