from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.label_engineering import DERIVED_HELPER_SCORES, FINAL_LABEL_COLUMNS, RAW_INPUT_FEATURES
from src.features.label_engineering import derive_labels
from src.data.synthetic import SyntheticConfig, generate_synthetic_dataset
from src.utils.logging_utils import configure_logging, log_event

DUNNHUMBY_EXPECTED = [
    "transaction_data.csv",
    "hh_demographic.csv",
    "product.csv",
    "campaign_desc.csv",
    "coupon.csv",
    "coupon_redempt.csv",
    "causal_data.csv",
]


def validate_dunnhumby(raw_dir: Path) -> tuple[bool, list[str]]:
    missing = [name for name in DUNNHUMBY_EXPECTED if not (raw_dir / name).exists()]
    return (len(missing) == 0, missing)


def load_dunnhumby_proxy(raw_dir: Path) -> pd.DataFrame:
    tx = pd.read_csv(raw_dir / "transaction_data.csv")
    hh = pd.read_csv(raw_dir / "hh_demographic.csv")
    tx = tx.rename(columns={"household_key": "customer_id"})
    hh = hh.rename(columns={"household_key": "customer_id"})
    df = tx.merge(hh[["customer_id"]], on="customer_id", how="left")
    if "DAY" in df.columns:
        df["recency_days"] = (df["DAY"].max() - df["DAY"]).clip(lower=0)
    else:
        df["recency_days"] = 30
    df["frequency_7d"] = 1
    df["avg_basket_value"] = df.get("SALES_VALUE", 10.0)
    df["offer_id"] = "offer_a"
    df["channel"] = "email"
    df["campaign_touches_30d"] = 2
    df["prior_response_rate"] = 0.5
    cols = [
        "customer_id",
        "recency_days",
        "frequency_7d",
        "avg_basket_value",
        "offer_id",
        "channel",
        "campaign_touches_30d",
        "prior_response_rate",
    ]
    return df[cols]


def prepare_dataset(dataset: str, raw_dir: Path, processed_dir: Path) -> dict[str, str]:
    logger = configure_logging()
    processed_dir.mkdir(parents=True, exist_ok=True)
    if dataset == "dunnhumby":
        valid, missing = validate_dunnhumby(raw_dir)
        if not valid:
            raise FileNotFoundError(f"Missing dunnhumby files: {missing}")
        df = load_dunnhumby_proxy(raw_dir)
        df["is_synthetic_row"] = 0
    else:
        df = generate_synthetic_dataset(SyntheticConfig(), raw_dir)
        df["is_synthetic_row"] = 1

    df = derive_labels(df)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    paths = {
        "train": str(processed_dir / "train.csv"),
        "val": str(processed_dir / "val.csv"),
        "test": str(processed_dir / "test.csv"),
        "all": str(processed_dir / "all.csv"),
        "metadata": str(processed_dir / "metadata.json"),
    }
    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)
    df.to_csv(paths["all"], index=False)

    metadata = {
        "dataset": dataset,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "column_groups": {
            "raw_input_features": [c for c in RAW_INPUT_FEATURES if c in df.columns],
            "derived_helper_scores": [c for c in DERIVED_HELPER_SCORES if c in df.columns],
            "final_label": [c for c in FINAL_LABEL_COLUMNS if c in df.columns],
        },
        "labels_are_proxy_policy": True,
        "relabeling_happened_after_split": False,
        "synthetic_rows_present": bool(df["is_synthetic_row"].astype(bool).any()),
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
    }
    with Path(paths["metadata"]).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log_event(logger, "dataset_prepared", **metadata)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare dunnhumby or synthetic dataset")
    parser.add_argument("--dataset", choices=["dunnhumby", "synthetic"], required=True)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--processed-dir", default="data/processed")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raw_dir = Path(args.raw_dir or f"data/raw/{args.dataset}")
    processed_dir = Path(args.processed_dir)
    paths = prepare_dataset(args.dataset, raw_dir, processed_dir)
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
