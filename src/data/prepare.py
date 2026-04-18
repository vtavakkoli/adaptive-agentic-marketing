from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.label_engineering import DERIVED_HELPER_SCORES, FINAL_LABEL_COLUMNS, RAW_INPUT_FEATURES
from src.features.label_engineering import derive_labels
from src.data.synthetic import SyntheticConfig, generate_synthetic_dataset
from src.utils.logging_utils import configure_logging, log_event

DUNNHUMBY_EXPECTED =[
    "transaction_data.csv",
    "hh_demographic.csv",
    "product.csv",
    "campaign_desc.csv",
    "coupon.csv",
    "coupon_redempt.csv",
    "causal_data.csv",
]


def validate_dunnhumby(raw_dir: Path) -> tuple[bool, list[str]]:
    missing =[name for name in DUNNHUMBY_EXPECTED if not (raw_dir / name).exists()]
    return (len(missing) == 0, missing)


def load_dunnhumby_proxy(raw_dir: Path) -> pd.DataFrame:
    tx = pd.read_csv(raw_dir / "transaction_data.csv")
    hh = pd.read_csv(raw_dir / "hh_demographic.csv")
    
    tx = tx.rename(columns={"household_key": "customer_id"})
    hh = hh.rename(columns={"household_key": "customer_id"})
    
    df = tx.merge(hh[["customer_id"]], on="customer_id", how="left")
    
    if "DAY" not in df.columns:
        df["DAY"] = 30
        
    df["recency_days"] = (df["DAY"].max() - df["DAY"]).clip(lower=0)
    
    # Convert DAY (integer) to a simulated date for rolling window calculations
    df["date"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(df["DAY"], unit="D")
    
    # Sort strictly by time to ensure calculations are historically robust
    df = df.sort_values(by=["customer_id", "date"]).reset_index(drop=True)
    
    # --- Base Column Setup (Must be done before creating the indexed DataFrame) ---
    df["dummy_count"] = 1
    
    if "RETAIL_DISC" in df.columns:
        df["has_disc"] = (df["RETAIL_DISC"] < 0).astype(int)
    else:
        df["has_disc"] = 0
    
    # Set date as index temporarily for the groupby engine
    df_idx = df.set_index("date")
    grouped_idx = df_idx.groupby("customer_id")
    grouped_orig = df.groupby("customer_id")
    
    # 1. Frequency 7d (Transactions in the trailing 7 days)
    # Using .values bypasses Pandas implicit index alignment which fails on duplicate dates
    roll_7d = grouped_idx["dummy_count"].rolling("7D").count().values
    df["frequency_7d"] = np.clip(roll_7d - 1, 0, None) 
    
    # 2. Campaign touches 30d
    roll_30d = grouped_idx["dummy_count"].rolling("30D").count().values
    df["campaign_touches_30d"] = np.clip(roll_30d - 1, 0, None)
    
    # 3. Average basket value
    if "SALES_VALUE" in df.columns:
        df["avg_basket_value"] = grouped_orig["SALES_VALUE"].transform(lambda x: x.shift().expanding().mean()).fillna(10.0)
    else:
        df["avg_basket_value"] = 10.0
        
    # 4. Prior response rate (Proxy: frequency of discount usage historically)
    df["prior_response_rate"] = grouped_orig["has_disc"].transform(lambda x: x.shift().expanding().mean()).fillna(0.0)
    
    # 5. rolling_response_rate_30d
    roll_disc_30d = grouped_idx["has_disc"].rolling("30D").sum().values
    roll_disc_30d_prev = np.clip(roll_disc_30d - df["has_disc"].values, 0, None)
    
    safe_denom = np.where(df["campaign_touches_30d"].values == 0, 1, df["campaign_touches_30d"].values)
    df["rolling_response_rate_30d"] = np.where(
        df["campaign_touches_30d"].values > 0, 
        roll_disc_30d_prev / safe_denom, 
        0.0
    )
                                               
    # 6. Offer ID and Channel (Deterministically mapped based on actual categorical distributions)
    channels = ["email", "sms", "push"]
    if "STORE_ID" in df.columns:
        df["channel"] = df["STORE_ID"].fillna(0).astype(int).apply(lambda x: channels[x % 3])
    else:
        df["channel"] = "email"
        
    if "WEEK_NO" in df.columns:
        df["offer_id"] = "offer_" + (df["WEEK_NO"] % 5).astype(str)
    else:
        df["offer_id"] = "offer_0"

    # Fulfill Feature Analysis Step 3 Recommendations: Add engineered interaction features
    df["recency_days_x_prior_response_rate"] = df["recency_days"] * df["prior_response_rate"]
    df["campaign_touches_30d_x_recency_days"] = df["campaign_touches_30d"] * df["recency_days"]
    df["avg_basket_value_x_frequency_7d"] = df["avg_basket_value"] * df["frequency_7d"]
    df["channel_x_offer_id"] = df["channel"].astype(str) + "_" + df["offer_id"].astype(str)

    cols =[
        "customer_id",
        "recency_days",
        "frequency_7d",
        "avg_basket_value",
        "offer_id",
        "channel",
        "campaign_touches_30d",
        "prior_response_rate",
        "recency_days_x_prior_response_rate",
        "campaign_touches_30d_x_recency_days",
        "avg_basket_value_x_frequency_7d",
        "channel_x_offer_id",
        "rolling_response_rate_30d",
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
            "raw_input_features":[c for c in RAW_INPUT_FEATURES if c in df.columns],
            "derived_helper_scores":[c for c in DERIVED_HELPER_SCORES if c in df.columns],
            "final_label":[c for c in FINAL_LABEL_COLUMNS if c in df.columns],
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
