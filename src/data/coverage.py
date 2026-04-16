from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.label_engineering import derive_labels

LOW_MED_HIGH_BINS = [-np.inf, 1 / 3, 2 / 3, np.inf]
LOW_MED_HIGH_LABELS = ["low", "medium", "high"]

COVERAGE_COLUMNS = [
    "need_score",
    "fatigue_score",
    "intrusiveness_risk",
    "offer_relevance",
    "prior_response_rate",
    "campaign_touches_30d",
]


@dataclass
class CoverageConfig:
    target_size: int = 100
    seed: int = 42


def _bucketize(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=LOW_MED_HIGH_BINS, labels=LOW_MED_HIGH_LABELS, include_lowest=True).astype(str)


def _stable_round(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def _augment_row(
    row: pd.Series,
    rng: np.random.Generator,
    augmentation_type: str,
    channel_values: list[str],
    tweaks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = row.copy()
    tweaks = tweaks or {}

    # Small perturbations anchored to the source case distribution.
    out["recency_days"] = int(np.clip(int(out["recency_days"]) + int(rng.integers(-8, 9)), 0, 180))
    out["frequency_7d"] = int(np.clip(int(out["frequency_7d"]) + int(rng.integers(-2, 3)), 0, 10))
    out["avg_basket_value"] = _stable_round(float(out["avg_basket_value"]) + float(rng.normal(0, 3.0)), 2.0, 300.0)
    out["campaign_touches_30d"] = int(np.clip(int(out["campaign_touches_30d"]) + int(rng.integers(-2, 3)), 0, 16))
    out["prior_response_rate"] = _stable_round(float(out["prior_response_rate"]) + float(rng.normal(0, 0.06)), 0.0, 1.0)

    # Deterministic channel diversification to keep coverage broad across supported channels.
    if channel_values:
        out["channel"] = channel_values[int(rng.integers(0, len(channel_values)))]

    for key, value in tweaks.items():
        out[key] = value

    out["augmentation_type"] = augmentation_type
    return out.to_dict()


def _nearest(df: pd.DataFrame, expr: pd.Series, n: int = 1) -> pd.DataFrame:
    return df.assign(_dist=expr.abs()).sort_values("_dist").head(n).drop(columns=["_dist"])


def build_coverage_test_set(source_df: pd.DataFrame, cfg: CoverageConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "source_case_id" in source_df.columns:
        source = source_df.copy()
    else:
        source = source_df.reset_index(drop=True).copy()
        source["source_case_id"] = source.index.astype(str)

    if not set(COVERAGE_COLUMNS).issubset(source.columns):
        source = derive_labels(source)

    rng = np.random.default_rng(cfg.seed)
    channel_values = sorted(source["channel"].dropna().astype(str).unique().tolist())
    action_values = sorted(source["action_class"].dropna().astype(str).unique().tolist())

    cases: list[dict[str, Any]] = []

    # 1) Boundary-focused rows near important decision thresholds.
    threshold_specs = [
        ("fatigue_score", 0.66),
        ("fatigue_score", 0.70),
        ("intrusiveness_risk", 0.70),
        ("offer_relevance", 0.35),
        ("offer_relevance", 0.60),
        ("offer_relevance", 0.75),
        ("need_score", 0.45),
        ("need_score", 0.65),
    ]
    for feature, threshold in threshold_specs:
        nearest = _nearest(source, source[feature] - threshold, n=4)
        for _, row in nearest.iterrows():
            augmented = _augment_row(
                row,
                rng=rng,
                augmentation_type="boundary_focus",
                channel_values=channel_values,
                tweaks={feature: _stable_round(threshold + float(rng.normal(0, 0.015)), 0.0, 1.0)},
            )
            augmented["edge_case_flag"] = 1
            cases.append(augmented)

    # 2) Explicit conflict scenarios from original rows.
    conflicts = {
        "high_need_high_fatigue": (source["need_score"] > 0.66) & (source["fatigue_score"] > 0.66),
        "high_relevance_high_intrusiveness": (source["offer_relevance"] > 0.66) & (source["intrusiveness_risk"] > 0.66),
        "low_relevance_low_fatigue": (source["offer_relevance"] < 0.34) & (source["fatigue_score"] < 0.34),
        "high_prior_response_with_saturation": (source["prior_response_rate"] > 0.66) & (source["campaign_touches_30d"] > 8),
    }
    for scenario_name, mask in conflicts.items():
        pool = source[mask]
        if pool.empty:
            pool = source.sample(min(8, len(source)), random_state=cfg.seed)
        sample_n = min(8, len(pool))
        sampled = pool.sample(sample_n, replace=False, random_state=int(rng.integers(1, 1_000_000)))
        for _, row in sampled.iterrows():
            augmented = _augment_row(
                row,
                rng=rng,
                augmentation_type=f"conflict_{scenario_name}",
                channel_values=channel_values,
            )
            augmented["edge_case_flag"] = 1
            cases.append(augmented)

    # 3) Stratified action/channel sampling with realistic perturbation.
    for action in action_values:
        for channel in channel_values:
            pool = source[(source["action_class"] == action) & (source["channel"] == channel)]
            if pool.empty:
                continue
            sampled = pool.sample(min(3, len(pool)), replace=False, random_state=int(rng.integers(1, 1_000_000)))
            for _, row in sampled.iterrows():
                augmented = _augment_row(
                    row,
                    rng=rng,
                    augmentation_type="stratified_action_channel",
                    channel_values=channel_values,
                    tweaks={"channel": channel},
                )
                augmented["edge_case_flag"] = 0
                cases.append(augmented)

    # 4) Fill remaining with underrepresented coverage buckets.
    out = pd.DataFrame(cases)
    out = derive_labels(out)
    for col in COVERAGE_COLUMNS:
        out[f"{col}_bucket"] = _bucketize(out[col])
    out["coverage_bucket"] = out[[f"{c}_bucket" for c in COVERAGE_COLUMNS]].agg("|".join, axis=1)

    # Drop exact duplicates while preserving traceability information.
    dedupe_subset = [
        "source_case_id",
        "recency_days",
        "frequency_7d",
        "avg_basket_value",
        "offer_id",
        "channel",
        "campaign_touches_30d",
        "prior_response_rate",
    ]
    out = out.drop_duplicates(subset=dedupe_subset).reset_index(drop=True)

    if len(out) < cfg.target_size:
        source_with_buckets = source.copy()
        for col in COVERAGE_COLUMNS:
            source_with_buckets[f"{col}_bucket"] = _bucketize(source_with_buckets[col])
        source_with_buckets["coverage_bucket"] = source_with_buckets[[f"{c}_bucket" for c in COVERAGE_COLUMNS]].agg("|".join, axis=1)
        existing_counts = out["coverage_bucket"].value_counts().to_dict()
        source_with_buckets["bucket_count"] = source_with_buckets["coverage_bucket"].map(existing_counts).fillna(0)
        filler = source_with_buckets.sort_values("bucket_count").head(cfg.target_size - len(out))
        filler_records = []
        for _, row in filler.iterrows():
            aug = _augment_row(
                row,
                rng=rng,
                augmentation_type="bucket_balancing",
                channel_values=channel_values,
            )
            aug["edge_case_flag"] = 0
            filler_records.append(aug)
        out = pd.concat([out, pd.DataFrame(filler_records)], ignore_index=True)
        out = derive_labels(out)
        for col in COVERAGE_COLUMNS:
            out[f"{col}_bucket"] = _bucketize(out[col])
        out["coverage_bucket"] = out[[f"{c}_bucket" for c in COVERAGE_COLUMNS]].agg("|".join, axis=1)

    out = out.sample(min(cfg.target_size, len(out)), random_state=cfg.seed).reset_index(drop=True)
    out["case_id"] = [f"coverage_{idx:03d}" for idx in range(1, len(out) + 1)]

    summary: dict[str, Any] = {
        "target_size": cfg.target_size,
        "actual_size": int(len(out)),
        "seed": cfg.seed,
        "derived_from_rows": int(len(source)),
        "augmentation_type_counts": out["augmentation_type"].value_counts().to_dict(),
        "edge_case_count": int(out["edge_case_flag"].sum()),
        "action_distribution": out["action_class"].value_counts().to_dict(),
        "channel_distribution": out["channel"].value_counts().to_dict(),
        "coverage_bucket_count": int(out["coverage_bucket"].nunique()),
    }

    return out, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a 100-case coverage-biased evaluation set from original data")
    parser.add_argument("--input", default="data/processed/test.csv")
    parser.add_argument("--output", default="artifacts/coverage_test_set.csv")
    parser.add_argument("--summary-output", default="artifacts/coverage_summary.json")
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = pd.read_csv(args.input)
    if "source_case_id" not in source.columns:
        source = source.reset_index(drop=True)
        source["source_case_id"] = [f"original_{idx:05d}" for idx in source.index]

    out, summary = build_coverage_test_set(source, CoverageConfig(target_size=args.target_size, seed=args.seed))

    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(output_path), "summary": str(summary_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
