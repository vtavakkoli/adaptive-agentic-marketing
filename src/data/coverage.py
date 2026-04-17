from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.label_engineering import derive_labels

EVAL_COLUMNS = [
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
    strategy: str = "unbiased"


def _ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    if set(EVAL_COLUMNS).issubset(df.columns) and "action_class" in df.columns:
        return df.copy()
    return derive_labels(df.copy())


def _stratified_unbiased_sample(source: pd.DataFrame, cfg: CoverageConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    work = source.copy()
    work["_strata"] = work["action_class"].astype(str) + "|" + work["channel"].astype(str)

    total = min(cfg.target_size, len(work))
    counts = work["_strata"].value_counts()
    alloc = (counts / counts.sum() * total).round().astype(int)
    if alloc.sum() == 0:
        alloc.iloc[0] = total

    diff = total - int(alloc.sum())
    if diff != 0:
        order = counts.sort_values(ascending=False).index.tolist()
        i = 0
        while diff != 0 and order:
            k = order[i % len(order)]
            if diff > 0:
                alloc[k] += 1
                diff -= 1
            elif alloc[k] > 0:
                alloc[k] -= 1
                diff += 1
            i += 1

    chunks: list[pd.DataFrame] = []
    for strata, n in alloc.items():
        if n <= 0:
            continue
        pool = work[work["_strata"] == strata]
        take = min(n, len(pool))
        if take > 0:
            chunks.append(pool.sample(take, replace=False, random_state=int(rng.integers(1, 1_000_000))))

    out = pd.concat(chunks, ignore_index=True) if chunks else work.head(total).copy()
    if len(out) < total:
        remaining = work.drop(index=out.index, errors="ignore")
        need = min(total - len(out), len(remaining))
        if need > 0:
            out = pd.concat([out, remaining.sample(need, random_state=cfg.seed)], ignore_index=True)

    out = out.sample(total, random_state=cfg.seed).reset_index(drop=True)
    out["sample_type"] = "unbiased_stratified"
    out["edge_case_flag"] = 0
    out["case_id"] = [f"unbiased_{idx:03d}" for idx in range(1, len(out) + 1)]
    return out.drop(columns=["_strata"], errors="ignore")


def build_coverage_test_set(source_df: pd.DataFrame, cfg: CoverageConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = source_df.reset_index(drop=True).copy()
    if "source_case_id" not in source.columns:
        source["source_case_id"] = [f"original_{idx:05d}" for idx in source.index]

    source = _ensure_labels(source)

    if cfg.strategy != "unbiased":
        raise ValueError("Only strategy='unbiased' is supported for evaluation-set generation")

    out = _stratified_unbiased_sample(source, cfg)

    summary: dict[str, Any] = {
        "target_size": cfg.target_size,
        "actual_size": int(len(out)),
        "seed": cfg.seed,
        "strategy": cfg.strategy,
        "derived_from_rows": int(len(source)),
        "sample_type_counts": out["sample_type"].value_counts().to_dict(),
        "edge_case_count": int(out["edge_case_flag"].sum()),
        "action_distribution": out["action_class"].value_counts().to_dict(),
        "channel_distribution": out["channel"].value_counts().to_dict(),
    }
    return out, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a 100-case unbiased evaluation set from original data")
    parser.add_argument("--input", default="data/processed/test.csv")
    parser.add_argument("--output", default="artifacts/unbiased_eval_set.csv")
    parser.add_argument("--summary-output", default="artifacts/unbiased_summary.json")
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", default="unbiased", choices=["unbiased"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = pd.read_csv(args.input)
    out, summary = build_coverage_test_set(
        source,
        CoverageConfig(target_size=args.target_size, seed=args.seed, strategy=args.strategy),
    )

    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(output_path), "summary": str(summary_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
