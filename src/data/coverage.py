from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.label_engineering import (
    DERIVED_HELPER_SCORES,
    FINAL_LABEL_COLUMNS,
    RAW_INPUT_FEATURES,
    derive_labels,
)

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
    diagnostic_target_size: int = 100
    diagnostic_per_class: int = 25
    seed: int = 42
    strategy: str = "heldout_primary_plus_balanced_diagnostic"


def _ensure_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if set(EVAL_COLUMNS).issubset(df.columns) and "action_class" in df.columns:
        return df.copy(), False
    return derive_labels(df.copy()), True


def _balanced_diagnostic_sample(source: pd.DataFrame, cfg: CoverageConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    rng = np.random.default_rng(cfg.seed)
    work = source.copy()
    class_order = ["do_nothing", "defer_action", "send_information", "send_reminder"]
    chunks: list[pd.DataFrame] = []
    alloc: dict[str, int] = {}

    for action in class_order:
        pool = work[work["action_class"].astype(str) == action]
        take = min(cfg.diagnostic_per_class, len(pool))
        alloc[action] = int(take)
        if take == 0:
            continue
        chunks.append(pool.sample(take, replace=False, random_state=int(rng.integers(1, 1_000_000))))

    out = pd.concat(chunks, ignore_index=True) if chunks else work.iloc[0:0].copy()
    total_target = min(cfg.diagnostic_target_size, len(work))
    if len(out) < total_target:
        remaining = work.drop(index=out.index, errors="ignore")
        need = min(total_target - len(out), len(remaining))
        if need > 0:
            extra = remaining.sample(need, random_state=cfg.seed)
            out = pd.concat([out, extra], ignore_index=True)

    out = out.sample(len(out), random_state=cfg.seed).reset_index(drop=True)
    out["sample_type"] = "balanced_diagnostic"
    out["edge_case_flag"] = 0
    out["case_id"] = [f"diagnostic_{idx:03d}" for idx in range(1, len(out) + 1)]
    return out, alloc


def build_coverage_test_set(source_df: pd.DataFrame, cfg: CoverageConfig) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    source = source_df.reset_index(drop=True).copy()
    if "source_case_id" not in source.columns:
        source["source_case_id"] = [f"original_{idx:05d}" for idx in source.index]

    labeled_source, relabeled_after_split = _ensure_labels(source)

    if cfg.strategy != "heldout_primary_plus_balanced_diagnostic":
        raise ValueError("Only strategy='heldout_primary_plus_balanced_diagnostic' is supported")

    duplicate_count = int(labeled_source.duplicated().sum())
    deduped = labeled_source.drop_duplicates().reset_index(drop=True)
    full = labeled_source.copy()
    full["sample_type"] = "heldout_primary"
    full["edge_case_flag"] = 0
    full["case_id"] = [f"test_{idx:05d}" for idx in range(1, len(full) + 1)]

    diagnostic, class_alloc = _balanced_diagnostic_sample(deduped, cfg)

    synthetic_present = bool(
        labeled_source["is_synthetic_row"].astype(bool).any() if "is_synthetic_row" in labeled_source.columns else False
    )
    summary: dict[str, Any] = {
        "primary_target_size": int(len(full)),
        "diagnostic_target_size": cfg.diagnostic_target_size,
        "diagnostic_actual_size": int(len(diagnostic)),
        "seed": cfg.seed,
        "strategy": cfg.strategy,
        "derived_from_rows": int(len(source)),
        "audit": {
            "duplicate_count_source": duplicate_count,
            "duplicate_count_diagnostic": int(diagnostic.duplicated().sum()),
            "class_counts_exact_source": labeled_source["action_class"].value_counts().to_dict(),
            "class_counts_exact_full_test": full["action_class"].value_counts().to_dict(),
            "class_counts_exact_diagnostic": diagnostic["action_class"].value_counts().to_dict(),
            "synthetic_rows_present": synthetic_present,
            "relabeled_after_split": relabeled_after_split,
            "column_groups": {
                "raw_input_features": [c for c in RAW_INPUT_FEATURES if c in labeled_source.columns],
                "derived_helper_scores": [c for c in DERIVED_HELPER_SCORES if c in labeled_source.columns],
                "final_label": [c for c in FINAL_LABEL_COLUMNS if c in labeled_source.columns],
            },
        },
        "diagnostic_class_allocation": class_alloc,
    }
    return {"full_test_benchmark": full, "diagnostic_balanced_100": diagnostic}, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build held-out full and diagnostic balanced evaluation sets from test data")
    parser.add_argument("--input", default="data/processed/test.csv")
    parser.add_argument("--output-full", default="artifacts/full_test_benchmark.csv")
    parser.add_argument("--output-diagnostic", default="artifacts/diagnostic_balanced_100.csv")
    parser.add_argument("--summary-output", default="artifacts/evaluation_set_audit.json")
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", default="heldout_primary_plus_balanced_diagnostic", choices=["heldout_primary_plus_balanced_diagnostic"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = pd.read_csv(args.input)
    outputs, summary = build_coverage_test_set(
        source,
        CoverageConfig(diagnostic_target_size=args.target_size, seed=args.seed, strategy=args.strategy),
    )

    output_full_path = Path(args.output_full)
    output_diag_path = Path(args.output_diagnostic)
    summary_path = Path(args.summary_output)
    output_full_path.parent.mkdir(parents=True, exist_ok=True)
    output_diag_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    outputs["full_test_benchmark"].to_csv(output_full_path, index=False)
    outputs["diagnostic_balanced_100"].to_csv(output_diag_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_full": str(output_full_path),
                "output_diagnostic": str(output_diag_path),
                "summary": str(summary_path),
                **summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
