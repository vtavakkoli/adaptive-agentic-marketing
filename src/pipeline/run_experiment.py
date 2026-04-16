from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.agentic.controller import AdaptiveAgenticController
from src.config import load_yaml
from src.evaluation.metrics import evaluate_predictions, timed_decisions
from src.evaluation.report import write_reports
from src.utils.logging_utils import configure_logging, log_event

MODES = [
    "rules_only",
    "xgboost_only",
    "slm_only",
    "adaptive_full_framework",
    "ablation_no_rules",
    "ablation_no_xgboost",
    "ablation_no_explanation",
    "ablation_no_content_generation",
]


def run_experiment(mode: str, test_df: pd.DataFrame, cfg: dict) -> tuple[list[dict], dict]:
    controller = AdaptiveAgenticController(cfg)
    rows = test_df.to_dict(orient="records")
    preds, elapsed = timed_decisions(lambda r: controller.decide(r, mode=mode), rows)
    metrics = evaluate_predictions(test_df, preds, elapsed)
    return preds, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive marketing experiments")
    parser.add_argument("--mode", default="adaptive_full_framework", choices=MODES + ["all"])
    parser.add_argument("--test-path", default="data/processed/test.csv")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-mode", default="synthetic")
    args = parser.parse_args()

    logger = configure_logging()
    cfg = load_yaml(args.config)
    test_df = pd.read_csv(args.test_path)

    modes = MODES if args.mode == "all" else [args.mode]
    all_metrics: dict[str, dict] = {}
    example_preds: list[dict] = []
    for mode in modes:
        preds, metrics = run_experiment(mode, test_df, cfg)
        all_metrics[mode] = metrics
        if not example_preds:
            example_preds = preds
        pd.DataFrame(preds).to_csv(f"outputs/predictions/{mode}.csv", index=False)
        log_event(logger, "experiment_complete", mode=mode, **metrics)

    write_reports(
        Path("outputs/reports"),
        all_metrics,
        dataset_mode=args.dataset_mode,
        dataset_summary={"rows": len(test_df), "mode": args.dataset_mode},
        feature_summary={"features": list(test_df.columns)},
        example_decisions=example_preds,
    )
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
