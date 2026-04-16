from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.agentic.controller import AdaptiveAgenticController
from src.config import load_yaml
from src.evaluation.metrics import evaluate_predictions
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


def run_experiment(
    mode: str,
    test_df: pd.DataFrame,
    cfg: dict,
    logger=None,
    progress_every: int = 1,
) -> tuple[list[dict], dict]:
    controller = AdaptiveAgenticController(cfg)
    rows = test_df.to_dict(orient="records")
    total = len(rows)
    preds: list[dict] = []
    start = time.perf_counter()
    for idx, row in enumerate(rows, start=1):
        preds.append(controller.decide(row, mode=mode))
        if logger and (idx % progress_every == 0 or idx == total):
            log_event(logger, "mode_progress", mode=mode, processed=idx, total=total)
    elapsed = time.perf_counter() - start
    metrics = evaluate_predictions(test_df, preds, elapsed)
    return preds, metrics


def _resolve_eval_sets(args: argparse.Namespace) -> dict[str, Path]:
    if args.evaluation_set == "coverage":
        return {"coverage": Path(args.coverage_test_path)}
    if args.evaluation_set == "original":
        return {"original": Path(args.test_path)}
    return {"original": Path(args.test_path), "coverage": Path(args.coverage_test_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive marketing experiments")
    parser.add_argument("--mode", default="adaptive_full_framework", choices=MODES + ["all"])
    parser.add_argument("--test-path", default="data/processed/test.csv")
    parser.add_argument("--coverage-test-path", default="artifacts/coverage_test_set.csv")
    parser.add_argument("--evaluation-set", default="coverage", choices=["coverage", "original", "both"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-mode", default="synthetic")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    logger = configure_logging()
    cfg = load_yaml(args.config)
    evaluation_sets = _resolve_eval_sets(args)

    loaded_datasets: dict[str, pd.DataFrame] = {}
    for eval_name, eval_path in evaluation_sets.items():
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation set '{eval_name}' does not exist at: {eval_path}")
        eval_df = pd.read_csv(eval_path)
        if args.max_rows is not None:
            eval_df = eval_df.head(args.max_rows).copy()
        loaded_datasets[eval_name] = eval_df
        log_event(
            logger,
            "experiment_dataset_loaded",
            evaluation_set=eval_name,
            path=str(eval_path),
            rows=len(eval_df),
            max_rows=args.max_rows,
        )

    modes = MODES if args.mode == "all" else [args.mode]
    all_metrics: dict[str, dict] = {}
    dataset_summary: dict[str, dict[str, str | int]] = {}

    for eval_name, eval_path in evaluation_sets.items():
        dataset_summary[eval_name] = {"rows": len(loaded_datasets[eval_name]), "path": str(eval_path)}

    example_preds: list[dict] = []
    for mode in modes:
        for eval_name, test_df in loaded_datasets.items():
            metric_key = f"{mode}__{eval_name}"
            log_event(logger, "mode_start", mode=mode, evaluation_set=eval_name)
            preds, metrics = run_experiment(mode, test_df, cfg, logger=logger)
            all_metrics[metric_key] = {**metrics, "evaluation_set": eval_name}
            if not example_preds:
                example_preds = preds
            pd.DataFrame(preds).to_csv(f"outputs/predictions/{metric_key}.csv", index=False)
            log_event(
                logger,
                "experiment_complete",
                mode=mode,
                evaluation_set=eval_name,
                accuracy=metrics.get("accuracy"),
                multiclass_accuracy=metrics.get("multiclass", {}).get("accuracy"),
                rule_violation_rate=metrics.get("rule_violation_rate"),
            )

    write_reports(
        Path("outputs/reports"),
        all_metrics,
        dataset_mode=args.dataset_mode,
        dataset_summary={
            "dataset_mode": args.dataset_mode,
            "evaluation_set": args.evaluation_set,
            "sets": dataset_summary,
        },
        feature_summary={
            eval_name: {"features": list(df.columns), "rows": len(df)} for eval_name, df in loaded_datasets.items()
        },
        example_decisions=example_preds,
    )
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
