from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.agentic.controller import AdaptiveAgenticController, MODE_ALIASES
from src.config import load_yaml
from src.data.feature_builder import build_features
from src.data.label_audit import feature_audit, run_label_audit
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.report import write_reports
from src.utils.logging_utils import configure_logging, log_event

MODES = [
    "rules_only",
    "xgboost_only",
    "slm_only",
    "adaptive_framework",
    "adaptive_hierarchical",
    "adaptive_ppo_agent",
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
    mode = MODE_ALIASES.get(mode, mode)
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
    metrics["label_audit"] = run_label_audit(build_features(test_df))
    metrics["feature_audit"] = feature_audit(build_features(test_df), pd.DataFrame(preds))
    return preds, metrics


def _resolve_eval_sets(args: argparse.Namespace) -> dict[str, Path]:
    if args.evaluation_set in {"diagnostic", "coverage", "unbiased"}:
        return {"diagnostic": Path(args.diagnostic_test_path)}
    if args.evaluation_set == "original":
        return {"original": Path(args.test_path)}
    return {"original": Path(args.test_path), "diagnostic": Path(args.diagnostic_test_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive marketing experiments")
    parser.add_argument("--mode", default="adaptive_hierarchical", choices=MODES + ["all", "adaptive_full", "adaptive_simple"])
    parser.add_argument("--test-path", default="data/processed/test.csv")
    parser.add_argument("--diagnostic-test-path", default="artifacts/diagnostic_balanced_100.csv")
    parser.add_argument("--evaluation-set", default="original", choices=["original", "diagnostic", "coverage", "unbiased", "both"])
    parser.add_argument("--config", default="configs/adaptive_hierarchical.yaml")
    parser.add_argument("--dataset-mode", default="synthetic")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
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

    requested_mode = MODE_ALIASES.get(args.mode, args.mode)
    modes = MODES if requested_mode == "all" else [requested_mode]
    if "adaptive_ppo_agent" in modes:
        model_path = Path(cfg.get("ppo", {}).get("model_path", "outputs/models/adaptive_ppo_agent.pt"))
        if not model_path.exists():
            if requested_mode == "all":
                modes = [m for m in modes if m != "adaptive_ppo_agent"]
                log_event(
                    logger,
                    "adaptive_ppo_agent_skipped",
                    reason="missing_model",
                    model_path=str(model_path),
                    hint=f"Train first with: python -m src.rl.train_ppo --train-path data/processed/train.csv --model-path {model_path}",
                )
            else:
                raise FileNotFoundError(
                    f"Missing adaptive_ppo_agent model at {model_path}. Train it first with: python -m src.rl.train_ppo --train-path data/processed/train.csv --model-path {model_path}"
                )
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
            metrics["calibration_enabled"] = bool(args.calibrate)
            metrics["seeds"] = int(args.seeds)
            all_metrics[metric_key] = {**metrics, "evaluation_set": eval_name}
            if not example_preds:
                example_preds = preds
            pred_dir = Path("outputs/predictions") / mode / eval_name
            pred_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(preds).to_csv(pred_dir / "predictions.csv", index=False)
            log_event(
                logger,
                "experiment_complete",
                mode=mode,
                evaluation_set=eval_name,
                accuracy=metrics.get("accuracy"),
                multiclass_accuracy=metrics.get("multiclass", {}).get("accuracy"),
                rule_violation_rate=metrics.get("rule_violation_rate"),
            )

    if args.report or True:
        write_reports(
            Path("outputs/reports"),
            all_metrics,
            dataset_mode=args.dataset_mode,
            dataset_summary={
                "dataset_mode": args.dataset_mode,
                "evaluation_set": args.evaluation_set,
                "primary_benchmark_definition": "held-out original test rows",
                "diagnostic_benchmark_definition": "balanced diagnostic subset sampled from held-out test rows",
                "sets": dataset_summary,
                "seed": args.seeds,
                "config_path": args.config,
                "mode": requested_mode,
            },
            feature_summary={
                eval_name: {"features": list(df.columns), "rows": len(df)} for eval_name, df in loaded_datasets.items()
            },
            example_decisions=example_preds,
        )
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
