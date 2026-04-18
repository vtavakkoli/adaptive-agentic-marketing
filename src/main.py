from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.data.prepare import validate_dunnhumby
from src.utils.logging_utils import configure_logging, log_event


def run(cmd: list[str], logger=None, stage: str | None = None) -> None:
    if logger and stage:
        log_event(logger, "full_test_stage_start", stage=stage, command=cmd)
    print("$", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        if logger and stage:
            log_event(logger, "full_test_stage_failed", stage=stage, returncode=result.returncode)
        raise SystemExit(result.returncode)
    if logger and stage:
        log_event(logger, "full_test_stage_complete", stage=stage)


def full_test(max_rows: int | None = None, evaluation_set: str = "unbiased", seed: int = 42) -> None:
    logger = configure_logging()
    dunnhumby_dir = Path("data/raw/dunnhumby")
    has_real, _ = validate_dunnhumby(dunnhumby_dir)
    dataset = "dunnhumby" if has_real else "synthetic"
    log_event(logger, "full_test_start", dataset=dataset, max_rows=max_rows, evaluation_set=evaluation_set, seed=seed)
    run([sys.executable, "-m", "src.data.prepare", "--dataset", dataset], logger=logger, stage="prepare_dataset")
    run([sys.executable, "-m", "src.training.train_xgboost"], logger=logger, stage="train_xgboost")
    enable_ppo = os.getenv("FULL_TEST_ENABLE_PPO", "1").strip().lower() not in {"0", "false", "no"}
    if enable_ppo:
        run(
            [
                sys.executable,
                "-m",
                "src.rl.train_ppo",
                "--train-path",
                "data/processed/train.csv",
                "--model-path",
                "outputs/models/adaptive_ppo_agent.pt",
                "--timesteps",
                os.getenv("FULL_TEST_PPO_TIMESTEPS", "4000"),
                "--seed",
                str(seed),
                "--horizon",
                os.getenv("FULL_TEST_PPO_HORIZON", "8"),
                "--config",
                "configs/adaptive_hierarchical.yaml",
            ],
            logger=logger,
            stage="train_adaptive_ppo_agent",
        )
    run([
        sys.executable,
        "-m",
        "src.data.coverage",
        "--input",
        "data/processed/test.csv",
        "--output",
        "artifacts/unbiased_eval_set.csv",
        "--summary-output",
        "artifacts/unbiased_summary.json",
        "--target-size",
        "100",
        "--seed",
        str(seed),
    ], logger=logger, stage="build_coverage_set")
    experiment_cmd = [
        sys.executable,
        "-m",
        "src.pipeline.run_experiment",
        "--mode",
        "all",
        "--dataset-mode",
        dataset,
        "--evaluation-set",
        evaluation_set,
    ]
    if max_rows is not None:
        experiment_cmd.extend(["--max-rows", str(max_rows)])
    run(experiment_cmd, logger=logger, stage="run_experiments")
    log_event(logger, "full_test_complete", dataset=dataset, max_rows=max_rows, evaluation_set=evaluation_set, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive agentic marketing main entrypoint")
    parser.add_argument("--full-test", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--full-test-mode", choices=["unbiased", "original", "both"], default=None)
    parser.add_argument("--full-test-original", action="store_true")
    parser.add_argument("--full-test-coverage", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.full_test:
        max_rows = args.max_rows
        if max_rows is None:
            default_limit = os.getenv("FULL_TEST_MAX_ROWS", "all").strip()
            if default_limit and default_limit.lower() != "all":
                max_rows = int(default_limit)
        seed = args.seed
        if seed is None:
            seed = int(os.getenv("FULL_TEST_SEED", "42"))

        if args.full_test_original:
            evaluation_set = "original"
        elif args.full_test_coverage:
            evaluation_set = "unbiased"
        else:
            evaluation_set = args.full_test_mode or os.getenv("FULL_TEST_MODE", "unbiased").strip().lower()
            if evaluation_set not in {"unbiased", "original", "both", "coverage"}:
                raise ValueError("FULL_TEST_MODE must be one of unbiased/original/both")

        full_test(max_rows=max_rows, evaluation_set=evaluation_set, seed=seed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
