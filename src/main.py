from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.data.prepare import validate_dunnhumby


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def full_test() -> None:
    dunnhumby_dir = Path("data/raw/dunnhumby")
    has_real, _ = validate_dunnhumby(dunnhumby_dir)
    dataset = "dunnhumby" if has_real else "synthetic"
    run([sys.executable, "-m", "src.data.prepare", "--dataset", dataset])
    run([sys.executable, "-m", "src.training.train_xgboost"])
    run([sys.executable, "-m", "src.pipeline.run_experiment", "--mode", "all", "--dataset-mode", dataset])


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive agentic marketing main entrypoint")
    parser.add_argument("--full-test", action="store_true")
    args = parser.parse_args()
    if args.full_test:
        full_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
