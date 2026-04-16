from pathlib import Path

import pandas as pd

from src.data.prepare import prepare_dataset
from src.pipeline.run_experiment import run_experiment


def test_pipeline_single_mode(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "proc"
    prepare_dataset("synthetic", raw_dir, proc_dir)
    df = pd.read_csv(proc_dir / "test.csv")
    preds, metrics = run_experiment("rules_only", df, {"rules": {}, "slm": {"enabled": False}})
    assert len(preds) == len(df)
    assert "accuracy" in metrics
