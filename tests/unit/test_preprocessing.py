from pathlib import Path

import pandas as pd

from src.data.prepare import prepare_dataset


def test_prepare_synthetic(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "proc"
    result = prepare_dataset("synthetic", raw_dir, proc_dir)
    assert Path(result["train"]).exists()
    df = pd.read_csv(result["train"])
    assert "need_score" in df.columns
