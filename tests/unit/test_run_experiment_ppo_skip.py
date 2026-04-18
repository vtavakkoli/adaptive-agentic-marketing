from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

import src.pipeline.run_experiment as run_experiment_module


def test_main_skips_adaptive_ppo_agent_for_mode_all_when_model_missing(tmp_path: Path, monkeypatch) -> None:
    data_path = tmp_path / "unbiased.csv"
    pd.DataFrame(
        [
            {
                "recency_days": 5,
                "frequency_7d": 1,
                "avg_basket_value": 20.0,
                "campaign_touches_30d": 2,
                "prior_response_rate": 0.4,
                "need_score": 0.7,
                "fatigue_score": 0.2,
                "intrusiveness_risk": 0.2,
                "offer_relevance": 0.7,
                "channel": "email",
                "offer_id": "offer_a",
            }
        ]
    ).to_csv(data_path, index=False)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "rules": {},
                "slm": {"enabled": False},
                "guardrails": {},
                "ppo": {"model_path": str(tmp_path / "missing_ppo.pt")},
            }
        ),
        encoding="utf-8",
    )

    captured_modes: list[str] = []

    def _fake_run_experiment(mode: str, test_df, cfg, logger=None, progress_every: int = 1):  # type: ignore[no-untyped-def]
        captured_modes.append(mode)
        return [], {"multiclass": {}, "evaluation_set": "unbiased"}

    monkeypatch.setattr(run_experiment_module, "run_experiment", _fake_run_experiment)
    monkeypatch.setattr(run_experiment_module, "write_reports", lambda *args, **kwargs: None)

    argv = [
        "run_experiment.py",
        "--mode",
        "all",
        "--evaluation-set",
        "unbiased",
        "--coverage-test-path",
        str(data_path),
        "--config",
        str(cfg_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run_experiment_module.main()

    assert "adaptive_ppo_agent" not in captured_modes
    assert "rules_only" in captured_modes
