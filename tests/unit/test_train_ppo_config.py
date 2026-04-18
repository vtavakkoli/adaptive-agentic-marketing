from src.rl.train_ppo import _build_ppo_config


def test_build_ppo_config_cli_seed_and_horizon_override_yaml() -> None:
    cfg = _build_ppo_config(
        ppo_cfg={"seed": 11, "horizon": 99, "gamma": 0.9, "rollout_steps": 32},
        cli_seed=42,
        cli_horizon=8,
    )
    assert cfg.seed == 42
    assert cfg.horizon == 8
    assert cfg.gamma == 0.9
    assert cfg.rollout_steps == 32
