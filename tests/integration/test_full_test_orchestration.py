from unittest.mock import patch

from src.main import full_test


def test_full_test_calls_commands() -> None:
    with patch("src.main.run") as run_mock:
        with patch("src.main.validate_dunnhumby", return_value=(False, [])):
            full_test(max_rows=100, evaluation_set="unbiased", seed=42)
    assert run_mock.call_count == 5
    ppo_cmd = run_mock.call_args_list[2].args[0]
    assert "src.rl.train_ppo" in ppo_cmd
    assert "outputs/models/adaptive_ppo_agent.pt" in ppo_cmd
    coverage_cmd = run_mock.call_args_list[3].args[0]
    assert "src.data.coverage" in coverage_cmd
    experiment_cmd = run_mock.call_args_list[-1].args[0]
    assert "--max-rows" in experiment_cmd
    assert "100" in experiment_cmd
    assert "--evaluation-set" in experiment_cmd
    assert "unbiased" in experiment_cmd
