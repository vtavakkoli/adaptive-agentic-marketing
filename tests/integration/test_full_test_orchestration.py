from unittest.mock import patch

from src.main import full_test


def test_full_test_calls_commands() -> None:
    with patch("src.main.run") as run_mock:
        with patch("src.main.validate_dunnhumby", return_value=(False, [])):
            full_test(max_rows=10)
    assert run_mock.call_count == 3
    experiment_cmd = run_mock.call_args_list[-1].args[0]
    assert "--max-rows" in experiment_cmd
    assert "10" in experiment_cmd
