from unittest.mock import patch

from src.main import full_test


def test_full_test_calls_commands() -> None:
    with patch("src.main.run") as run_mock:
        with patch("src.main.validate_dunnhumby", return_value=(False, [])):
            full_test()
    assert run_mock.call_count == 3
