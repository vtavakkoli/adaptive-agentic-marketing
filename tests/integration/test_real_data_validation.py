from pathlib import Path

from src.data.prepare import validate_dunnhumby


def test_real_data_validation_missing(tmp_path: Path) -> None:
    valid, missing = validate_dunnhumby(tmp_path)
    assert not valid
    assert len(missing) > 0
