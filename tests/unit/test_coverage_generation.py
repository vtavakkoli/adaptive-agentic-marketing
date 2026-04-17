import pandas as pd

from src.data.coverage import CoverageConfig, build_coverage_test_set
from src.data.synthetic import SyntheticConfig, generate_synthetic_dataset
from src.features.label_engineering import derive_labels


def test_build_unbiased_eval_set_has_expected_shape_and_metadata(tmp_path) -> None:
    raw_df = generate_synthetic_dataset(SyntheticConfig(n_rows=600, seed=7), tmp_path)
    labeled_df = derive_labels(raw_df)
    labeled_df = labeled_df.reset_index(drop=True)
    labeled_df["source_case_id"] = [f"case_{idx}" for idx in labeled_df.index]

    eval_df, summary = build_coverage_test_set(labeled_df, CoverageConfig(target_size=100, seed=11, strategy="unbiased"))

    assert len(eval_df) == 100
    assert summary["actual_size"] == 100
    assert summary["seed"] == 11
    assert summary["strategy"] == "unbiased"
    assert {"source_case_id", "sample_type", "edge_case_flag", "case_id"}.issubset(eval_df.columns)
    assert eval_df["action_class"].nunique() >= 3
    assert eval_df["channel"].nunique() >= 2
    assert (eval_df["edge_case_flag"] == 0).all()
