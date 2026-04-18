import pandas as pd

from src.data.coverage import CoverageConfig, build_coverage_test_set
from src.data.synthetic import SyntheticConfig, generate_synthetic_dataset
from src.features.label_engineering import derive_labels


def test_build_unbiased_eval_set_has_expected_shape_and_metadata(tmp_path) -> None:
    raw_df = generate_synthetic_dataset(SyntheticConfig(n_rows=600, seed=7), tmp_path)
    labeled_df = derive_labels(raw_df)
    labeled_df = labeled_df.reset_index(drop=True)
    labeled_df["source_case_id"] = [f"case_{idx}" for idx in labeled_df.index]

    outputs, summary = build_coverage_test_set(
        labeled_df,
        CoverageConfig(diagnostic_target_size=100, diagnostic_per_class=25, seed=11),
    )
    full_df = outputs["full_test_benchmark"]
    eval_df = outputs["diagnostic_balanced_100"]

    assert len(eval_df) == 100
    assert summary["diagnostic_actual_size"] == 100
    assert summary["seed"] == 11
    assert summary["strategy"] == "heldout_primary_plus_balanced_diagnostic"
    assert {"source_case_id", "sample_type", "edge_case_flag", "case_id"}.issubset(eval_df.columns)
    assert len(full_df) == len(labeled_df)
    assert eval_df["action_class"].nunique() >= 3
    assert eval_df["channel"].nunique() >= 2
    assert (eval_df["edge_case_flag"] == 0).all()
    assert summary["audit"]["duplicate_count_diagnostic"] == 0
