from pathlib import Path

from src.evaluation.report import write_reports


def test_write_reports_creates_chart_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "reports"
    metrics = {
        "rules_only__original": {
            "accuracy": 0.8,
            "f1": 0.75,
            "rule_violation_rate": 0.12,
            "fatigue_avoidance_rate": 0.9,
            "latency_per_decision": 0.02,
            "evaluation_set": "original",
            "no_action_distribution": {"pred_no_action_rate": 0.4, "true_no_action_rate": 0.5},
            "multiclass": {"accuracy": 0.62, "macro_f1": 0.55, "weighted_f1": 0.58},
            "action_class_distribution_true": {"do_nothing": 40, "send_information": 30, "recommend_offer_a": 30},
        },
        "rules_only__coverage": {
            "accuracy": 0.7,
            "f1": 0.69,
            "rule_violation_rate": 0.15,
            "fatigue_avoidance_rate": 0.85,
            "latency_per_decision": 0.03,
            "evaluation_set": "coverage",
            "no_action_distribution": {"pred_no_action_rate": 0.55, "true_no_action_rate": 0.6},
            "multiclass": {"accuracy": 0.57, "macro_f1": 0.52, "weighted_f1": 0.54},
            "action_class_distribution_true": {"do_nothing": 60, "send_information": 20, "recommend_offer_a": 20},
        },
    }

    write_reports(
        output_dir=out_dir,
        metrics=metrics,
        dataset_mode="synthetic",
        dataset_summary={"rows": 100},
        feature_summary={"features": ["a", "b"]},
        example_decisions=[{"selected_action": "do_nothing"}],
    )

    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "final_report.html").exists()
    assert (out_dir / "test_set_distribution.png").exists()
    assert (out_dir / "final_results_chart.png").exists()

    html = (out_dir / "final_report.html").read_text(encoding="utf-8")
    assert "Test set bias / coverage analysis" in html
    assert "Final result chart" in html
