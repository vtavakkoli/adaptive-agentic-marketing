from pathlib import Path

from src.evaluation.report import write_reports


def test_write_reports_creates_scientific_artifacts_and_sections(tmp_path: Path) -> None:
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
            "prediction_diversity": {"predicted_action_entropy": 1.2, "distribution_shift_l1": 0.2},
            "multiclass": {
                "accuracy": 0.62,
                "macro_f1": 0.55,
                "weighted_f1": 0.58,
                "balanced_accuracy": 0.6,
                "labels": ["do_nothing", "defer_action", "send_information", "send_reminder"],
                "confusion_matrix": [[10, 2, 0, 0], [1, 9, 1, 0], [0, 1, 8, 1], [0, 0, 1, 7]],
                "per_class": {
                    "do_nothing": {"f1": 0.83},
                    "defer_action": {"f1": 0.75},
                    "send_information": {"f1": 0.7},
                    "send_reminder": {"f1": 0.68},
                },
            },
            "action_class_distribution_true": {"do_nothing": 40, "send_information": 30, "send_reminder": 30},
            "action_class_distribution_pred": {"do_nothing": 45, "send_information": 25, "send_reminder": 30},
        },
        "rules_only__coverage": {
            "accuracy": 0.7,
            "f1": 0.69,
            "rule_violation_rate": 0.15,
            "fatigue_avoidance_rate": 0.85,
            "latency_per_decision": 0.03,
            "evaluation_set": "coverage",
            "no_action_distribution": {"pred_no_action_rate": 0.55, "true_no_action_rate": 0.6},
            "prediction_diversity": {"predicted_action_entropy": 1.1, "distribution_shift_l1": 0.25},
            "multiclass": {
                "accuracy": 0.57,
                "macro_f1": 0.52,
                "weighted_f1": 0.54,
                "balanced_accuracy": 0.55,
                "labels": ["do_nothing", "defer_action", "send_information", "send_reminder"],
                "confusion_matrix": [[12, 1, 0, 0], [1, 7, 2, 0], [1, 1, 5, 2], [0, 0, 2, 6]],
                "per_class": {
                    "do_nothing": {"f1": 0.85},
                    "defer_action": {"f1": 0.6},
                    "send_information": {"f1": 0.5},
                    "send_reminder": {"f1": 0.6},
                },
            },
            "action_class_distribution_true": {"do_nothing": 60, "send_information": 20, "send_reminder": 20},
            "action_class_distribution_pred": {"do_nothing": 52, "send_information": 28, "send_reminder": 20},
        },
    }

    write_reports(
        output_dir=out_dir,
        metrics=metrics,
        dataset_mode="synthetic",
        dataset_summary={"evaluation_set": "both", "rows": 100},
        feature_summary={"features": ["a", "b"]},
        example_decisions=[{"selected_action": "do_nothing", "confidence": 0.9, "no_action": True}],
    )

    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "final_report.html").exists()
    assert (out_dir / "grouped_mode_metrics.png").exists()
    assert (out_dir / "data_bias_check.png").exists()
    assert (out_dir / "confusion_matrix_multiclass.png").exists()
    assert (out_dir / "per_class_f1.png").exists()
    assert (out_dir / "latency_by_mode.png").exists()
    assert (out_dir / "validity_flags.json").exists()

    html = (out_dir / "final_report.html").read_text(encoding="utf-8")
    assert "Executive Summary" in html
    assert "Scientific Validity Warnings" in html
    assert "Threats to Validity" in html
    assert "Reproducibility" in html
