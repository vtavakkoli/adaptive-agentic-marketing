from pathlib import Path

import src.evaluation.report as report_module


def test_warning_box_shows_when_leakage_flag_triggered(tmp_path: Path, monkeypatch) -> None:
    metrics = {
        "adaptive_full_framework__coverage": {
            "accuracy": 1.0,
            "f1": 1.0,
            "rule_violation_rate": 0.0,
            "latency_per_decision": 0.01,
            "evaluation_set": "coverage",
            "no_action_distribution": {"pred_no_action_rate": 0.4, "true_no_action_rate": 0.4},
            "prediction_diversity": {"predicted_action_entropy": 1.0, "distribution_shift_l1": 0.0},
            "multiclass": {
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "weighted_f1": 1.0,
                "balanced_accuracy": 1.0,
                "labels": ["do_nothing", "defer_action", "send_information", "send_reminder"],
                "confusion_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "per_class": {
                    "do_nothing": {"f1": 1.0},
                    "defer_action": {"f1": 1.0},
                    "send_information": {"f1": 1.0},
                    "send_reminder": {"f1": 1.0},
                },
            },
            "action_class_distribution_true": {"do_nothing": 1, "defer_action": 1, "send_information": 1, "send_reminder": 1},
            "action_class_distribution_pred": {"do_nothing": 1, "defer_action": 1, "send_information": 1, "send_reminder": 1},
        }
    }

    monkeypatch.setattr(report_module, "_detect_leakage_risk", lambda: {"leakage_safe": False, "suspicious_llm_keys": ["action_class"]})

    out_dir = tmp_path / "r"
    report_module.write_reports(out_dir, metrics, "synthetic", {"evaluation_set": "coverage"}, {}, [])

    html = (out_dir / "final_report.html").read_text(encoding="utf-8")
    assert "Target leakage risk detected" in html
