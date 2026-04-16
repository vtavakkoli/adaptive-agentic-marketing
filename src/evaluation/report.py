from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

matplotlib.use("Agg")

REPORT_TEMPLATE = """
<html><head><title>Adaptive Agentic Marketing Final Report</title></head>
<body>
<h1>An Adaptive Agentic Framework for Responsible Personalized Marketing Using Edge-Deployable SLMs</h1>
<p><b>Timestamp:</b> {{ timestamp }}</p>
<p><b>Dataset mode:</b> {{ dataset_mode }}</p>
<p><b>Coverage note:</b> The default full-test uses a 100-case coverage-biased evaluation set derived from original data (not purely synthetic).</p>
<h2>Dataset summary</h2><pre>{{ dataset_summary }}</pre>
<h2>Feature summary</h2><pre>{{ feature_summary }}</pre>
<h2>Architecture overview</h2><p>Rules + XGBoost + SLM (Ollama gemma4:e2b) + explanations + optional content generation.</p>
<h2>Experiment comparison table</h2>{{ experiments_table }}
<h2>Metrics tables</h2><pre>{{ metrics_json }}</pre>
<h2>Test set bias / coverage analysis</h2>
<p>{{ bias_assessment }}</p>
{% if test_data_chart %}
<img src="{{ test_data_chart }}" alt="Test-set action distribution chart" style="max-width: 980px;" />
{% endif %}
<h2>Final result chart</h2>
{% if final_result_chart %}
<img src="{{ final_result_chart }}" alt="Final results performance chart" style="max-width: 980px;" />
{% endif %}
<h2>Class distribution charts</h2><p>See summary.csv and metrics.json for no-action distribution, per-action counts, and multiclass confusion matrices.</p>
<h2>No-action analysis</h2><p>No-action behavior is explicitly treated as a safe first-class output.</p>
<h2>Ablation results</h2><p>Ablation modes are included in the experiment table.</p>
<h2>Example decisions with explanations</h2><pre>{{ examples }}</pre>
<h2>Ethical/responsible marketing notes</h2><p>The framework enforces fatigue and intrusiveness controls and supports do_nothing as a default-safe action.</p>
<h2>Conclusion</h2><p>Adaptive orchestration can reduce unnecessary contacts while preserving relevance.</p>
</body></html>
"""


def _build_summary_rows(metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for mode_key, values in metrics.items():
        multi = values.get("multiclass", {})
        no_action = values.get("no_action_distribution", {})
        rows.append(
            {
                "mode": mode_key,
                "evaluation_set": values.get("evaluation_set", "unknown"),
                "binary_accuracy": values.get("accuracy"),
                "binary_f1": values.get("f1"),
                "multiclass_accuracy": multi.get("accuracy"),
                "multiclass_macro_f1": multi.get("macro_f1"),
                "multiclass_weighted_f1": multi.get("weighted_f1"),
                "pred_no_action_rate": no_action.get("pred_no_action_rate"),
                "true_no_action_rate": no_action.get("true_no_action_rate"),
                "rule_violation_rate": values.get("rule_violation_rate"),
                "fatigue_avoidance_rate": values.get("fatigue_avoidance_rate"),
                "latency_per_decision": values.get("latency_per_decision"),
            }
        )
    return pd.DataFrame(rows)


def _normalize(counts: dict[str, int], labels: list[str]) -> np.ndarray:
    vec = np.array([counts.get(label, 0) for label in labels], dtype=float)
    total = vec.sum()
    if total <= 0:
        return np.zeros(len(labels), dtype=float)
    return vec / total


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        if not mask.any():
            return 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _extract_eval_set_distributions(metrics: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_eval: dict[str, dict[str, int]] = {}
    for values in metrics.values():
        eval_set = values.get("evaluation_set", "unknown")
        if eval_set not in by_eval and "action_class_distribution_true" in values:
            by_eval[eval_set] = values["action_class_distribution_true"]
    return by_eval


def _build_bias_assessment(metrics: dict[str, dict[str, Any]]) -> str:
    distributions = _extract_eval_set_distributions(metrics)
    if "coverage" not in distributions:
        return "Coverage set was not included in this run, so dataset-bias comparison is not available."
    if "original" not in distributions:
        return "Only the coverage-biased test set was evaluated in this run (no original-vs-coverage bias comparison)."

    labels = sorted(set(distributions["original"].keys()) | set(distributions["coverage"].keys()))
    original = _normalize(distributions["original"], labels)
    coverage = _normalize(distributions["coverage"], labels)
    jsd = _js_divergence(original, coverage)

    if jsd > 0.10:
        status = "Coverage set is intentionally biased for stress-testing (distribution differs substantially from original)."
    elif jsd > 0.03:
        status = "Coverage set is moderately shifted from original test distribution."
    else:
        status = "Coverage set distribution is close to original (limited bias shift detected)."

    return f"{status} Jensen-Shannon divergence on action distribution: {jsd:.3f}."


def _write_test_data_chart(output_dir: Path, metrics: dict[str, dict[str, Any]]) -> str | None:
    distributions = _extract_eval_set_distributions(metrics)
    if not distributions:
        return None

    eval_sets = sorted(distributions.keys())
    labels = sorted({action for counts in distributions.values() for action in counts.keys()})
    if not labels:
        return None

    x = np.arange(len(labels))
    width = 0.8 / max(len(eval_sets), 1)
    fig, ax = plt.subplots(figsize=(12, 5))

    for idx, eval_set in enumerate(eval_sets):
        normalized = _normalize(distributions[eval_set], labels)
        ax.bar(x + idx * width, normalized, width=width, label=eval_set)

    ax.set_xticks(x + width * (len(eval_sets) - 1) / 2)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Share of action class")
    ax.set_title("Test-set action distribution (bias / coverage view)")
    ax.legend()
    fig.tight_layout()

    out_path = output_dir / "test_set_distribution.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path.name


def _write_final_result_chart(output_dir: Path, summary_df: pd.DataFrame) -> str | None:
    if summary_df.empty:
        return None

    chart_df = summary_df.copy()
    chart_df["mode_label"] = chart_df["mode"].str.replace("__", "\n", regex=False)

    x = np.arange(len(chart_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, chart_df["multiclass_macro_f1"], width=width, label="Macro F1")
    ax.bar(x + width / 2, chart_df["binary_accuracy"], width=width, label="Binary no-action accuracy")

    ax.set_xticks(x)
    ax.set_xticklabels(chart_df["mode_label"], rotation=50, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("Final results by mode/evaluation set")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()

    out_path = output_dir / "final_results_chart.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path.name


def write_reports(
    output_dir: Path,
    metrics: dict[str, dict[str, Any]],
    dataset_mode: str,
    dataset_summary: dict[str, Any],
    feature_summary: dict[str, Any],
    example_decisions: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.csv"
    html_path = output_dir / "final_report.html"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary_df = _build_summary_rows(metrics)
    summary_df.to_csv(summary_path, index=False)

    test_data_chart = _write_test_data_chart(output_dir, metrics)
    final_result_chart = _write_final_result_chart(output_dir, summary_df)
    bias_assessment = _build_bias_assessment(metrics)

    experiments_table = summary_df.to_html(index=False)
    html = Template(REPORT_TEMPLATE).render(
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_mode=dataset_mode,
        dataset_summary=dataset_summary,
        feature_summary=feature_summary,
        experiments_table=experiments_table,
        metrics_json=json.dumps(metrics, indent=2),
        examples=json.dumps(example_decisions[:5], indent=2),
        test_data_chart=test_data_chart,
        final_result_chart=final_result_chart,
        bias_assessment=bias_assessment,
    )
    html_path.write_text(html, encoding="utf-8")
