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

PALETTE = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#D55E00"]
MODE_ORDER = [
    "rules_only",
    "xgboost_only",
    "slm_only",
    "adaptive_full_framework",
    "ablation_no_rules",
    "ablation_no_xgboost",
]

REPORT_TEMPLATE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Adaptive Agentic Marketing Final Report</title>
<style>
body { font-family: Inter, Segoe UI, Arial, sans-serif; margin: 24px auto; max-width: 1200px; color: #1f2937; line-height: 1.45; }
h1, h2, h3 { color: #111827; margin-bottom: 0.4rem; }
.muted { color: #4b5563; }
.grid { display: grid; gap: 12px; }
.cards { grid-template-columns: repeat(3, minmax(220px, 1fr)); }
.card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; }
.kpi { font-size: 1.5rem; font-weight: 700; color: #111827; }
.warn { border-left: 5px solid #dc2626; background: #fef2f2; padding: 12px; margin: 10px 0; border-radius: 8px; }
.ok { border-left: 5px solid #16a34a; background: #f0fdf4; padding: 12px; margin: 10px 0; border-radius: 8px; }
section { margin: 24px 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.92rem; }
th, td { border: 1px solid #e5e7eb; padding: 8px; text-align: left; }
th { background: #f3f4f6; }
.best { font-weight: 700; color: #065f46; }
.fig { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 8px; margin-bottom: 12px; }
.fig img { max-width: 100%; height: auto; }
.caption { font-size: 0.88rem; color: #374151; margin: 6px 4px; }
.example { border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; background: #fcfcfd; margin: 8px 0; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #e5e7eb; font-size: 0.8rem; margin-right: 8px; }
code, pre { background: #f3f4f6; padding: 3px 6px; border-radius: 6px; }
</style>
</head>
<body>
<h1>Adaptive Agentic Marketing Evaluation Report</h1>
<p class="muted"><b>Timestamp:</b> {{ timestamp }} | <b>Dataset mode:</b> {{ dataset_mode }} | <b>Evaluation set:</b> {{ evaluation_set }} | <b>Model:</b> {{ model_name }} | <b>Version:</b> {{ version_info }}</p>

<section>
<h2>Executive Summary</h2>
<div class="grid cards">
{% for c in summary_cards %}
  <div class="card"><div class="muted">{{ c.title }}</div><div class="kpi">{{ c.value }}</div></div>
{% endfor %}
</div>
</section>

<section>
<h2>Scientific Validity Warnings</h2>
{% if warnings %}
  {% for w in warnings %}<div class="warn">{{ w }}</div>{% endfor %}
{% else %}
  <div class="ok">No critical validity flags detected in this run.</div>
{% endif %}
</section>

<section>
<h2>Experiment Comparison</h2>
{{ experiments_table }}
</section>

<section>
<h2>Main Charts</h2>
{% for fig in figures %}
<div class="fig"><img src="{{ fig.file }}" alt="{{ fig.caption }}" /><div class="caption">{{ fig.caption }}</div></div>
{% endfor %}
</section>

<section>
<h2>Example Decisions</h2>
{% for ex in examples %}
<div class="example">
  <span class="badge">action={{ ex.selected_action }}</span>
  <span class="badge">confidence={{ ex.confidence }}</span>
  <span class="badge">no_action={{ ex.no_action }}</span>
  <span class="badge">rule_forced={{ ex.rule_forced }}</span>
  <div><b>Explanation:</b> {{ ex.explanation }}</div>
  <table>
    <tr><th>Score</th><th>Value</th></tr>
    {% for s in ex.scores %}<tr><td>{{ s.name }}</td><td>{{ s.value }}</td></tr>{% endfor %}
  </table>
</div>
{% endfor %}
</section>

<section>
<h2>Threats to Validity</h2>
<ul>
<li>Labels may be proxy labels from rules/heuristics rather than true customer outcomes.</li>
<li>Synthetic or coverage-biased evaluation may not perfectly represent deployment populations.</li>
<li>Leakage-sensitive fields must be excluded from LLM inputs for trustworthy adaptive claims.</li>
<li>Perfect or near-perfect scores should be treated cautiously and audited for data/label artifacts.</li>
</ul>
</section>

<section>
<h2>Reproducibility</h2>
<p><b>Command:</b> <code>{{ reproducibility.command }}</code></p>
<pre>{{ reproducibility.details }}</pre>
<p><b>Outputs:</b> metrics.json, summary.csv, figures/*.png, final_report.html, validity_flags.json</p>
</section>

<section>
<h2>Conclusion</h2>
<p>{{ conclusion }}</p>
</section>
</body>
</html>
"""


def _short_mode(mode: str) -> str:
    return {
        "adaptive_full_framework": "adaptive_full",
        "ablation_no_rules": "no_rules",
        "ablation_no_xgboost": "no_xgb",
    }.get(mode, mode)


def _mode_name(metric_key: str) -> str:
    return metric_key.split("__", 1)[0]


def _build_summary_rows(metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for mode_key, values in metrics.items():
        mode_name = _mode_name(mode_key)
        multi = values.get("multiclass", {})
        per_class = multi.get("per_class", {})
        no_action = values.get("no_action_distribution", {})
        rows.append(
            {
                "mode": mode_name,
                "evaluation_set": values.get("evaluation_set", "unknown"),
                "macro_f1": multi.get("macro_f1", 0.0),
                "weighted_f1": multi.get("weighted_f1", 0.0),
                "multiclass_accuracy": multi.get("accuracy", 0.0),
                "balanced_accuracy": multi.get("balanced_accuracy", 0.0),
                "binary_no_action_accuracy": values.get("accuracy", 0.0),
                "rule_violation_rate": values.get("rule_violation_rate", 0.0),
                "latency_per_decision": values.get("latency_per_decision", 0.0),
                "pred_no_action_rate": no_action.get("pred_no_action_rate", 0.0),
                "true_no_action_rate": no_action.get("true_no_action_rate", 0.0),
                "predicted_action_entropy": values.get("prediction_diversity", {}).get("predicted_action_entropy", 0.0),
                "distribution_shift_l1": values.get("prediction_diversity", {}).get("distribution_shift_l1", 0.0),
                "send_reminder_f1": per_class.get("send_reminder", {}).get("f1", 0.0),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["mode"] = pd.Categorical(df["mode"], categories=MODE_ORDER, ordered=True)
        df = df.sort_values(["evaluation_set", "mode"], na_position="last").reset_index(drop=True)
    return df


def _detect_leakage_risk() -> dict[str, Any]:
    from src.agentic.controller import LEAKAGE_BLOCKLIST, LLM_ALLOWED_FEATURE_KEYS

    suspicious = [k for k in LLM_ALLOWED_FEATURE_KEYS if any(token in k.lower() for token in LEAKAGE_BLOCKLIST)]
    leakage_safe = len(suspicious) == 0
    return {"leakage_safe": leakage_safe, "suspicious_llm_keys": suspicious}


def _extract_eval_set_distributions(metrics: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_eval: dict[str, dict[str, int]] = {}
    for values in metrics.values():
        eval_set = values.get("evaluation_set", "unknown")
        if eval_set not in by_eval and "action_class_distribution_true" in values:
            by_eval[eval_set] = values["action_class_distribution_true"]
    return by_eval


def _normalize(counts: dict[str, int], labels: list[str]) -> np.ndarray:
    vec = np.array([counts.get(label, 0) for label in labels], dtype=float)
    total = vec.sum()
    return vec / total if total > 0 else np.zeros(len(labels), dtype=float)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask]))) if mask.any() else 0.0

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _build_warnings(metrics: dict[str, dict[str, Any]], summary_df: pd.DataFrame, leakage_status: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not leakage_status["leakage_safe"]:
        warnings.append(f"Target leakage risk detected in LLM input keys: {leakage_status['suspicious_llm_keys']}")
    if summary_df["macro_f1"].max() >= 0.999:
        warnings.append("Perfect metric detected. Results may be inflated; audit leakage and label generation.")
    eval_sets = set(summary_df.get("evaluation_set", [])) if not summary_df.empty else set()
    if eval_sets == {"coverage"}:
        warnings.append("Only coverage set evaluated; no original-vs-coverage comparison available.")
    if len(eval_sets) <= 1:
        warnings.append("Only one evaluation split present.")
    all_actions = {a for values in metrics.values() for a in values.get("multiclass", {}).get("labels", [])}
    expected_actions = {"do_nothing", "defer_action", "send_information", "send_reminder"}
    if all_actions and all_actions != expected_actions:
        warnings.append(f"Action-space mismatch: observed={sorted(all_actions)}, expected={sorted(expected_actions)}")
    warnings.append("Labels appear proxy/derived; external outcome validation is still required.")
    return warnings


def _plot_grouped_metrics(output_dir: Path, df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    grouped = df.groupby("mode", as_index=False, observed=False)[["macro_f1", "multiclass_accuracy", "binary_no_action_accuracy"]].mean()
    x = np.arange(len(grouped))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, col in enumerate(["macro_f1", "multiclass_accuracy", "binary_no_action_accuracy"]):
        bars = ax.bar(x + (idx - 1) * width, grouped[col], width, label=col.replace("_", " "), color=PALETTE[idx])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_mode(str(m)) for m in grouped["mode"]], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score", fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    out = output_dir / "grouped_mode_metrics.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out.name


def _plot_confusion_matrix(output_dir: Path, metrics: dict[str, dict[str, Any]]) -> str | None:
    if "adaptive_full_framework__coverage" in metrics:
        chosen = metrics["adaptive_full_framework__coverage"]
    elif "adaptive_full_framework__original" in metrics:
        chosen = metrics["adaptive_full_framework__original"]
    elif metrics:
        chosen = next(iter(metrics.values()))
    else:
        return None
    multi = chosen.get("multiclass", {})
    labels = multi.get("labels", [])
    cm = np.array(multi.get("confusion_matrix", []), dtype=float)
    if cm.size == 0:
        return None
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i,j])}\n({norm[i,j]:.2f})", ha="center", va="center", fontsize=8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = output_dir / "confusion_matrix_multiclass.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out.name


def _plot_class_distribution(output_dir: Path, metrics: dict[str, dict[str, Any]]) -> str | None:
    chosen = next(iter(metrics.values())) if metrics else {}
    true_counts = chosen.get("action_class_distribution_true", {})
    pred_counts = chosen.get("action_class_distribution_pred", {})
    labels = sorted(set(true_counts) | set(pred_counts))
    if not labels:
        return None
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, [true_counts.get(l, 0) for l in labels], width, label="true", color=PALETTE[0])
    bars2 = ax.bar(x + width / 2, [pred_counts.get(l, 0) for l in labels], width, label="pred", color=PALETTE[1])
    for bars in [bars1, bars2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"{int(b.get_height())}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend()
    fig.tight_layout()
    out = output_dir / "class_distribution_true_vs_pred.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out.name


def _plot_per_class_f1(output_dir: Path, metrics: dict[str, dict[str, Any]]) -> str | None:
    chosen = next(iter(metrics.values())) if metrics else {}
    per_class = chosen.get("multiclass", {}).get("per_class", {})
    if not per_class:
        return None
    labels = list(per_class.keys())
    vals = [per_class[l]["f1"] for l in labels]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, vals, color=PALETTE[2])
    ax.set_ylim(0, 1)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    out = output_dir / "per_class_f1.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out.name


def _plot_latency_by_mode(output_dir: Path, df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    grouped = df.groupby("mode", as_index=False, observed=False)["latency_per_decision"].mean()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar([_short_mode(str(v)) for v in grouped["mode"]], grouped["latency_per_decision"], color=PALETTE[3])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0005, f"{b.get_height():.3f}", ha="center", fontsize=8)
    ax.set_ylabel("Seconds/decision")
    fig.tight_layout()
    out = output_dir / "latency_by_mode.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out.name


def _plot_no_action_rates(output_dir: Path, df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    grouped = df.groupby("mode", as_index=False, observed=False)[["true_no_action_rate", "pred_no_action_rate"]].mean()
    x = np.arange(len(grouped))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - width / 2, grouped["true_no_action_rate"], width, label="true", color=PALETTE[4])
    ax.bar(x + width / 2, grouped["pred_no_action_rate"], width, label="pred", color=PALETTE[5])
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_mode(str(v)) for v in grouped["mode"]])
    ax.legend()
    fig.tight_layout()
    out = output_dir / "no_action_rate_comparison.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out.name


def _plot_distribution_shift(output_dir: Path, metrics: dict[str, dict[str, Any]]) -> tuple[str | None, str]:
    distributions = _extract_eval_set_distributions(metrics)
    if "original" not in distributions or "coverage" not in distributions:
        return None, "Original-vs-coverage distribution shift chart unavailable."
    labels = sorted(set(distributions["original"]) | set(distributions["coverage"]))
    orig = _normalize(distributions["original"], labels)
    cov = _normalize(distributions["coverage"], labels)
    jsd = _js_divergence(orig, cov)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width / 2, orig, width, label="original", color=PALETTE[0])
    ax.bar(x + width / 2, cov, width, label="coverage", color=PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    out = output_dir / "distribution_shift_original_vs_coverage.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out.name, f"JS divergence between original and coverage label distributions: {jsd:.3f}."


def _format_examples(example_decisions: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    formatted = []
    for ex in example_decisions[:limit]:
        scores = ex.get("supporting_scores", {})
        top_scores = [
            {"name": k, "value": f"{float(v):.3f}"}
            for k, v in list(scores.items())[:5]
        ]
        formatted.append(
            {
                "selected_action": ex.get("selected_action", "unknown"),
                "confidence": f"{float(ex.get('confidence', 0.0)):.3f}",
                "no_action": ex.get("no_action", False),
                "rule_forced": ex.get("rule_forced", False),
                "explanation": ex.get("explanation", ""),
                "scores": top_scores,
            }
        )
    return formatted


def _best_class(value: float, series: pd.Series) -> str:
    return "best" if value == series.max() else ""


def write_reports(
    output_dir: Path,
    metrics: dict[str, dict[str, Any]],
    dataset_mode: str,
    dataset_summary: dict[str, Any],
    feature_summary: dict[str, Any],
    example_decisions: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = _build_summary_rows(metrics)
    leakage_status = _detect_leakage_risk()
    warnings = _build_warnings(metrics, summary_df, leakage_status)

    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.csv"
    validity_flags_path = output_dir / "validity_flags.json"
    html_path = output_dir / "final_report.html"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary_df.to_csv(summary_path, index=False)
    validity_flags_path.write_text(
        json.dumps({"leakage": leakage_status, "warnings": warnings}, indent=2),
        encoding="utf-8",
    )

    figures: list[dict[str, str]] = []
    chart_calls = [
        (_plot_grouped_metrics(output_dir, summary_df), "Grouped bars: macro F1, multiclass accuracy, and binary no-action accuracy across modes."),
        (_plot_confusion_matrix(output_dir, metrics), "Multiclass confusion matrix with absolute counts and normalized row percentages."),
        (_plot_class_distribution(output_dir, metrics), "True vs predicted class distributions for action labels."),
        (_plot_per_class_f1(output_dir, metrics), "Per-class F1 scores for the selected evaluation slice."),
        (_plot_latency_by_mode(output_dir, summary_df), "Latency per decision by mode (seconds)."),
        (_plot_no_action_rates(output_dir, summary_df), "True vs predicted no_action rates by mode."),
    ]
    for fname, caption in chart_calls:
        if fname:
            figures.append({"file": fname, "caption": caption})
    dist_chart, dist_note = _plot_distribution_shift(output_dir, metrics)
    if dist_chart:
        figures.append({"file": dist_chart, "caption": f"Distribution shift view. {dist_note}"})

    display_df = summary_df.copy()
    if not display_df.empty:
        for col in ["macro_f1", "multiclass_accuracy", "binary_no_action_accuracy"]:
            display_df[col] = display_df[col].map(lambda v: f'<span class="{_best_class(float(v), summary_df[col])}">{float(v):.3f}</span>')
        display_df["mode"] = display_df["mode"].map(_short_mode)
    experiments_table = display_df.to_html(index=False, escape=False)

    primary = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    summary_cards = [
        {"title": "Macro F1", "value": f"{primary.get('macro_f1', 0.0):.3f}"},
        {"title": "Multiclass accuracy", "value": f"{primary.get('multiclass_accuracy', 0.0):.3f}"},
        {"title": "Binary no-action accuracy", "value": f"{primary.get('binary_no_action_accuracy', 0.0):.3f}"},
        {"title": "Rule violation rate", "value": f"{primary.get('rule_violation_rate', 0.0):.3f}"},
        {"title": "Latency / decision (s)", "value": f"{primary.get('latency_per_decision', 0.0):.4f}"},
        {"title": "Leakage status", "value": "PASS" if leakage_status["leakage_safe"] else "FAIL"},
    ]

    conclusion = (
        "Leakage-sensitive fields were excluded and metrics indicate promising performance, but deployment conclusions remain tentative until validated on real-world outcomes."
        if leakage_status["leakage_safe"] and primary.get("macro_f1", 0) < 0.999
        else "Results are preliminary and likely inflated; leakage/label artifacts must be resolved before making adaptive-performance claims."
    )

    html = Template(REPORT_TEMPLATE).render(
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_mode=dataset_mode,
        evaluation_set=dataset_summary.get("evaluation_set", "unknown"),
        model_name="xgboost(binary no_action) + SLM controller",
        version_info="local",
        summary_cards=summary_cards,
        warnings=warnings,
        experiments_table=experiments_table,
        figures=figures,
        examples=_format_examples(example_decisions),
        reproducibility={
            "command": "python -m src.main --full-test",
            "details": json.dumps(
                {
                    "dataset_summary": dataset_summary,
                    "feature_summary": feature_summary,
                    "seed": dataset_summary.get("seed", "not_provided"),
                    "distribution_note": dist_note,
                },
                indent=2,
            ),
        },
        conclusion=conclusion,
    )
    html_path.write_text(html, encoding="utf-8")
