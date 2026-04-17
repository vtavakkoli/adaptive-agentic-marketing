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
    "adaptive_simple",
    "adaptive_hierarchical",
    "ablation_no_rules",
    "ablation_no_xgboost",
]

REPORT_TEMPLATE = """
<!doctype html><html><head><meta charset="utf-8" /><title>Adaptive Agentic Marketing Final Report</title>
<style>body { font-family: Inter, Segoe UI, Arial, sans-serif; margin: 24px auto; max-width: 1300px; color: #1f2937; line-height: 1.45; }
h1, h2, h3 { color: #111827; margin-bottom: 0.4rem; }.muted { color: #4b5563; } .warn { border-left: 5px solid #dc2626; background: #fef2f2; padding: 10px; margin: 8px 0; border-radius: 8px; }
.ok { border-left: 5px solid #16a34a; background: #f0fdf4; padding: 10px; margin: 8px 0; border-radius: 8px; } section { margin: 20px 0; } table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { border: 1px solid #e5e7eb; padding: 8px; text-align: left; } th { background: #f3f4f6; } .fig img { max-width: 100%; } .badge { display:inline-block;background:#e5e7eb;padding:3px 8px;border-radius:999px;margin-right:6px;font-size:12px; }
</style></head><body>
<h1>Adaptive Agentic Marketing Evaluation Report</h1>
<p class="muted"><b>Timestamp:</b> {{ timestamp }} | <b>Dataset mode:</b> {{ dataset_mode }} | <b>Evaluation set:</b> {{ evaluation_set }} | <b>Version:</b> {{ version_info }}</p>
<section><h2>Executive Summary</h2><p>This report compares adaptive_simple (legacy flat baseline) and adaptive_hierarchical (new staged framework) with calibration, guardrails, and fallback diagnostics.</p></section>
<section><h2>Architecture</h2><ul>
<li><b>adaptive_simple</b> = previous flat <code>adaptive_full</code> baseline (migration alias retained internally only).</li>
<li><b>adaptive_hierarchical</b> = hierarchical Stage A (do_nothing vs action) + Stage B (defer/send_info/send_reminder) with uncertainty-aware fallback.</li>
<li>Includes class-sensitive policy costs, calibrated confidence usage, and hard policy guardrails before final action emission.</li>
</ul></section>
<section><h2>Scientific Validity Warnings</h2>{% if warnings %}{% for w in warnings %}<div class="warn">{{ w }}</div>{% endfor %}{% else %}<div class="ok">No critical validity flags detected in this run.</div>{% endif %}</section>
<section><h2>Main Result Table</h2>{{ experiments_table }}</section>
<section><h2>Diagnostics</h2>{% for fig in figures %}<div class="fig"><img src="{{ fig.file }}" alt="{{ fig.caption }}" /><div>{{ fig.caption }}</div></div>{% endfor %}</section>
<section><h2>Example Decisions</h2>{% for ex in examples %}<div><span class="badge">true={{ ex.true_label }}</span><span class="badge">pred={{ ex.selected_action }}</span><span class="badge">conf={{ ex.confidence }}</span><span class="badge">stageA={{ ex.stage_a }}</span><span class="badge">stageB={{ ex.stage_b }}</span><span class="badge">guardrail={{ ex.guardrail }}</span><span class="badge">fallback={{ ex.fallback }}</span><div><b>top features:</b> {{ ex.top_features }}</div><div><b>explanation:</b> {{ ex.explanation }}</div></div><hr/>{% endfor %}</section>
<section><h2>Threats to Validity</h2><ul><li>Proxy label risk may inflate alignment with policy heuristics over true outcomes.</li><li>Uncertainty threshold sensitivity can materially change abstention and defer rates.</li><li>Class definition ambiguity (especially reminder prerequisites) can cap achievable macro F1.</li><li>Distribution shift between offline data and deployment traffic may degrade calibration.</li></ul></section>
<section><h2>Reproducibility</h2><p><code>{{ reproducibility.command }}</code></p><pre>{{ reproducibility.details }}</pre></section>
</body></html>
"""


def _short_mode(mode: str) -> str:
    if mode in {"adaptive_full", "adaptive_full_framework"}:
        return "adaptive_simple"
    return mode


def _mode_name(metric_key: str) -> str:
    return _short_mode(metric_key.split("__", 1)[0])


def _build_summary_rows(metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for mode_key, values in metrics.items():
        mode_name = _mode_name(mode_key)
        multi = values.get("multiclass", {})
        rows.append(
            {
                "mode": mode_name,
                "macro_f1": multi.get("macro_f1", 0.0),
                "weighted_f1": multi.get("weighted_f1", 0.0),
                "multiclass_accuracy": multi.get("accuracy", 0.0),
                "balanced_accuracy": multi.get("balanced_accuracy", 0.0),
                "stageA_binary_f1": values.get("stageA_binary_f1", 0.0),
                "action_only_macro_f1": values.get("action_only_macro_f1", 0.0),
                "ece": values.get("ece", 0.0),
                "brier": values.get("brier", 0.0),
                "abstention_rate": values.get("abstention_rate", 0.0),
                "guardrail_override_rate": values.get("guardrail_override_rate", 0.0),
                "latency_total_s": values.get("latency_total_s", 0.0),
                "evaluation_set": values.get("evaluation_set", "unknown"),
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
    return {"leakage_safe": len(suspicious) == 0, "suspicious_llm_keys": suspicious}


def _plot_grouped_mode_metrics(output_dir: Path, df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    grouped = df.groupby("mode", as_index=False, observed=False)[["macro_f1", "multiclass_accuracy", "balanced_accuracy"]].mean()
    x = np.arange(len(grouped)); w = 0.25
    fig, ax = plt.subplots(figsize=(10,4.5))
    for i,col in enumerate(["macro_f1","multiclass_accuracy","balanced_accuracy"]):
        ax.bar(x+(i-1)*w, grouped[col], width=w, label=col)
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in grouped["mode"]], rotation=20)
    ax.legend(); fig.tight_layout()
    out = output_dir / "grouped_mode_metrics.png"; fig.savefig(out, dpi=220); plt.close(fig)
    return out.name

def _build_warnings(metrics: dict[str, dict[str, Any]], summary_df: pd.DataFrame, leakage_status: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not leakage_status.get("leakage_safe", True):
        warnings.append(f"Target leakage risk detected in LLM input keys: {leakage_status.get('suspicious_llm_keys', [])}")
    if summary_df.empty:
        warnings.append("No summary rows were generated.")
        return warnings
    if (summary_df["ece"] > 0.2).any():
        warnings.append("Calibration ECE exceeds 0.2 for at least one mode.")
    if (summary_df["abstention_rate"] > 0.7).any():
        warnings.append("High abstention rate indicates potential over-conservative fallback thresholds.")
    actions = {a for v in metrics.values() for a in v.get("multiclass", {}).get("labels", [])}
    if "send_reminder" not in actions or "defer_action" not in actions:
        warnings.append("Some action classes are absent in predictions; confusion diagonality claims are limited.")
    warnings.append("adaptive_simple is the renamed legacy flat baseline; adaptive_hierarchical is the new primary framework.")
    return warnings


def _plot_confusion_matrix(output_dir: Path, metrics: dict[str, dict[str, Any]]) -> str | None:
    chosen = None
    for key in ["adaptive_hierarchical__coverage", "adaptive_hierarchical__original", "adaptive_simple__coverage"]:
        if key in metrics:
            chosen = metrics[key]
            break
    if chosen is None and metrics:
        chosen = next(iter(metrics.values()))
    if not chosen:
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
    ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout()
    out = output_dir / "confusion_matrix_multiclass.png"; fig.savefig(out, dpi=220); plt.close(fig)
    return out.name


def _plot_per_class_f1(output_dir: Path, metrics: dict[str, dict[str, Any]]) -> str | None:
    chosen = next(iter(metrics.values())) if metrics else {}
    per_class = chosen.get("multiclass", {}).get("per_class", {})
    if not per_class:
        return None
    labels = list(per_class.keys()); vals = [per_class[l]["f1"] for l in labels]
    fig, ax = plt.subplots(figsize=(9, 4.5)); bars = ax.bar(labels, vals, color=PALETTE[2]); ax.set_ylim(0, 1)
    for b in bars: ax.text(b.get_x() + b.get_width()/2, b.get_height()+0.01, f"{b.get_height():.2f}", ha="center", fontsize=8)
    fig.tight_layout(); out = output_dir / "per_class_f1.png"; fig.savefig(out, dpi=220); plt.close(fig)
    return out.name


def _plot_latency_by_mode(output_dir: Path, df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    grouped = df.groupby("mode", as_index=False, observed=False)["latency_total_s"].mean()
    fig, ax = plt.subplots(figsize=(9, 4.5)); ax.bar([str(v) for v in grouped["mode"]], grouped["latency_total_s"], color=PALETTE[3])
    ax.set_ylabel("Seconds total"); fig.tight_layout(); out = output_dir / "latency_by_mode.png"; fig.savefig(out, dpi=220); plt.close(fig)
    return out.name


def _plot_reliability_diagram(output_dir: Path, metrics: dict[str, dict[str, Any]]) -> str | None:
    if not metrics:
        return None
    df = _build_summary_rows(metrics)
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(df["ece"], df["brier"], color=PALETTE[0])
    for _, row in df.iterrows():
        ax.text(float(row["ece"]) + 0.002, float(row["brier"]) + 0.002, str(row["mode"]), fontsize=8)
    ax.set_xlabel("ECE"); ax.set_ylabel("Brier"); ax.set_title("Calibration diagnostics by mode")
    fig.tight_layout(); out = output_dir / "reliability_diagram.png"; fig.savefig(out, dpi=220); plt.close(fig)
    return out.name


def _format_examples(example_decisions: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    out = []
    for ex in example_decisions[:limit]:
        out.append(
            {
                "true_label": ex.get("true_label", "unknown"),
                "selected_action": ex.get("selected_action", "unknown"),
                "confidence": f"{float(ex.get('confidence', 0.0)):.3f}",
                "stage_a": f"{ex.get('stage_a_prediction', 'n/a')} ({float(ex.get('stage_a_probability', 0.0)):.2f})",
                "stage_b": f"{ex.get('stage_b_prediction', 'n/a')} ({float(ex.get('stage_b_probability', 0.0) or 0.0):.2f})",
                "guardrail": ex.get("guardrail_overrode", False),
                "fallback": ex.get("fallback_reason") or "none",
                "top_features": ", ".join(ex.get("top_features", [])[:4]),
                "explanation": ex.get("explanation", ""),
            }
        )
    return out


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

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "validity_flags.json").write_text(json.dumps({"leakage": leakage_status, "warnings": warnings}, indent=2), encoding="utf-8")

    figs = []
    for fn, cap in [
        (_plot_grouped_mode_metrics(output_dir, summary_df), "Grouped mode metrics."),
        (_plot_confusion_matrix(output_dir, metrics), "Normalized confusion matrix (counts + row normalized)."),
        (_plot_per_class_f1(output_dir, metrics), "Per-class precision/recall/F1 diagnostic focus chart."),
        (_plot_reliability_diagram(output_dir, metrics), "Calibration diagnostics (ECE vs Brier)."),
        (_plot_latency_by_mode(output_dir, summary_df), "Latency breakdown by mode."),
    ]:
        if fn:
            figs.append({"file": fn, "caption": cap})

    display_df = summary_df.copy()
    experiments_table = display_df.to_html(index=False, escape=False)
    html = Template(REPORT_TEMPLATE).render(
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_mode=dataset_mode,
        evaluation_set=dataset_summary.get("evaluation_set", "unknown"),
        version_info="local",
        warnings=warnings,
        experiments_table=experiments_table,
        figures=figs,
        examples=_format_examples(example_decisions),
        reproducibility={
            "command": "python -m src.pipeline.run_experiment --mode adaptive_hierarchical --config configs/adaptive_hierarchical.yaml --seeds 3 --report --calibrate",
            "details": json.dumps(
                {
                    "dataset_summary": dataset_summary,
                    "feature_summary": feature_summary,
                    "seed": dataset_summary.get("seed", "not_provided"),
                    "config": dataset_summary.get("config_path", "unknown"),
                    "git_commit_hash": dataset_summary.get("git_commit_hash", "not_available"),
                },
                indent=2,
            ),
        },
    )
    clean_html = html.replace("file:///", "").replace("NaN", "")
    (output_dir / "final_report.html").write_text(clean_html, encoding="utf-8")
