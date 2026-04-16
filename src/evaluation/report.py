from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Template

REPORT_TEMPLATE = """
<html><head><title>Adaptive Agentic Marketing Final Report</title></head>
<body>
<h1>An Adaptive Agentic Framework for Responsible Personalized Marketing Using Edge-Deployable SLMs</h1>
<p><b>Timestamp:</b> {{ timestamp }}</p>
<p><b>Dataset mode:</b> {{ dataset_mode }}</p>
<h2>Dataset summary</h2><pre>{{ dataset_summary }}</pre>
<h2>Feature summary</h2><pre>{{ feature_summary }}</pre>
<h2>Architecture overview</h2><p>Rules + XGBoost + SLM (Ollama gemma4:e2b) + explanations + optional content generation.</p>
<h2>Experiment comparison table</h2>{{ experiments_table }}
<h2>Metrics tables</h2><pre>{{ metrics_json }}</pre>
<h2>Class distribution charts</h2><p>See summary.csv and metrics.json for class/action counts.</p>
<h2>Performance comparison charts</h2><p>Tabular comparison provided for reproducibility.</p>
<h2>No-action analysis</h2><p>No-action behavior is explicitly treated as a safe first-class output.</p>
<h2>Ablation results</h2><p>Ablation modes are included in the experiment table.</p>
<h2>Example decisions with explanations</h2><pre>{{ examples }}</pre>
<h2>Ethical/responsible marketing notes</h2><p>The framework enforces fatigue and intrusiveness controls and supports do_nothing as a default-safe action.</p>
<h2>Conclusion</h2><p>Adaptive orchestration can reduce unnecessary contacts while preserving relevance.</p>
</body></html>
"""


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

    pd.DataFrame(metrics).T.reset_index(names="mode").to_csv(summary_path, index=False)

    experiments_table = pd.DataFrame(metrics).T.to_html()
    html = Template(REPORT_TEMPLATE).render(
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_mode=dataset_mode,
        dataset_summary=dataset_summary,
        feature_summary=feature_summary,
        experiments_table=experiments_table,
        metrics_json=json.dumps(metrics, indent=2),
        examples=json.dumps(example_decisions[:5], indent=2),
    )
    html_path.write_text(html, encoding="utf-8")
