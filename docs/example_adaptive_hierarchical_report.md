# Example Final Report Snapshot (adaptive_hierarchical)

This repository now generates `outputs/reports/final_report.html` with:

- Architecture section documenting `adaptive_simple` rename and `adaptive_hierarchical` framework.
- Main results table with macro F1, balanced accuracy, stage-A binary F1, action-only macro F1, ECE, Brier, abstention, guardrail overrides, and latency.
- Diagnostic figures (confusion matrix, per-class F1, calibration scatter/reliability proxy, latency chart).
- Example decision traces including stage outputs, fallback reasons, and top feature contributors.

Run command:

```bash
python -m src.pipeline.run_experiment \
  --mode adaptive_hierarchical \
  --config configs/adaptive_hierarchical.yaml \
  --seeds 3 \
  --report \
  --calibrate
```
