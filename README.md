# adaptive-agentic-marketing

Research-oriented Python application for: **"An Adaptive Agentic Framework for Responsible Personalized Marketing Using Edge-Deployable"**.

## Motivation
This framework is designed to infer likely customer needs and reduce intrusive or unnecessary outreach in personalized marketing. The decision system explicitly supports **`do_nothing`** as a first-class safe action.

## Stack
- Python 3.12
- Docker / Docker Compose
- Ollama + `gemma4:e2b`
- XGBoost
- FastAPI

## Repository Structure
- `src/`
- `tests/`
- `configs/`
- `data/`
- `scripts/`
- `notebooks/`
- `outputs/`
- `docs/`

## Dunnhumby setup
Place raw files in:
- `data/raw/dunnhumby/transaction_data.csv`
- `data/raw/dunnhumby/hh_demographic.csv`
- `data/raw/dunnhumby/product.csv`
- `data/raw/dunnhumby/campaign_desc.csv`
- `data/raw/dunnhumby/coupon.csv`
- `data/raw/dunnhumby/coupon_redempt.csv`
- `data/raw/dunnhumby/causal_data.csv`

If missing, the system automatically falls back to synthetic mode.

## Start Ollama and model (host)
Run Ollama on your host machine and pull the model:
```bash
ollama serve
ollama pull gemma4:e2b
```
Docker services connect to host Ollama via `http://host.docker.internal:11434`.

## Environment variables (.env optional)
Docker Compose now uses safe defaults, so a `.env` file is **not required**.  
If you want custom values, copy `.env.example` to `.env` and edit:
```bash
cp .env.example .env
```
On Windows PowerShell:
```powershell
Copy-Item .env.example .env
```

## Build
```bash
docker compose build
```

## Data preparation
Docker Compose:
```bash
docker compose run --rm app python -m src.data.prepare --dataset dunnhumby
docker compose run --rm app python -m src.data.prepare --dataset synthetic
```

Equivalent local Python:
```bash
python -m src.data.prepare --dataset dunnhumby
python -m src.data.prepare --dataset synthetic
```

## Train and experiments
Docker Compose:
```bash
docker compose run --rm app python -m src.training.train_xgboost
docker compose run --rm app python -m src.pipeline.run_experiment --mode adaptive_framework
docker compose run --rm run_experiment
```

Equivalent local Python:
```bash
python -m src.training.train_xgboost
python -m src.pipeline.run_experiment --mode adaptive_framework
```

`run_experiment` is a convenience Docker Compose service that assumes prepared data already exists, then:
1. runs `adaptive_framework` on the unbiased evaluation set, and
2. generates experiment reports in `outputs/reports/`.

## Full end-to-end test
```bash
docker compose run --rm full-test
```
This verifies environment assumptions, picks real-data mode when available (otherwise synthetic mode), preprocesses data, trains XGBoost, builds a **100-case unbiased evaluation set derived from the original test split**, runs all experiment modes, calls the host Ollama SLM endpoint, computes metrics, and generates reports.

### What changed
- `full-test` now defaults to `FULL_TEST_MODE=unbiased`, which runs the **unbiased 100-case** evaluation set (`artifacts/unbiased_eval_set.csv`).
- Unbiased set generation is reproducible with `FULL_TEST_SEED` (default `42`).
- Original test-set evaluation is still available.

### Unbiased 100-case set generation
The unbiased set is sampled from the original labeled test data (`data/processed/test.csv`) using stratified random sampling over `action_class` and `channel` to preserve distributional realism while avoiding augmentation bias.

Each generated row includes:
- `source_case_id`
- `sample_type` (`unbiased_stratified`)
- `edge_case_flag` (always `0` for unbiased sampling)
- `case_id`

### Run modes
```bash
# default: 100-case unbiased evaluation
docker compose run --rm full-test

# explicit unbiased mode
docker compose run --rm full-test-coverage

# evaluate original test split only
docker compose run --rm full-test-original

# run both original + unbiased in one report
docker compose run --rm -e FULL_TEST_MODE=both full-test
```

Optional caps are still supported:
```bash
# cap each evaluation set to 50 rows (debug)
docker compose run --rm -e FULL_TEST_MAX_ROWS=50 full-test
```

LLM request/response payloads and stage progress are written to:
- `outputs/logs/app.log`

### Final report charts
The generated `outputs/reports/final_report.html` now includes:
- a **test-set bias/coverage chart** (action distribution by evaluation set) to show whether the evaluated test data is shifted/biased, and
- a **final results chart** comparing key performance metrics across modes/evaluation sets.

PNG artifacts are also saved:
- `outputs/reports/test_set_distribution.png`
- `outputs/reports/final_results_chart.png`

## API
```bash
docker compose up api
```
Endpoints:
- `POST /decide`
- `GET /health`
- `GET /config`
- `GET /report/latest`

Example:
```bash
curl -X POST http://localhost:8000/decide -H 'content-type: application/json' -d @examples/decide_request.json
```

## Outputs
- `outputs/models/`
- `outputs/logs/`
- `outputs/predictions/`
- `outputs/reports/metrics.json`
- `outputs/reports/summary.csv`
- `outputs/reports/final_report.html`


## Adaptive hierarchical framework (new primary mode)

- Baseline rename: `adaptive_framework` is the previous flat `adaptive_full` implementation.
- New mode: `adaptive_hierarchical` adds hierarchical Stage A/Stage B selection, cost-sensitive control, calibrated uncertainty fallback, and hard guardrails.

Run the new framework:

```bash
python -m src.pipeline.run_experiment --mode adaptive_hierarchical --config configs/adaptive_hierarchical.yaml --seeds 3 --report --calibrate
```

Migration note: internal alias `adaptive_full -> adaptive_framework` remains for compatibility, but reports and CLI now use `adaptive_framework`.

## Experimental mode: `adaptive_ppo_agent`

This repository now includes a fully custom deep RL stack in PyTorch for sequential decision-making under a simulated MDP.

### Scientific scope and honesty
- The source dataset is static/tabular, so PPO is **not** trained on observed trajectories.
- Instead, PPO is trained in a **custom simulator-based marketing MDP** initialized from processed rows.
- This is a sequential policy extension for research benchmarking, **not** a direct supervised classifier.
- Reward composition and transition assumptions are configurable and can materially affect policy behavior.

### Train
```bash
python -m src.rl.train_ppo \
  --train-path data/processed/train.csv \
  --model-path outputs/models/adaptive_ppo_agent.pt \
  --timesteps 8000 \
  --seed 42 \
  --horizon 8 \
  --config configs/adaptive_hierarchical.yaml
```

### Evaluate
```bash
python -m src.rl.evaluate_ppo \
  --eval-path data/processed/test.csv \
  --model-path outputs/models/adaptive_ppo_agent.pt \
  --output-path outputs/predictions/adaptive_ppo_agent/eval/predictions.csv \
  --config configs/adaptive_hierarchical.yaml
```

### Full test orchestration
Use `ENABLE_PPO=1` to include PPO training in `scripts/full_test.sh`.
