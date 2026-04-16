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
docker compose run --rm app python -m src.pipeline.run_experiment --mode adaptive_full_framework
```

Equivalent local Python:
```bash
python -m src.training.train_xgboost
python -m src.pipeline.run_experiment --mode adaptive_full_framework
```

## Full end-to-end test
```bash
docker compose run --rm full-test
```
This verifies environment assumptions, picks real-data mode when available (otherwise synthetic mode), preprocesses data, trains XGBoost, builds a **100-case coverage-biased evaluation set derived from the original test split**, runs all experiment modes, calls the host Ollama SLM endpoint, computes metrics, and generates reports.

### What changed
- `full-test` now defaults to `FULL_TEST_MODE=coverage`, which runs the **coverage-biased 100-case** evaluation set (`artifacts/coverage_test_set.csv`).
- Coverage set generation is reproducible with `FULL_TEST_SEED` (default `42`).
- Original test-set evaluation is still available.

### Coverage-biased 100-case set generation
The coverage set is built from the original labeled test data (`data/processed/test.csv`) with traceable metadata:
- controlled perturbations anchored to each source row,
- threshold/boundary-focused cases (near rule cutoffs),
- conflict scenarios (for example high need + high fatigue),
- stratified action/channel sampling,
- bucket balancing across low/medium/high value bands.

Each generated row includes:
- `source_case_id`
- `augmentation_type`
- `coverage_bucket`
- `edge_case_flag`
- `case_id`

### Run modes
```bash
# default: 100-case coverage-biased evaluation
docker compose run --rm full-test

# explicit coverage mode
docker compose run --rm full-test-coverage

# evaluate original test split only
docker compose run --rm full-test-original

# run both original + coverage in one report
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
