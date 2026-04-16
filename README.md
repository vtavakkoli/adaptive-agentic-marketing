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
This verifies environment assumptions, picks real-data mode when available, otherwise synthetic mode, then preprocesses, derives labels, trains XGBoost, runs all experiment modes, calls the host Ollama SLM endpoint, computes metrics, and generates reports.

By default, full test experiments run on **10 rows** for quick feedback and visible progress logs.  
You can increase or disable the cap later:
```bash
# run 50 rows
docker compose run --rm -e FULL_TEST_MAX_ROWS=50 full-test

# run the full test set
docker compose run --rm -e FULL_TEST_MAX_ROWS=all full-test
```

LLM request/response payloads and stage progress are written to:
- `outputs/logs/app.log`

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
