.PHONY: build api full-test prepare-synth prepare-dh train run-exp test

build:
	docker compose build

api:
	docker compose up api

full-test:
	docker compose run --rm full-test

prepare-synth:
	docker compose run --rm app python -m src.data.prepare --dataset synthetic

prepare-dh:
	docker compose run --rm app python -m src.data.prepare --dataset dunnhumby

train:
	docker compose run --rm app python -m src.training.train_xgboost

run-exp:
	docker compose run --rm app python -m src.pipeline.run_experiment --mode adaptive_full_framework

test:
	pytest -q
