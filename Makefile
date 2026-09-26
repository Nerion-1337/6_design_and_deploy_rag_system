.PHONY: install fetch-data build-index run-api test evaluate test-all docker-build docker-test

install:
	uv sync --frozen

fetch-data:
	uv run python -m src.data.fetch_openagenda

build-index:
	uv run python -m src.indexing.build_index

run-api:
	uv run python -m uvicorn src.api.main:app --reload --port 8001

test:
	uv run python -m pytest tests/ -v

api-test:
	uv run -m pytest tests/api_test.py -v

evaluate:
	uv run python -m tests.evaluate_rag

test-all:
	bash scripts/run_all_tests.sh

docker-build:
	docker compose build

docker-test:
	docker compose run puls_events_rag bash scripts/run_all_tests.sh
