.DEFAULT_GOAL := help
.PHONY: help up down logs test lint fmt e2e

help: ## Show this help
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-6s %s\n", $$1, $$2}'

up: ## Build and start the stack (UI and API on http://localhost:8000)
	docker compose up --build -d

down: ## Stop the stack (the model cache and results are kept)
	docker compose down

logs: ## Follow the API and worker logs
	docker compose logs -f api worker

test: ## Unit and integration tests with coverage (needs ffmpeg)
	uv run pytest --cov --cov-fail-under=80

lint: ## Lint, check formatting and type-check
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy

fmt: ## Format code and apply safe lint fixes
	uv run ruff check --fix src tests
	uv run ruff format src tests

e2e: ## Start the stack and run the end-to-end tests against it
	docker compose up --build -d --wait
	uv run pytest -m e2e -v
