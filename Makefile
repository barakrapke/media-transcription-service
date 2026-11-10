.DEFAULT_GOAL := help
SHELL := /bin/bash

E2E_PROJECT := transcription-e2e
E2E_PORT    ?= 8001
E2E_MODEL   ?= Systran/faster-whisper-tiny

.PHONY: help install up down logs dev-redis dev-api dev-worker lint fmt test test-docker e2e

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies (including dev tools) with uv
	uv sync --frozen

up: ## Build and start the full stack (UI/API on http://localhost:8000)
	docker compose up --build -d
	@echo "UI: http://localhost:$${API_PORT:-8000}  API docs: http://localhost:$${API_PORT:-8000}/docs"
	@echo "The first start downloads the model; follow it with: make logs"

down: ## Stop the stack (keeps volumes: model cache, results)
	docker compose down

logs: ## Follow API and worker logs
	docker compose logs -f api worker

dev-redis: ## Run only Redis in Docker, for running the API/worker locally
	docker run --rm -d --name transcription-dev-redis -p 6379:6379 redis:7.4-alpine

dev-api: ## Run the API locally (needs Redis and ffmpeg)
	uv run python -m transcriber.api

dev-worker: ## Run a worker locally (needs Redis and ffmpeg)
	uv run python -m transcriber.worker

lint: ## Lint, check formatting and type-check
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy

fmt: ## Auto-format and apply safe lint fixes
	uv run ruff check --fix src tests
	uv run ruff format src tests

test: ## Unit + integration tests with coverage (needs ffmpeg)
	uv run pytest --cov --cov-report=term --cov-fail-under=80

test-docker: ## Unit + integration tests inside Docker (no local ffmpeg/Python needed)
	docker run --rm -v "$(CURDIR)":/src:ro -w /src -e UV_PROJECT_ENVIRONMENT=/tmp/venv \
	  -e UV_CACHE_DIR=/tmp/uv-cache -e UV_LINK_MODE=copy -e COVERAGE_FILE=/tmp/.coverage ghcr.io/astral-sh/uv:0.6.14-python3.12-bookworm-slim \
	  sh -c 'apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg >/dev/null \
	    && uv sync --frozen && uv run --no-sync pytest --cov --cov-fail-under=80 -p no:cacheprovider' 

e2e: ## End-to-end tests against a dedicated stack using the tiny model
	API_PORT=$(E2E_PORT) MODEL_NAME=$(E2E_MODEL) docker compose -p $(E2E_PROJECT) up --build -d --wait
	E2E_BASE_URL=http://localhost:$(E2E_PORT) uv run pytest -m e2e -v; \
	  status=$$?; \
	  if [ $$status -ne 0 ]; then docker compose -p $(E2E_PROJECT) logs --tail 100 api worker; fi; \
	  docker compose -p $(E2E_PROJECT) down; \
	  exit $$status
