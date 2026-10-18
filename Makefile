# Hyperplane developer entry points. `make` (or `make help`) lists targets.

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# Host-run processes (backend-dev, worker) read the same settings as Compose.
-include .env
export

COMPOSE := docker compose --env-file .env -f infra/docker-compose.yml
BACKEND := cd backend &&
FRONTEND := cd frontend &&

.PHONY: help env up down logs backend-dev worker frontend-dev \
        test backend-test frontend-test lint backend-lint frontend-lint \
        format api-generate hooks

help: ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(firstword $(MAKEFILE_LIST)) | awk -F':.*?## ' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

env: ## Create .env from .env.example if missing
	@test -f .env || { cp .env.example .env && echo "created .env from .env.example"; }

# ── Docker stack ──────────────────────────────────────────────────────────────
up: env ## Build and start the whole stack, wait until every service is healthy
	$(COMPOSE) up --build --detach --wait

down: ## Stop the stack (keeps volumes; add ARGS=-v to drop them)
	$(COMPOSE) down $(ARGS)

logs: ## Follow logs (all services, or s=<service>)
	$(COMPOSE) logs --follow --tail=200 $(s)

# ── Local processes (infra from `make up`, or your own) ───────────────────────
backend-dev: env ## Run the API with reload on :8000
	$(BACKEND) uv run fastapi dev app/main.py

worker: env ## Run an RQ worker on the host
	$(BACKEND) uv run rq worker --url "$(REDIS_URL)" "$(JOB_QUEUE_NAME)"

frontend-dev: ## Run ng serve on :4200 (proxies /api → :8000)
	$(FRONTEND) npm start

api-generate: ## Regenerate the typed API client from the running backend's OpenAPI
	$(FRONTEND) npm run api:generate

# ── Quality ───────────────────────────────────────────────────────────────────
test: backend-test frontend-test ## Run all unit/component tests

backend-test:
	$(BACKEND) uv run pytest

frontend-test:
	$(FRONTEND) npm test -- --watch=false

lint: backend-lint frontend-lint ## Lint, format-check and type-check everything

backend-lint:
	$(BACKEND) uv run ruff check . && uv run ruff format --check . && uv run mypy app

frontend-lint:
	$(FRONTEND) npm run lint && npx prettier --check .

format: ## Auto-format and apply safe lint fixes
	$(BACKEND) uv run ruff check --fix . && uv run ruff format .
	$(FRONTEND) npx prettier --write .

hooks: ## Install the pre-commit git hooks
	pre-commit install
