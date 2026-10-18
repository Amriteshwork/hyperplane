# syntax=docker/dockerfile:1
# Backend image (FastAPI api + RQ worker share it; Compose picks the command).
# Build context: repo root — `docker build -f infra/docker/backend.Dockerfile .`

ARG PYTHON_VERSION=3.14
ARG UV_VERSION=0.13.1

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ── builder: resolve the locked environment into /app/.venv ───────────────────
FROM python:${PYTHON_VERSION}-slim-trixie AS builder
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# Dependencies first (cached layer; only invalidated by pyproject/lock changes).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=backend/uv.lock,target=uv.lock \
    uv sync --locked --no-dev --no-install-project

COPY backend/ /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# ── runtime: no uv, no build tooling, non-root ───────────────────────────────
FROM python:${PYTHON_VERSION}-slim-trixie AS runtime
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home app \
 && mkdir -p /data/storage && chown app:app /data/storage

COPY --from=builder --chown=app:app /app /app
COPY --chown=app:app content/ /srv/content/
COPY --chown=app:app datasets/ /srv/datasets/

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONTENT_DIR=/srv/content \
    DATASETS_DIR=/srv/datasets \
    LOCAL_STORAGE_ROOT=/data/storage

USER app
WORKDIR /app
EXPOSE 8000
CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8000"]
