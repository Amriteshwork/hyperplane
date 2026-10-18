# syntax=docker/dockerfile:1
# Frontend image: Angular production build served by nginx (which also proxies /api).
# Build context: repo root — `docker build -f infra/docker/frontend.Dockerfile .`

ARG NODE_VERSION=24.22
ARG NGINX_VERSION=1.30

# ── build ─────────────────────────────────────────────────────────────────────
FROM node:${NODE_VERSION}-alpine AS build
WORKDIR /app
RUN --mount=type=cache,target=/root/.npm \
    --mount=type=bind,source=frontend/package.json,target=package.json \
    --mount=type=bind,source=frontend/package-lock.json,target=package-lock.json \
    npm ci --no-audit --no-fund
COPY frontend/ ./
# The generated API client (core/api/) is committed, so no live backend is needed here.
RUN npm run build

# ── serve ─────────────────────────────────────────────────────────────────────
FROM nginx:${NGINX_VERSION}-alpine AS runtime
COPY infra/nginx/default.conf /etc/nginx/conf.d/default.conf
COPY infra/nginx/security-headers.conf /etc/nginx/snippets/security-headers.conf
COPY --from=build /app/dist/frontend/browser/ /usr/share/nginx/html/
EXPOSE 80
