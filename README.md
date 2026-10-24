# Hyperplane

**Learn machine learning by doing it.**

Hyperplane is an open, hands-on platform for learning ML. Bring a dataset (or pick a sample),
get algorithm recommendations with reasons, read the theory behind the algorithm, train a model,
and see the exact code that ran, explained line by line. Then download the trained model and
take the code with you.

> **Status:** early development — Phase 1 (Supervised Learning). See [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## Why

Most ML tutorials split *reading* from *doing*. Notebooks show code without context; courses show
theory without your own data. Hyperplane puts them on one screen:

- **Your data, real models** — upload a CSV/Parquet/Excel file or start from a built-in dataset.
- **Recommendations that teach** — every suggested algorithm comes with *why* (and *why not*).
- **Curated learning** — the best blogs and videos for every concept an algorithm uses; where good
  material doesn't exist, we write and record it ourselves.
- **No black boxes** — the training script you see is the one that produced your model, with a
  short and a deep explanation for every line. An "Under the hood" view shows the algorithm
  implemented from scratch in NumPy.
- **Take it home** — export the model (`.joblib`), metadata, a reproducible `train.py`,
  a `predict.py`, and a Jupyter notebook.

## How it works

```
Data  →  Explore  →  Target & features  →  Recommend  →  Learn  →  Configure  →  Train  →  Evaluate  →  Code  →  Export
```

1. Upload a dataset and get an automatic profile: column types, missing values, distributions, correlations.
2. Choose a target. Hyperplane infers the task type (regression / binary / multiclass).
3. Get ranked algorithm recommendations, or pick any compatible algorithm.
4. Read the algorithm page: intuition, math, assumptions, pitfalls, and curated resources.
5. Tune preprocessing and hyperparameters (beginner settings first, advanced ones tucked away).
6. Train and watch progress live; review metrics and charts.
7. Explore the explained code, then download everything.

## Tech stack

| Layer    | Tools |
|----------|-------|
| Frontend | Angular, Tailwind CSS, Spartan UI, Apache ECharts, Shiki, KaTeX |
| Backend  | Python, FastAPI, Pydantic, SQLAlchemy, Alembic |
| ML       | scikit-learn, pandas, NumPy, joblib |
| Infra    | PostgreSQL, Redis + RQ, S3-compatible storage (Garage), Docker Compose |

Architecture details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Getting started

### Prerequisites
- Docker Engine + the Compose plugin, and `make`
- For local development without Docker: Python 3.14 with [`uv`](https://github.com/astral-sh/uv),
  Node.js 24 LTS (or ≥ 22.22.3)

### Run everything with Docker

```bash
git clone https://github.com/Amriteshwork/hyperplane.git
cd hyperplane
make up          # creates .env from .env.example, builds images, waits until all services are healthy
```

- App: http://localhost:8080 — the home page shows the status of every service
- API docs (OpenAPI): http://localhost:8000/docs
- `make logs`, `make down` (`make down ARGS=-v` also deletes the data volumes); `make` lists all targets

Services: `web` (nginx + Angular), `api` (FastAPI), `worker` (RQ), `migrate` (runs Alembic once),
`postgres`, `redis`, `garage` (S3-compatible storage). Settings are documented in `.env.example`.

### Run the apps locally

Keep the infrastructure in Docker and run the apps on your machine:

```bash
docker compose --env-file .env -f infra/docker-compose.yml up -d --wait postgres redis garage

# backend (terminal 1 and 2)
cd backend
uv sync
uv run alembic upgrade head
make -C .. backend-dev                # API on :8000 with reload
make -C .. worker                     # RQ worker

# frontend (terminal 3)
cd frontend
npm ci
npm start                             # http://localhost:4200, proxies /api to :8000
npm run api:generate                  # after changing backend schemas: regenerate the typed client
```

### Tests & checks

```bash
make lint                             # ruff, ruff format, mypy, angular-eslint, prettier
make test                             # pytest + Vitest
cd frontend && npm run e2e            # Playwright smoke tests (desktop + mobile)
cd backend && uv run python -m scripts.content_check   # validate content/
make hooks                            # optional: pre-commit hooks
```

CI (`.github/workflows/ci.yml`) runs the same checks plus migrations, the e2e suite and
`content_check`.

## Repository layout

```
backend/     FastAPI app, ML engine (algorithm packs), worker, tests
frontend/    Angular app
content/     Learning content as Markdown/YAML: algorithms, concepts, curated resources, our posts
datasets/    Built-in sample datasets
infra/       Docker Compose, Dockerfiles, nginx
docs/        Architecture, roadmap, architecture decision records (ADRs)
```

## Algorithms (Phase 1)

| Family | Algorithms |
|--------|------------|
| Linear models | Linear, Polynomial, Ridge, Lasso, Elastic Net, Logistic Regression, SGD Regressor / Classifier |
| Instance-based | K-Nearest Neighbors |
| Probabilistic | Naive Bayes (Gaussian, Multinomial, Bernoulli), LDA / QDA |
| Support Vector Machines | Linear SVM, Kernel SVM, SVR |
| Tree-based | Decision Tree |
| Ensembles | Bagging, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost / LightGBM / CatBoost, Voting & Stacking |
| Neural | MLP Regressor / Classifier |

Progress is tracked in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Contributing

Contributions are welcome, especially learning content.

- **Add or improve content:** edit files under `content/`. Suggest a great blog or video by adding it
  to the relevant `resources.yaml` with a one-line reason it's worth a learner's time.
- **Add an algorithm:** create an *algorithm pack* in `backend/app/ml/algorithms/<id>/` following the
  contract in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#4-the-algorithm-pack-contract), plus its
  content folder. The shared contract tests must pass.
- Use [Conventional Commits](https://www.conventionalcommits.org/) and open a PR against `main`.

## License

To be decided before the first public release.

---

Built by [Amritesh](https://traintestval.com) · [GitHub](https://github.com/Amriteshwork) · [X](https://x.com/Amriteshnandan8) · [LinkedIn](https://www.linkedin.com/in/amriteshnandan)
