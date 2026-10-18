"""FastAPI app factory. `fastapi dev|run app/main.py` serves the module-level `app`."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import Settings, get_settings
from app.core.deps import build_container
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging
from app.core.middleware import REQUEST_ID_HEADER, RequestIdMiddleware


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app; pass `settings` to override the environment (tests)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Here rather than at import: uvicorn applies its own logging config after import.
        configure_logging(settings.log_level, settings.log_format)
        container = build_container(settings)
        app.state.container = container
        structlog.get_logger(__name__).info(
            "startup", env=settings.app_env, algorithms=len(container.registry)
        )
        try:
            yield
        finally:
            await container.aclose()

    app = FastAPI(
        title="Hyperplane API",
        version="0.1.0",
        lifespan=lifespan,
        openapi_url="/openapi.json",
    )
    register_error_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER],
    )
    # Added last → outermost, so the request id covers CORS and error responses too.
    app.add_middleware(RequestIdMiddleware)
    app.include_router(api_router)
    return app


app = create_app()
