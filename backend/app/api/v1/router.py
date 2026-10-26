"""Assembles the v1 API."""

from fastapi import APIRouter

from app.api.v1.routes import (
    algorithms,
    concepts,
    datasets,
    experiments,
    families,
    health,
    papers,
    recommendations,
    runs,
    samples,
    tracks,
)

api_router = APIRouter(prefix="/api/v1")
for module in (
    health,
    samples,
    datasets,
    families,
    algorithms,
    recommendations,
    experiments,
    runs,
    concepts,
    tracks,
    papers,
):
    api_router.include_router(module.router)
