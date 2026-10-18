"""Health check response."""

from enum import StrEnum

from pydantic import BaseModel, Field


class ServiceState(StrEnum):
    UP = "up"
    DOWN = "down"


class OverallState(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"


class ServiceStatus(BaseModel):
    status: ServiceState
    latency_ms: float = Field(description="Time the probe took, in milliseconds.")
    error: str | None = Field(
        default=None, description="Error type when down (no internals are exposed)."
    )


class HealthStatus(BaseModel):
    status: OverallState
    services: dict[str, ServiceStatus] = Field(description="Keyed by service: db, redis, storage.")
