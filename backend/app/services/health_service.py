"""Runs every dependency probe concurrently and summarises the result."""

import time
from typing import Protocol

import anyio
import structlog

from app.schemas.health import HealthStatus, OverallState, ServiceState, ServiceStatus

log = structlog.get_logger(__name__)


class HealthProbe(Protocol):
    name: str

    async def check(self) -> None:
        """Return if the dependency is reachable; raise otherwise."""
        ...


class HealthService:
    def __init__(self, probes: list[HealthProbe], *, timeout_seconds: float) -> None:
        self._probes = probes
        self._timeout = timeout_seconds

    async def check(self) -> HealthStatus:
        """Probe all dependencies in parallel; a slow probe counts as down."""
        results: dict[str, ServiceStatus] = {}
        async with anyio.create_task_group() as tg:
            for probe in self._probes:
                tg.start_soon(self._run, probe, results)
        overall = (
            OverallState.OK
            if all(s.status is ServiceState.UP for s in results.values())
            else OverallState.DEGRADED
        )
        return HealthStatus(status=overall, services=dict(sorted(results.items())))

    async def _run(self, probe: HealthProbe, results: dict[str, ServiceStatus]) -> None:
        started = time.perf_counter()
        error: str | None = None
        try:
            with anyio.fail_after(self._timeout):
                await probe.check()
        except TimeoutError:
            error = "Timeout"
        except Exception as exc:  # noqa: BLE001  (any failure means the service is down)
            error = type(exc).__name__
            log.warning("health_probe_failed", service=probe.name, error=repr(exc))
        results[probe.name] = ServiceStatus(
            status=ServiceState.DOWN if error else ServiceState.UP,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            error=error,
        )
