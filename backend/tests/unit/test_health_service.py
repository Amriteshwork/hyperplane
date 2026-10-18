import anyio
import pytest

from app.schemas.health import OverallState, ServiceState
from app.services.health_service import HealthService
from tests.fakes.health import StaticProbe


class SlowProbe:
    name = "slow"

    async def check(self) -> None:
        await anyio.sleep(5)


@pytest.mark.anyio
async def test_slow_probe_times_out() -> None:
    result = await HealthService([SlowProbe(), StaticProbe("db")], timeout_seconds=0.05).check()

    assert result.status is OverallState.DEGRADED
    assert result.services["slow"].status is ServiceState.DOWN
    assert result.services["slow"].error == "Timeout"
    assert result.services["db"].status is ServiceState.UP


@pytest.mark.anyio
async def test_no_probes_is_ok() -> None:
    result = await HealthService([], timeout_seconds=1).check()
    assert result.status is OverallState.OK
