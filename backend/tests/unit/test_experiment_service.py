"""Live run events, which the API tests can't drive (TestClient reads a stream to its end)."""

from dataclasses import replace
from datetime import timedelta
from uuid import UUID, uuid4

import pytest

from app.domain.datasets import Dataset, DatasetSource, DatasetStatus, FileFormat
from app.domain.experiments import Experiment, Run, RunStatus
from app.ingest.readers import TableLimits
from app.jobs.names import run_events_channel
from app.ml.preprocessing import PreprocessingConfig
from app.ml.registry import AlgorithmRegistry
from app.ml.types import TaskType
from app.services.dataset_service import DatasetService, UploadPolicy
from app.services.experiment_service import ExperimentService
from tests.fakes.clock import FixedClock
from tests.fakes.events import InMemoryEventBus
from tests.fakes.jobs import FakeJobQueue
from tests.fakes.repositories import InMemoryDatasetRepository, InMemoryRunRepository
from tests.fakes.storage import InMemoryStorage

pytestmark = pytest.mark.anyio
SESSION = uuid4()


class FinishesAfterFirstRead(InMemoryRunRepository):
    """Simulates the run finishing between the ownership check and the subscription."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    async def get_run(self, run_id: UUID) -> Run | None:
        self.reads += 1
        run = await super().get_run(run_id)
        if run is not None and self.reads > 1:
            return replace(run, status=RunStatus.SUCCEEDED)
        return run


async def _setup(
    runs: InMemoryRunRepository, bus: InMemoryEventBus
) -> tuple[ExperimentService, Run]:
    clock = FixedClock()
    datasets = InMemoryDatasetRepository()
    dataset = Dataset(
        id=uuid4(),
        session_id=SESSION,
        name="d.csv",
        source=DatasetSource.UPLOAD,
        sample_id=None,
        format=FileFormat.CSV,
        size_bytes=1,
        raw_key="raw",
        parquet_key="parquet",
        n_rows=10,
        n_columns=2,
        status=DatasetStatus.READY,
        profile=None,
        error=None,
        created_at=clock.now(),
        expires_at=clock.now() + timedelta(days=1),
    )
    await datasets.add(dataset)
    experiment = Experiment(
        id=uuid4(),
        dataset_id=dataset.id,
        target="y",
        task_type=TaskType.REGRESSION,
        preprocessing=PreprocessingConfig(numeric=("x",)),
        test_size=0.2,
        seed=0,
        created_at=clock.now(),
    )
    run = Run(
        id=uuid4(),
        experiment_id=experiment.id,
        algorithm_id="linear_regression",
        algorithm_version="test",
        params={},
        status=RunStatus.RUNNING,
        created_at=clock.now(),
    )
    await runs.add_experiment(experiment)
    await runs.add_run(run)
    policy = UploadPolicy(
        max_bytes=1, limits=TableLimits(max_rows=1, max_columns=1), retention=timedelta(days=1)
    )
    dataset_service = DatasetService(datasets, InMemoryStorage(), FakeJobQueue(), clock, policy)
    service = ExperimentService(
        dataset_service, runs, AlgorithmRegistry.discover(), FakeJobQueue(), bus, clock
    )
    return service, run


async def test_follows_a_live_run_until_it_finishes() -> None:
    bus = InMemoryEventBus()
    service, run = await _setup(InMemoryRunRepository(), bus)
    channel = run_events_channel(run.id)

    events = await service.open_events(SESSION, run.id)
    stream = aiter(events)
    assert await anext(stream) == {"type": "status", "status": "running"}

    bus.publish(channel, {"type": "progress", "step": 1, "total": 3, "message": "Fitting"})
    bus.publish(channel, {"type": "done", "status": "succeeded"})
    bus.publish(channel, {"type": "progress", "step": 9})  # after the end: never delivered
    assert [e["type"] async for e in stream] == ["progress", "done"]
    await events.aclose()  # the route's dependency does this when the response ends
    assert bus.subscriber_count(channel) == 0


async def test_closing_before_iterating_releases_the_subscription() -> None:
    bus = InMemoryEventBus()
    service, run = await _setup(InMemoryRunRepository(), bus)

    events = await service.open_events(SESSION, run.id)
    assert bus.subscriber_count(run_events_channel(run.id)) == 1
    await events.aclose()
    assert bus.subscriber_count(run_events_channel(run.id)) == 0


async def test_a_run_finishing_before_the_subscription_still_ends_the_stream() -> None:
    bus = InMemoryEventBus()
    service, run = await _setup(FinishesAfterFirstRead(), bus)

    events = await service.open_events(SESSION, run.id)

    assert [e async for e in events] == [{"type": "done", "status": "succeeded"}]
    assert bus.subscriber_count(run_events_channel(run.id)) == 0
