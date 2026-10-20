"""Trains one run. Runs in the worker (`app.workers.jobs.train_run`)."""

from dataclasses import replace
from uuid import UUID

import sklearn
import structlog

from app.core.clock import Clock
from app.db.repositories.base import DatasetRepository, RunRepository
from app.domain.experiments import Run, RunStatus, RunView
from app.domain.run_events import final_event, progress_event, status_event
from app.ingest.parquet import read_parquet
from app.jobs.base import Event, EventPublisher
from app.jobs.names import run_events_channel
from app.ml.registry import AlgorithmRegistry
from app.ml.training import dump_model, library_versions, train
from app.storage.base import StorageBackend

log = structlog.get_logger(__name__)

# Errors are shown to the learner; long tracebacks-in-messages are cut.
_MAX_ERROR_LENGTH = 500


class TrainingService:
    def __init__(
        self,
        datasets: DatasetRepository,
        runs: RunRepository,
        storage: StorageBackend,
        registry: AlgorithmRegistry,
        events: EventPublisher,
        clock: Clock,
    ) -> None:
        self._datasets = datasets
        self._runs = runs
        self._storage = storage
        self._registry = registry
        self._events = events
        self._clock = clock

    async def run(self, run_id: UUID) -> None:
        """Train, store the model and report; on failure mark the run failed and re-raise."""
        view = await self._load(run_id)
        if view is None:
            log.warning("training_skipped_missing_run", run_id=str(run_id))
            return
        run = replace(view.run, status=RunStatus.RUNNING, started_at=self._clock.now())
        await self._runs.update_run(run)
        self._publish(run, status_event(run.status))
        try:
            dataset = await self._datasets.get(view.experiment.dataset_id)
            if dataset is None:
                raise LookupError("The dataset no longer exists.")
            result = train(
                read_parquet(self._storage.read(dataset.parquet_key)),
                view.training_config(),
                self._registry.get(run.algorithm_id),
                progress=lambda p: self._publish(run, progress_event(p)),
            )
            artifact_key = f"runs/{run.id}/model.joblib"
            self._storage.write(artifact_key, dump_model(result.pipeline))
        except Exception as exc:
            message = f"Training failed: {type(exc).__name__}: {exc}"[:_MAX_ERROR_LENGTH]
            await self._finish(replace(run, status=RunStatus.FAILED, error=message))
            raise
        await self._finish(
            replace(
                run,
                status=RunStatus.SUCCEEDED,
                report=result.report,
                artifact_key=artifact_key,
                env=library_versions(),
            )
        )
        log.info("run_trained", run_id=str(run.id), algorithm=run.algorithm_id)

    async def _load(self, run_id: UUID) -> RunView | None:
        run = await self._runs.get_run(run_id)
        experiment = await self._runs.get_experiment(run.experiment_id) if run else None
        return RunView(run, experiment) if run and experiment else None

    async def _finish(self, run: Run) -> None:
        finished = replace(run, finished_at=self._clock.now())
        await self._runs.update_run(finished)
        self._publish(finished, final_event(finished))

    def _publish(self, run: Run, event: Event) -> None:
        self._events.publish(run_events_channel(run.id), event)


def algorithm_version() -> str:
    """Packs wrap scikit-learn estimators, so its version pins the algorithm's behaviour."""
    return f"scikit-learn {sklearn.__version__}"
