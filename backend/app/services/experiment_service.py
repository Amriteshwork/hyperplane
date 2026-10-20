"""Experiment/run use cases: validate a training setup, queue the run, report on it."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from uuid import UUID, uuid4

import structlog
from pydantic import ValidationError

from app.core.clock import Clock
from app.core.errors import NotFoundError, ServiceUnavailableError, UnprocessableError
from app.db.repositories.base import RunRepository
from app.domain.experiments import Experiment, Run, RunStatus, RunView
from app.domain.run_events import final_event, is_final, status_event
from app.jobs.base import Event, EventStream, EventSubscriber, JobQueue
from app.jobs.names import TRAIN_RUN, run_events_channel
from app.ml.base import AlgorithmPack
from app.ml.evaluation import EVALUATORS
from app.ml.preprocessing import FEATURE_ROLES, UnsupportedFeatureError, assign_roles
from app.ml.profiling import DatasetProfile
from app.ml.registry import AlgorithmRegistry, UnknownAlgorithmError
from app.ml.types import TaskType
from app.schemas.experiments import (
    AlgorithmOptionOut,
    ExperimentIn,
    FeatureOptionOut,
    TargetOptionOut,
    TrainingOptionsOut,
)
from app.services.dataset_service import DatasetService
from app.services.training_service import algorithm_version

log = structlog.get_logger(__name__)


class ExperimentService:
    def __init__(
        self,
        datasets: DatasetService,
        runs: RunRepository,
        registry: AlgorithmRegistry,
        queue: JobQueue,
        events: EventSubscriber,
        clock: Clock,
    ) -> None:
        self._datasets = datasets
        self._runs = runs
        self._registry = registry
        self._queue = queue
        self._events = events
        self._clock = clock

    async def create(self, session_id: UUID, request: ExperimentIn) -> RunView:
        """Validate the setup against the dataset's profile and queue one training run."""
        profile = await self._datasets.get_profile(session_id, request.dataset_id)
        task = _task_type(profile, request.target)
        _check_features(profile, request.target, request.features)
        try:
            preprocessing = assign_roles(profile, request.features)
        except UnsupportedFeatureError as exc:
            raise UnprocessableError(str(exc)) from None
        pack = self._compatible_pack(request.algorithm_id, task, profile.n_rows)
        try:
            params = pack.params_model.model_validate(request.params)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
            )
            raise UnprocessableError(f"Invalid hyperparameters: {problems}.") from None

        now = self._clock.now()
        experiment = Experiment(
            id=uuid4(),
            dataset_id=request.dataset_id,
            target=request.target,
            task_type=task,
            preprocessing=preprocessing,
            test_size=request.split.test_size,
            seed=request.seed,
            created_at=now,
        )
        run = Run(
            id=uuid4(),
            experiment_id=experiment.id,
            algorithm_id=pack.spec.id,
            algorithm_version=algorithm_version(),
            params=params.model_dump(mode="json"),
            status=RunStatus.QUEUED,
            created_at=now,
        )
        await self._runs.add_experiment(experiment)
        await self._runs.add_run(run)
        try:
            self._queue.enqueue(TRAIN_RUN, run_id=str(run.id))
        except Exception as exc:
            log.error("training_enqueue_failed", run_id=str(run.id), error=repr(exc))
            run = replace(run, status=RunStatus.FAILED, error="Could not queue training.")
            await self._runs.update_run(run)
            raise ServiceUnavailableError("The job queue is unavailable; try again.") from exc
        log.info("run_queued", run_id=str(run.id), algorithm=run.algorithm_id)
        return RunView(run, experiment)

    async def training_options(self, session_id: UUID, dataset_id: UUID) -> TrainingOptionsOut:
        profile = await self._datasets.get_profile(session_id, dataset_id)
        return TrainingOptionsOut(
            targets=[
                TargetOptionOut(
                    column=candidate.column,
                    task_type=candidate.task_type,
                    reason=candidate.reason,
                    algorithms=[
                        AlgorithmOptionOut(id=pack.spec.id, name=pack.spec.name)
                        for pack in self._registry.list()
                        if candidate.task_type in pack.spec.task_types
                        and candidate.task_type in EVALUATORS
                    ],
                )
                for candidate in profile.target_candidates
            ],
            features=[
                FeatureOptionOut(
                    column=column.name,
                    semantic_type=column.semantic_type,
                    role=FEATURE_ROLES[column.semantic_type],
                )
                for column in profile.columns
            ],
        )

    async def get_run(self, session_id: UUID, run_id: UUID) -> RunView:
        """The run, if its dataset belongs to this session."""
        run = await self._runs.get_run(run_id)
        experiment = await self._runs.get_experiment(run.experiment_id) if run else None
        if run is None or experiment is None:
            raise NotFoundError("Run not found.")
        try:
            await self._datasets.get(session_id, experiment.dataset_id)
        except NotFoundError:
            raise NotFoundError("Run not found.") from None
        return RunView(run, experiment)

    async def open_events(self, session_id: UUID, run_id: UUID) -> RunEvents:
        """The run's events from its current status until it finishes. Close it when done.

        All database reads happen here, before streaming, so the stream holds no connection.
        """
        run = (await self.get_run(session_id, run_id)).run
        if run.status.finished:
            return RunEvents(final_event(run))
        subscription = await self._events.subscribe(run_events_channel(run_id))
        # Re-read: the run may have finished before the subscription started.
        latest = await self._runs.get_run(run_id) or run
        if latest.status.finished:
            await subscription.aclose()
            return RunEvents(final_event(latest))
        return RunEvents(status_event(latest.status), subscription)

    def _compatible_pack(self, algorithm_id: str, task: TaskType, n_rows: int) -> AlgorithmPack:
        try:
            pack = self._registry.get(algorithm_id)
        except UnknownAlgorithmError:
            raise UnprocessableError(f"Unknown algorithm {algorithm_id!r}.") from None
        if task not in pack.spec.task_types:
            raise UnprocessableError(f"{pack.spec.name} can't be used for {task} tasks.")
        if task not in EVALUATORS:
            raise UnprocessableError(f"Training {task} models isn't available yet.")
        if n_rows > pack.spec.max_rows_default:
            raise UnprocessableError(
                f"{pack.spec.name} supports up to {pack.spec.max_rows_default:,} rows; "
                f"this dataset has {n_rows:,}."
            )
        return pack


def _task_type(profile: DatasetProfile, target: str) -> TaskType:
    if target not in {c.name for c in profile.columns}:
        raise UnprocessableError(f"The dataset has no column {target!r}.")
    for candidate in profile.target_candidates:
        if candidate.column == target:
            return candidate.task_type
    raise UnprocessableError(f"{target!r} can't be predicted (wrong type or too many gaps).")


def _check_features(profile: DatasetProfile, target: str, features: list[str]) -> None:
    columns = {c.name for c in profile.columns}
    if unknown := [f for f in features if f not in columns]:
        raise UnprocessableError(f"Unknown feature columns: {', '.join(unknown)}.")
    if target in features:
        raise UnprocessableError("The target can't also be a feature.")
    if len(set(features)) != len(features):
        raise UnprocessableError("Each feature may be listed only once.")


@dataclass(frozen=True, slots=True)
class RunEvents:
    """`first`, then live events until a final one. `aclose` releases the subscription."""

    first: Event
    subscription: EventStream | None = None

    def __aiter__(self) -> AsyncIterator[Event]:
        return self._events()

    async def _events(self) -> AsyncIterator[Event]:
        yield self.first
        if self.subscription is None or is_final(self.first):
            return
        async for event in self.subscription:
            yield event
            if is_final(event):
                return

    async def aclose(self) -> None:
        if self.subscription is not None:
            await self.subscription.aclose()
