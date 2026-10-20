from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ExperimentRow, RunRow
from app.domain.experiments import Experiment, Run, RunStatus
from app.ml.preprocessing import PreprocessingConfig
from app.ml.training import TrainingReport
from app.ml.types import TaskType

# `runs.metrics` holds the report minus its plots, which go to `runs.plots`.
_PLOTS = "plots"


class SqlRunRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def add_experiment(self, experiment: Experiment) -> None:
        self._db.add(ExperimentRow(**_experiment_columns(experiment)))
        await self._db.commit()

    async def get_experiment(self, experiment_id: UUID) -> Experiment | None:
        row = await self._db.get(ExperimentRow, experiment_id, populate_existing=True)
        return _experiment(row) if row is not None else None

    async def add_run(self, run: Run) -> None:
        self._db.add(RunRow(**_run_columns(run)))
        await self._db.commit()

    async def get_run(self, run_id: UUID) -> Run | None:
        row = await self._db.get(RunRow, run_id, populate_existing=True)
        return _run(row) if row is not None else None

    async def update_run(self, run: Run) -> None:
        row = await self._db.get(RunRow, run.id)
        if row is None:
            raise LookupError(f"run {run.id} does not exist")
        for column, value in _run_columns(run).items():
            setattr(row, column, value)
        await self._db.commit()


def _experiment_columns(experiment: Experiment) -> dict[str, Any]:
    return {
        "id": experiment.id,
        "dataset_id": experiment.dataset_id,
        "target": experiment.target,
        "features": experiment.preprocessing.features,
        "task_type": experiment.task_type.value,
        "preprocessing": experiment.preprocessing.model_dump(mode="json"),
        "split": {"test_size": experiment.test_size},
        "seed": experiment.seed,
        "created_at": experiment.created_at,
    }


def _experiment(row: ExperimentRow) -> Experiment:
    return Experiment(
        id=row.id,
        dataset_id=row.dataset_id,
        target=row.target,
        task_type=TaskType(row.task_type),
        preprocessing=PreprocessingConfig.model_validate(row.preprocessing),
        test_size=float(row.split["test_size"]),
        seed=row.seed,
        created_at=row.created_at,
    )


def _run_columns(run: Run) -> dict[str, Any]:
    report = run.report.model_dump(mode="json") if run.report else None
    evaluation = report["evaluation"] if report else None
    return {
        "id": run.id,
        "experiment_id": run.experiment_id,
        "algorithm_id": run.algorithm_id,
        "algorithm_version": run.algorithm_version,
        "params": run.params,
        "status": run.status.value,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "metrics": (
            {**report, "evaluation": {"metrics": evaluation["metrics"]}}
            if report and evaluation
            else None
        ),
        "plots": {_PLOTS: evaluation[_PLOTS]} if evaluation else None,
        "artifact_key": run.artifact_key,
        "error": run.error,
        "env": run.env,
        "created_at": run.created_at,
    }


def _run(row: RunRow) -> Run:
    report = None
    if row.metrics is not None:
        data = dict(row.metrics)
        plots = (row.plots or {}).get(_PLOTS, [])
        data["evaluation"] = {**data["evaluation"], _PLOTS: plots}
        report = TrainingReport.model_validate(data)
    return Run(
        id=row.id,
        experiment_id=row.experiment_id,
        algorithm_id=row.algorithm_id,
        algorithm_version=row.algorithm_version,
        params=row.params,
        status=RunStatus(row.status),
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        report=report,
        artifact_key=row.artifact_key,
        error=row.error,
        env=row.env,
    )
