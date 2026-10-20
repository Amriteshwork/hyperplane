"""The platform run: split → preprocess → fit → evaluate, exactly as the training template does.

The golden-equivalence test runs the rendered template and checks it predicts what `train` does.
"""

import io
import platform
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import joblib
import numpy as np
import pandas as pd
import sklearn
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.ml.base import (
    Buildable,
    CoefTable,
    Parameterised,
    Progress,
    ProgressFn,
    ReportsClassCoefficients,
    ReportsCoefficients,
)
from app.ml.codegen import CodegenContext
from app.ml.evaluation import EVALUATORS, Evaluation, Predictions
from app.ml.preprocessing import PreprocessingConfig, build_preprocessor
from app.ml.types import TaskType

STAGES = ("Splitting the data", "Fitting the model", "Evaluating on the test set")


@runtime_checkable
class Trainable(Buildable, Parameterised, Protocol):
    """What the trainer needs from a pack."""


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    target: str
    task_type: TaskType
    preprocessing: PreprocessingConfig
    test_size: float = Field(gt=0, lt=1)
    seed: int
    #: Validated hyperparameters with defaults filled in, as JSON.
    params: dict[str, JsonValue]


class Coefficient(BaseModel):
    model_config = ConfigDict(frozen=True)

    feature: str
    value: float
    #: The class this weight belongs to (classifiers only).
    target_class: str | None = None


class TrainingReport(BaseModel):
    """The JSON-safe outcome of a run, stored on it and returned by the API."""

    model_config = ConfigDict(frozen=True)

    n_train: int
    n_test: int
    evaluation: Evaluation
    #: Largest absolute effect first (per class, in class order); empty if not reported.
    coefficients: list[Coefficient] = []


@dataclass(frozen=True, slots=True)
class TrainingResult:
    pipeline: Pipeline
    report: TrainingReport
    X_test: pd.DataFrame
    predictions: NDArray[Any]
    #: Class probabilities on `X_test`, for classifiers that have them.
    probabilities: NDArray[Any] | None = None


def train(
    df: pd.DataFrame,
    config: TrainingConfig,
    pack: Trainable,
    progress: ProgressFn | None = None,
) -> TrainingResult:
    report = _reporter(progress)
    df = _missing_as_nan(df).dropna(subset=[config.target])
    X = df[config.preprocessing.features]
    y = df[config.target]

    report(0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.seed
    )
    model = pack.build_estimator(pack.params_model.model_validate(config.params))
    pipeline = Pipeline(
        [("preprocess", build_preprocessor(config.preprocessing)), ("model", model)]
    )

    report(1)
    pipeline.fit(X_train, y_train)

    report(2)
    predictions = pipeline.predict(X_test)
    # Pipelines expose predict_proba / classes_ only when the model does (regressors don't).
    probabilities = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
    classes = tuple(getattr(pipeline, "classes_", ()))
    evaluation = EVALUATORS[config.task_type](
        Predictions(y_test, predictions, probabilities, classes)
    )
    return TrainingResult(
        pipeline=pipeline,
        report=TrainingReport(
            n_train=len(X_train),
            n_test=len(X_test),
            evaluation=evaluation,
            coefficients=_coefficients(pack, pipeline),
        ),
        X_test=X_test,
        predictions=predictions,
        probabilities=probabilities,
    )


def codegen_context(config: TrainingConfig, data_file: str) -> CodegenContext:
    """The template settings that reproduce `train(…, config, …)` on `data_file`."""
    return CodegenContext(
        data_file=data_file,
        target=config.target,
        task_type=config.task_type,
        numeric_features=list(config.preprocessing.numeric),
        categorical_features=list(config.preprocessing.categorical),
        test_size=config.test_size,
        seed=config.seed,
        params=config.params,
    )


def dump_model(pipeline: Pipeline) -> bytes:
    """The fitted pipeline as joblib bytes. Only ever loaded by the user, never by the server."""
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    return buffer.getvalue()


def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def _missing_as_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet gives None for gaps in object columns where the script's CSV gives NaN, and
    scikit-learn's imputers only treat NaN as missing."""
    df = df.copy()
    for column in df.columns[df.dtypes == object]:
        df[column] = df[column].where(df[column].notna(), np.nan)
    return df


def _reporter(progress: ProgressFn | None) -> Callable[[int], None]:
    def report(step: int) -> None:
        if progress is not None:
            progress(Progress(step=step, total=len(STAGES), message=STAGES[step]))

    return report


def _coefficients(pack: Trainable, pipeline: Pipeline) -> list[Coefficient]:
    names = [str(n) for n in pipeline.named_steps["preprocess"].get_feature_names_out()]
    model = pipeline.named_steps["model"]
    if isinstance(pack, ReportsClassCoefficients):
        return [
            Coefficient(feature=name, value=value, target_class=label)
            for label, table in pack.class_coefficients(model, names).items()
            for name, value in _ranked(table)
        ]
    if isinstance(pack, ReportsCoefficients):
        return [
            Coefficient(feature=n, value=v) for n, v in _ranked(pack.coefficients(model, names))
        ]
    return []


def _ranked(table: CoefTable) -> list[tuple[str, float]]:
    return sorted(table.items(), key=lambda item: abs(item[1]), reverse=True)
