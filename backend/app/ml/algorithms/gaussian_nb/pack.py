from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import cast

import pandas as pd
from numpy.typing import ArrayLike
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from app.ml.algorithms.gaussian_nb import recommend
from app.ml.algorithms.gaussian_nb.estimator import build_estimator
from app.ml.algorithms.gaussian_nb.params import GaussianNBParams
from app.ml.algorithms.gaussian_nb.report import gaussian_report, smoothing_curve
from app.ml.algorithms.gaussian_nb.spec import SPEC
from app.ml.base import (
    Estimator,
    ExplainedTemplate,
    GaussianReport,
    Suitability,
    TaskType,
    TuningCurve,
)
from app.ml.codegen import TemplateFile, load_templates
from app.ml.profiling import DatasetProfile

_DIR = Path(__file__).parent


@cache
def _templates() -> Mapping[str, ExplainedTemplate]:
    return load_templates(
        _DIR,
        {
            "train": TemplateFile("train.py.j2", "Training script"),
            "scratch": TemplateFile("scratch.py", "Under the hood (NumPy)"),
        },
    )


class GaussianNBPack:
    spec = SPEC
    params_model: type[BaseModel] = GaussianNBParams

    def build_estimator(self, params: BaseModel) -> Estimator:
        params = GaussianNBParams.model_validate(params.model_dump())
        return cast(Estimator, build_estimator(params))

    def templates(self) -> Mapping[str, ExplainedTemplate]:
        return _templates()

    def score(self, profile: DatasetProfile, task: TaskType) -> Suitability:
        return recommend.score(profile, task)

    def tuning_curve(
        self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: ArrayLike
    ) -> TuningCurve | None:
        return smoothing_curve(pipeline, X_train, y_train)

    def class_gaussians(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> GaussianReport:
        return gaussian_report(pipeline, X_train, y_train, X_test, y_test)
