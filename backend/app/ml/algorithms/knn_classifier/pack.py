from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from app.ml.algorithms.knn_classifier import recommend
from app.ml.algorithms.knn_classifier.estimator import build_estimator
from app.ml.algorithms.knn_classifier.params import KNNClassifierParams
from app.ml.algorithms.knn_classifier.spec import SPEC
from app.ml.algorithms.knn_inspect import k_curve, neighbour_report
from app.ml.base import (
    Estimator,
    ExplainedTemplate,
    NeighbourReport,
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


class KNNClassifierPack:
    spec = SPEC
    params_model: type[BaseModel] = KNNClassifierParams

    def build_estimator(self, params: BaseModel) -> Estimator:
        params = KNNClassifierParams.model_validate(params.model_dump())
        return cast(Estimator, build_estimator(params))

    def templates(self) -> Mapping[str, ExplainedTemplate]:
        return _templates()

    def score(self, profile: DatasetProfile, task: TaskType) -> Suitability:
        return recommend.score(profile, task)

    def tuning_curve(
        self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: ArrayLike
    ) -> TuningCurve | None:
        return k_curve(pipeline, X_train, y_train, "Cross-validated accuracy")

    def neighbours(
        self,
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> NeighbourReport:
        predictions = pipeline.predict(X_test)
        votes = pipeline.predict_proba(X_test)
        # How little of the vote the right class got; a class unseen in training got none.
        actual = pd.Index(pipeline.classes_).get_indexer(y_test)
        errors = 1 - np.where(actual >= 0, votes[np.arange(len(votes)), actual], 0.0)
        return neighbour_report(pipeline, X_train, y_train, X_test, y_test, predictions, errors)
