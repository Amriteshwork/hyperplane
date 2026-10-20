from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import cast

from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

from app.ml.algorithms.logistic_regression import recommend
from app.ml.algorithms.logistic_regression.estimator import build_estimator
from app.ml.algorithms.logistic_regression.params import LogisticRegressionParams
from app.ml.algorithms.logistic_regression.spec import SPEC
from app.ml.base import ClassCoefTable, Estimator, ExplainedTemplate, Suitability, TaskType
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


class LogisticRegressionPack:
    spec = SPEC
    params_model: type[BaseModel] = LogisticRegressionParams

    def build_estimator(self, params: BaseModel) -> Estimator:
        params = LogisticRegressionParams.model_validate(params.model_dump())
        return cast(Estimator, build_estimator(params))

    def templates(self) -> Mapping[str, ExplainedTemplate]:
        return _templates()

    def score(self, profile: DatasetProfile, task: TaskType) -> Suitability:
        return recommend.score(profile, task)

    def class_coefficients(self, fitted: Estimator, feature_names: list[str]) -> ClassCoefTable:
        model = cast(LogisticRegression, fitted)
        classes = [str(c) for c in model.classes_]
        # Binary models learn one row of weights, for the positive class classes_[1].
        scored = classes[1:] if len(classes) == 2 else classes
        return {
            label: {name: float(c) for name, c in zip(feature_names, row, strict=True)}
            for label, row in zip(scored, model.coef_, strict=True)
        }
