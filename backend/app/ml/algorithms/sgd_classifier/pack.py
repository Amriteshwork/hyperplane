from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import cast

from pydantic import BaseModel
from sklearn.linear_model import SGDClassifier

from app.ml.algorithms.sgd_classifier import recommend
from app.ml.algorithms.sgd_classifier.estimator import build_estimator
from app.ml.algorithms.sgd_classifier.params import SGDClassifierParams
from app.ml.algorithms.sgd_classifier.spec import SPEC
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


class SGDClassifierPack:
    spec = SPEC
    params_model: type[BaseModel] = SGDClassifierParams

    def build_estimator(self, params: BaseModel) -> Estimator:
        params = SGDClassifierParams.model_validate(params.model_dump())
        return cast(Estimator, build_estimator(params))

    def templates(self) -> Mapping[str, ExplainedTemplate]:
        return _templates()

    def score(self, profile: DatasetProfile, task: TaskType) -> Suitability:
        return recommend.score(profile, task)

    def class_coefficients(self, fitted: Estimator, feature_names: list[str]) -> ClassCoefTable:
        model = cast(SGDClassifier, fitted)
        classes = [str(c) for c in model.classes_]
        # Binary models learn one row of weights, for the positive class classes_[1];
        # with more classes, one-vs-rest learns a row per class.
        scored = classes[1:] if len(classes) == 2 else classes
        return {
            label: {name: float(c) for name, c in zip(feature_names, row, strict=True)}
            for label, row in zip(scored, model.coef_, strict=True)
        }
