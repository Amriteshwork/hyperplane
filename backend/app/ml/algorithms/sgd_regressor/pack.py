from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import cast

from pydantic import BaseModel
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import SGDRegressor

from app.ml.algorithms.sgd_regressor import recommend
from app.ml.algorithms.sgd_regressor.estimator import build_estimator
from app.ml.algorithms.sgd_regressor.params import SGDRegressorParams
from app.ml.algorithms.sgd_regressor.spec import SPEC
from app.ml.base import CoefTable, Estimator, ExplainedTemplate, Suitability, TaskType
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


class SGDRegressorPack:
    spec = SPEC
    params_model: type[BaseModel] = SGDRegressorParams

    def build_estimator(self, params: BaseModel) -> Estimator:
        params = SGDRegressorParams.model_validate(params.model_dump())
        return cast(Estimator, build_estimator(params))

    def templates(self) -> Mapping[str, ExplainedTemplate]:
        return _templates()

    def score(self, profile: DatasetProfile, task: TaskType) -> Suitability:
        return recommend.score(profile, task)

    def coefficients(self, fitted: Estimator, feature_names: list[str]) -> CoefTable:
        model = cast(TransformedTargetRegressor, fitted)
        # SGD learned weights for the standardised target; scale them back to its units.
        coef = cast(SGDRegressor, model.regressor_).coef_ * model.transformer_.scale_[0]
        return {name: float(c) for name, c in zip(feature_names, coef, strict=True)}
