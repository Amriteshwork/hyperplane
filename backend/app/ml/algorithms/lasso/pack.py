from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import cast

from pydantic import BaseModel
from sklearn.linear_model import Lasso

from app.ml.algorithms.lasso import recommend
from app.ml.algorithms.lasso.estimator import build_estimator
from app.ml.algorithms.lasso.params import LassoParams
from app.ml.algorithms.lasso.spec import SPEC
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


class LassoPack:
    spec = SPEC
    params_model: type[BaseModel] = LassoParams

    def build_estimator(self, params: BaseModel) -> Estimator:
        params = LassoParams.model_validate(params.model_dump())
        return cast(Estimator, build_estimator(params))

    def templates(self) -> Mapping[str, ExplainedTemplate]:
        return _templates()

    def score(self, profile: DatasetProfile, task: TaskType) -> Suitability:
        return recommend.score(profile, task)

    def coefficients(self, fitted: Estimator, feature_names: list[str]) -> CoefTable:
        coef = cast(Lasso, fitted).coef_
        return {name: float(c) for name, c in zip(feature_names, coef, strict=True)}
