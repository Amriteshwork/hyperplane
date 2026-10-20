from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import cast

from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from app.ml.algorithms.polynomial_regression import recommend
from app.ml.algorithms.polynomial_regression.estimator import build_estimator
from app.ml.algorithms.polynomial_regression.params import PolynomialRegressionParams
from app.ml.algorithms.polynomial_regression.spec import SPEC
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


class PolynomialRegressionPack:
    spec = SPEC
    params_model: type[BaseModel] = PolynomialRegressionParams

    def build_estimator(self, params: BaseModel) -> Estimator:
        params = PolynomialRegressionParams.model_validate(params.model_dump())
        return cast(Estimator, build_estimator(params))

    def templates(self) -> Mapping[str, ExplainedTemplate]:
        return _templates()

    def score(self, profile: DatasetProfile, task: TaskType) -> Suitability:
        return recommend.score(profile, task)

    def coefficients(self, fitted: Estimator, feature_names: list[str]) -> CoefTable:
        steps = cast(Pipeline, fitted).named_steps
        # One coefficient per expanded term, named like "x0^2" or "x0 x1".
        names = cast(PolynomialFeatures, steps["expand"]).get_feature_names_out(feature_names)
        coef = cast(LinearRegression, steps["regress"]).coef_
        return {str(name): float(c) for name, c in zip(names, coef, strict=True)}
