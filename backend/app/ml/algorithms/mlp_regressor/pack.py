from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import cast

from pydantic import BaseModel
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor

from app.ml.algorithms.mlp_inspect import input_weight_importance, mlp_network
from app.ml.algorithms.mlp_regressor import recommend
from app.ml.algorithms.mlp_regressor.estimator import build_estimator
from app.ml.algorithms.mlp_regressor.params import MLPRegressorParams
from app.ml.algorithms.mlp_regressor.spec import SPEC
from app.ml.base import (
    Estimator,
    ExplainedTemplate,
    ImportanceTable,
    Network,
    Suitability,
    TaskType,
    TrainingCurve,
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


class MLPRegressorPack:
    spec = SPEC
    params_model: type[BaseModel] = MLPRegressorParams

    def build_estimator(self, params: BaseModel) -> Estimator:
        params = MLPRegressorParams.model_validate(params.model_dump())
        return cast(Estimator, build_estimator(params))

    def templates(self) -> Mapping[str, ExplainedTemplate]:
        return _templates()

    def score(self, profile: DatasetProfile, task: TaskType) -> Suitability:
        return recommend.score(profile, task)

    def training_curve(self, fitted: Estimator) -> TrainingCurve | None:
        model = _mlp(fitted)
        if not hasattr(model, "loss_curve_"):  # lbfgs: full-batch, no epochs
            return None
        return TrainingCurve(
            loss=model.loss_curve_,
            validation=model.validation_scores_,
            validation_label="Validation R²",
        )

    def network(self, fitted: Estimator, feature_names: list[str]) -> Network:
        # The output is the standardised target; TransformedTargetRegressor unscales it.
        return mlp_network(_mlp(fitted), feature_names, ["prediction"])

    def feature_importance(self, fitted: Estimator, feature_names: list[str]) -> ImportanceTable:
        return input_weight_importance(_mlp(fitted), feature_names)


def _mlp(fitted: Estimator) -> MLPRegressor:
    return cast(MLPRegressor, cast(TransformedTargetRegressor, fitted).regressor_)
