from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import cast

from pydantic import BaseModel
from sklearn.neural_network import MLPClassifier

from app.ml.algorithms.mlp_classifier import recommend
from app.ml.algorithms.mlp_classifier.estimator import build_estimator
from app.ml.algorithms.mlp_classifier.params import MLPClassifierParams
from app.ml.algorithms.mlp_classifier.spec import SPEC
from app.ml.algorithms.mlp_inspect import input_weight_importance, mlp_network
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


class MLPClassifierPack:
    spec = SPEC
    params_model: type[BaseModel] = MLPClassifierParams

    def build_estimator(self, params: BaseModel) -> Estimator:
        params = MLPClassifierParams.model_validate(params.model_dump())
        return cast(Estimator, build_estimator(params))

    def templates(self) -> Mapping[str, ExplainedTemplate]:
        return _templates()

    def score(self, profile: DatasetProfile, task: TaskType) -> Suitability:
        return recommend.score(profile, task)

    def training_curve(self, fitted: Estimator) -> TrainingCurve | None:
        model = cast(MLPClassifier, fitted)
        if not hasattr(model, "loss_curve_"):  # lbfgs: full-batch, no epochs
            return None
        return TrainingCurve(
            loss=model.loss_curve_,
            validation=model.validation_scores_,
            validation_label="Validation accuracy",
        )

    def network(self, fitted: Estimator, feature_names: list[str]) -> Network:
        model = cast(MLPClassifier, fitted)
        classes = [str(c) for c in model.classes_]
        # Two classes share one logistic output: the probability of the last class.
        outputs = [f"P({classes[-1]})"] if model.n_outputs_ == 1 else classes
        return mlp_network(model, feature_names, outputs)

    def feature_importance(self, fitted: Estimator, feature_names: list[str]) -> ImportanceTable:
        return input_weight_importance(cast(MLPClassifier, fitted), feature_names)
