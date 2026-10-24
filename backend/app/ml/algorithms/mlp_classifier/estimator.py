from sklearn.neural_network import MLPClassifier

from app.ml.algorithms.mlp_classifier.params import MLPClassifierParams
from app.ml.algorithms.mlp_params import mlp_kwargs


def build_estimator(params: MLPClassifierParams) -> MLPClassifier:
    """What the platform runs; `train.py.j2` must construct the same estimator."""
    return MLPClassifier(**mlp_kwargs(params))
