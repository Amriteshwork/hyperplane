from sklearn.linear_model import SGDClassifier

from app.ml.algorithms.sgd_classifier.params import SGDClassifierParams


def build_estimator(params: SGDClassifierParams) -> SGDClassifier:
    """What the platform runs; `train.py.j2` must construct the same estimator."""
    return SGDClassifier(**params.model_dump())
