from sklearn.neighbors import KNeighborsClassifier

from app.ml.algorithms.knn_classifier.params import KNNClassifierParams


def build_estimator(params: KNNClassifierParams) -> KNeighborsClassifier:
    """What the platform runs; `train.py.j2` must construct the same estimator."""
    return KNeighborsClassifier(**params.model_dump())
