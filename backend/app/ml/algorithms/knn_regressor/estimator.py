from sklearn.neighbors import KNeighborsRegressor

from app.ml.algorithms.knn_regressor.params import KNNRegressorParams


def build_estimator(params: KNNRegressorParams) -> KNeighborsRegressor:
    """What the platform runs; `train.py.j2` must construct the same estimator."""
    return KNeighborsRegressor(**params.model_dump())
