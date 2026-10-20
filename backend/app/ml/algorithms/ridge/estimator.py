from sklearn.linear_model import Ridge

from app.ml.algorithms.ridge.params import RidgeParams


def build_estimator(params: RidgeParams) -> Ridge:
    """What the platform runs; `train.py.j2` must construct the same estimator."""
    return Ridge(**params.model_dump())
