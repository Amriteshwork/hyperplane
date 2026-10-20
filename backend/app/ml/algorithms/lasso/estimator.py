from sklearn.linear_model import Lasso

from app.ml.algorithms.lasso.params import LassoParams


def build_estimator(params: LassoParams) -> Lasso:
    """What the platform runs; `train.py.j2` must construct the same estimator."""
    return Lasso(**params.model_dump())
