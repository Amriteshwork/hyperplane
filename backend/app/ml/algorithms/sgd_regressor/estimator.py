from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from app.ml.algorithms.sgd_regressor.params import SGDRegressorParams


def build_estimator(params: SGDRegressorParams) -> TransformedTargetRegressor:
    """What the platform runs; `train.py.j2` must construct the same estimator.

    The target is standardised too, so `tol`, `epsilon` and the learning rate don't depend on
    its units (ADR 0018); predictions come back in the original units.
    """
    return TransformedTargetRegressor(
        regressor=SGDRegressor(**params.model_dump()), transformer=StandardScaler()
    )
