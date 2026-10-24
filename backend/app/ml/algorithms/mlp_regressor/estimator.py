from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from app.ml.algorithms.mlp_params import mlp_kwargs
from app.ml.algorithms.mlp_regressor.params import MLPRegressorParams


def build_estimator(params: MLPRegressorParams) -> TransformedTargetRegressor:
    """What the platform runs; `train.py.j2` must construct the same estimator.

    The target is standardised too, as for SGD (ADR 0018): the output weights start small,
    and the loss, `tol` and learning rate then mean the same whatever the target's units.
    """
    return TransformedTargetRegressor(
        regressor=MLPRegressor(**mlp_kwargs(params)), transformer=StandardScaler()
    )
