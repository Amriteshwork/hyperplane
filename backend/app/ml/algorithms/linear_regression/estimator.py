from sklearn.linear_model import LinearRegression

from app.ml.algorithms.linear_regression.params import LinearRegressionParams


def build_estimator(params: LinearRegressionParams) -> LinearRegression:
    """What the platform runs; `train.py.j2` must construct the same estimator."""
    return LinearRegression(**params.model_dump())
