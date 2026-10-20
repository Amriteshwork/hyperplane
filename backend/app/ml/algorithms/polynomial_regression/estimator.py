from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from app.ml.algorithms.polynomial_regression.params import PolynomialRegressionParams


def build_estimator(params: PolynomialRegressionParams) -> Pipeline:
    """What the platform runs; `train.py.j2` must construct the same estimator."""
    return Pipeline(
        [
            (
                "expand",
                PolynomialFeatures(
                    degree=params.degree,
                    interaction_only=params.interaction_only,
                    # The regression learns the intercept; a column of ones would duplicate it.
                    include_bias=False,
                ),
            ),
            ("regress", LinearRegression(fit_intercept=params.fit_intercept)),
        ]
    )
