"""Polynomial Regression from scratch with NumPy: expand the features, then ordinary least squares."""

# @a:imports
from itertools import combinations, combinations_with_replacement

import numpy as np


# @a:expand
def polynomial_features(
    X: np.ndarray, degree: int = 2, interaction_only: bool = False
) -> np.ndarray:
    """Every product of up to `degree` columns, in scikit-learn's PolynomialFeatures order."""
    pick = combinations if interaction_only else combinations_with_replacement
    columns = [
        np.prod(X[:, list(term)], axis=1)  # @a:term
        for d in range(1, degree + 1)
        for term in pick(range(X.shape[1]), d)
    ]
    return np.column_stack(columns)


# @a:class
class PolynomialRegressionScratch:
    def __init__(self, degree: int = 2, interaction_only: bool = False) -> None:
        self.degree = degree
        self.interaction_only = interaction_only
        self.coef_: np.ndarray = np.empty(0)
        self.intercept_: float = 0.0

    # @a:fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> PolynomialRegressionScratch:
        P = polynomial_features(np.asarray(X, dtype=float), self.degree, self.interaction_only)
        A = np.column_stack([np.ones(len(P)), P])
        theta, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)
        self.intercept_, self.coef_ = float(theta[0]), theta[1:]
        return self

    # @a:predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        P = polynomial_features(np.asarray(X, dtype=float), self.degree, self.interaction_only)
        return P @ self.coef_ + self.intercept_


# @a:degree-sweep
def degree_sweep(
    x: np.ndarray, y: np.ndarray, degrees: range = range(1, 13), seed: int = 0
) -> list[tuple[int, float, float]]:
    """Train and test MSE of one feature `x` for each degree: the overfitting curve."""
    order = np.random.default_rng(seed).permutation(len(x))
    train, test = order[: len(x) * 3 // 4], order[len(x) * 3 // 4 :]
    results = []
    for degree in degrees:
        model = PolynomialRegressionScratch(degree).fit(x[train, None], y[train])
        mse = [np.mean((model.predict(x[i, None]) - y[i]) ** 2) for i in (train, test)]
        results.append((degree, float(mse[0]), float(mse[1])))  # @a:errors
    return results
