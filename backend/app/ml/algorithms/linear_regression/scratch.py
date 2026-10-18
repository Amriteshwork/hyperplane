"""Linear Regression from scratch with NumPy: what scikit-learn's LinearRegression does inside."""

import numpy as np  # @a:numpy


# @a:class
class LinearRegressionScratch:
    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray = np.empty(0)
        self.intercept_: float = 0.0

    # @a:design-matrix
    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        if not self.fit_intercept:
            return X
        return np.column_stack([np.ones(len(X)), X])

    # @a:fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegressionScratch:
        A = self._design_matrix(np.asarray(X, dtype=float))
        # @a:solve
        theta, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)
        # @a:unpack
        if self.fit_intercept:
            self.intercept_, self.coef_ = float(theta[0]), theta[1:]
        else:
            self.intercept_, self.coef_ = 0.0, theta
        return self

    # @a:predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


# @a:gradient-descent
def fit_gradient_descent(
    X: np.ndarray, y: np.ndarray, learning_rate: float = 0.1, epochs: int = 5_000
) -> tuple[np.ndarray, float]:
    """The same model, learned iteratively. Assumes X is standardised (mean 0, std 1)."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        error = X @ w + b - y  # @a:error
        w -= learning_rate * (2 / n) * (X.T @ error)  # @a:update
        b -= learning_rate * (2 / n) * error.sum()
    return w, b


# @a:r2
def r2_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot)
