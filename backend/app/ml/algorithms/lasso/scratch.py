"""Lasso from scratch with NumPy: coordinate descent, one soft-thresholded coefficient at a time."""

import numpy as np  # @a:numpy


# @a:soft-threshold
def soft_threshold(z: float, t: float) -> float:
    """Move z towards 0 by t, stopping at 0: the step that makes coefficients exactly zero."""
    return float(np.sign(z) * max(abs(z) - t, 0.0))


# @a:class
class LassoScratch:
    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: np.ndarray = np.empty(0)
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    # @a:fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> LassoScratch:
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        # @a:center
        x_mean = X.mean(axis=0) if self.fit_intercept else np.zeros(X.shape[1])
        y_mean = y.mean() if self.fit_intercept else 0.0
        Xc, yc = X - x_mean, y - y_mean
        # @a:init
        n = len(yc)
        curvature = (Xc**2).sum(axis=0) / n  # each column's mean square
        beta = np.zeros(X.shape[1])
        residual = yc.copy()  # y − Xβ, kept up to date as β changes
        # @a:sweep
        for sweep in range(1, self.max_iter + 1):
            largest_step = 0.0
            for j in np.flatnonzero(curvature):  # a constant column carries no signal: skip it
                # @a:update
                rho = Xc[:, j] @ residual / n + curvature[j] * beta[j]
                new = soft_threshold(rho, self.alpha) / curvature[j]
                residual -= Xc[:, j] * (new - beta[j])
                largest_step = max(largest_step, abs(new - beta[j]))
                beta[j] = new
            # @a:converged
            if largest_step < self.tol:
                break
        self.coef_, self.n_iter_ = beta, sweep
        # @a:intercept
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        return self

    # @a:predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


# @a:alpha-max
def alpha_max(X: np.ndarray, y: np.ndarray) -> float:
    """The smallest α at which every coefficient is 0 (with an intercept)."""
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    return float(np.max(np.abs(Xc.T @ yc)) / len(y))


# @a:path
def lasso_path(X: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Coefficients for each α (one row per α): watch features drop out one by one as α grows."""
    return np.array(
        [LassoScratch(alpha, tol=1e-10, max_iter=10_000).fit(X, y).coef_ for alpha in alphas]
    )
