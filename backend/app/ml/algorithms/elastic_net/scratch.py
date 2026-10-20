"""Elastic Net from scratch with NumPy: Lasso's coordinate descent with an L2 term added."""

import numpy as np  # @a:numpy


# @a:soft-threshold
def soft_threshold(z: float, t: float) -> float:
    """Move z towards 0 by t, stopping at 0: the step that makes coefficients exactly zero."""
    return float(np.sign(z) * max(abs(z) - t, 0.0))


# @a:class
class ElasticNetScratch:
    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: np.ndarray = np.empty(0)
        self.intercept_: float = 0.0
        self.n_iter_: int = 0

    # @a:fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> ElasticNetScratch:
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        # @a:center
        x_mean = X.mean(axis=0) if self.fit_intercept else np.zeros(X.shape[1])
        y_mean = y.mean() if self.fit_intercept else 0.0
        Xc, yc = X - x_mean, y - y_mean
        # @a:split-penalty
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1 - self.l1_ratio)
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
                z = Xc[:, j] @ residual / n + curvature[j] * beta[j]
                new = soft_threshold(z, l1) / (curvature[j] + l2)
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
def alpha_max(X: np.ndarray, y: np.ndarray, l1_ratio: float) -> float:
    """The smallest α at which every coefficient is 0 (with an intercept)."""
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    return float(np.max(np.abs(Xc.T @ yc)) / (len(y) * l1_ratio))


# @a:grid-search
def grid_search(
    X: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray,
    l1_ratios: np.ndarray,
    k: int = 5,
    seed: int = 0,
) -> tuple[tuple[float, float], np.ndarray]:
    """The (α, l1_ratio) with the lowest k-fold validation MSE, and the mean MSE of every pair."""
    folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), k)  # @a:folds
    scores = np.zeros((len(alphas), len(l1_ratios)))
    for i, alpha in enumerate(alphas):
        for j, ratio in enumerate(l1_ratios):  # @a:pairs
            for fold in folds:
                train = np.setdiff1d(np.arange(len(y)), fold)  # every row not in this fold
                model = ElasticNetScratch(alpha, ratio).fit(X[train], y[train])
                fold_mse = np.mean((model.predict(X[fold]) - y[fold]) ** 2)
                scores[i, j] += fold_mse / k  # @a:fold-score
    best_i, best_j = np.unravel_index(np.argmin(scores), scores.shape)
    return (float(alphas[best_i]), float(l1_ratios[best_j])), scores
