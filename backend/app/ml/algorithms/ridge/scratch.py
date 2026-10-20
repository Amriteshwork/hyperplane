"""Ridge regression from scratch with NumPy: least squares plus an L2 penalty, in closed form."""

import numpy as np  # @a:numpy


# @a:class
class RidgeScratch:
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True) -> None:
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray = np.empty(0)
        self.intercept_: float = 0.0

    # @a:fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeScratch:
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        # @a:center
        x_mean = X.mean(axis=0) if self.fit_intercept else np.zeros(X.shape[1])
        y_mean = y.mean() if self.fit_intercept else 0.0
        Xc, yc = X - x_mean, y - y_mean
        # @a:solve
        penalty = self.alpha * np.eye(X.shape[1])
        self.coef_ = np.linalg.solve(Xc.T @ Xc + penalty, Xc.T @ yc)
        # @a:intercept
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        return self

    # @a:predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


# @a:path
def ridge_path(X: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Coefficients for each α (one row per α): watch them shrink towards 0 as α grows."""
    return np.array([RidgeScratch(alpha).fit(X, y).coef_ for alpha in alphas])


# @a:cross-validate
def choose_alpha(
    X: np.ndarray, y: np.ndarray, alphas: np.ndarray, k: int = 5, seed: int = 0
) -> tuple[float, np.ndarray]:
    """The α with the lowest k-fold validation MSE, and the mean MSE of every α."""
    folds = np.array_split(np.random.default_rng(seed).permutation(len(y)), k)  # @a:folds
    scores = np.zeros(len(alphas))
    for i, alpha in enumerate(alphas):
        for fold in folds:
            train = np.setdiff1d(np.arange(len(y)), fold)  # every row not in this fold
            model = RidgeScratch(alpha).fit(X[train], y[train])
            scores[i] += np.mean((model.predict(X[fold]) - y[fold]) ** 2) / k  # @a:fold-score
    return float(alphas[np.argmin(scores)]), scores
