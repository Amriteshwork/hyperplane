"""Logistic regression from scratch with NumPy: gradient descent on the penalised log-loss."""

import numpy as np  # @a:numpy


# @a:sigmoid
def sigmoid(z: np.ndarray) -> np.ndarray:
    """1 / (1 + e^−z), squashing any score into a probability; written to never overflow."""
    return np.exp(-np.logaddexp(0.0, -z))


# @a:softmax
def softmax(Z: np.ndarray) -> np.ndarray:
    """Exponentiate each row's scores and normalise them to sum to 1."""
    E = np.exp(Z - Z.max(axis=1, keepdims=True))  # subtracting the max avoids overflow
    return E / E.sum(axis=1, keepdims=True)


# @a:class
class LogisticRegressionScratch:
    def __init__(
        self, C: float = 1.0, fit_intercept: bool = True, max_iter: int = 100_000, tol: float = 1e-8
    ) -> None:
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.classes_: np.ndarray = np.empty(0)
        self.coef_: np.ndarray = np.empty((0, 0))
        self.intercept_: np.ndarray = np.empty(0)
        self.n_iter_: int = 0

    # @a:fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegressionScratch:
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        # @a:targets
        self.classes_ = np.unique(y)
        binary = len(self.classes_) == 2
        scored = self.classes_[1:] if binary else self.classes_  # one score per column
        Y = (y[:, None] == scored).astype(float)  # 1 where the row belongs to that class
        # @a:design
        n = len(y)
        X1 = np.column_stack([X, np.ones(n)]) if self.fit_intercept else X
        penalised = np.ones((X1.shape[1], 1))
        if self.fit_intercept:
            penalised[-1] = 0.0  # the intercept is not penalised
        W = np.zeros((X1.shape[1], Y.shape[1]))
        # @a:step
        curvature = 0.25 if binary else 0.5  # the loss bends at most this much per unit
        step = 1 / (curvature * np.linalg.norm(X1, 2) ** 2 / n + 1 / (self.C * n))
        # @a:descend
        for iteration in range(1, self.max_iter + 1):
            P = sigmoid(X1 @ W) if binary else softmax(X1 @ W)
            gradient = X1.T @ (P - Y) / n + penalised * W / (self.C * n)  # @a:gradient
            # @a:converged
            if np.abs(gradient).max() < self.tol:
                break
            W -= step * gradient
        self.n_iter_ = iteration
        # @a:store
        self.coef_ = W[:-1].T if self.fit_intercept else W.T
        self.intercept_ = W[-1] if self.fit_intercept else np.zeros(W.shape[1])
        return self

    # @a:predict-proba
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = np.asarray(X, dtype=float) @ self.coef_.T + self.intercept_
        if len(self.classes_) == 2:
            p = sigmoid(Z[:, 0])  # the probability of classes_[1]
            return np.column_stack([1 - p, p])
        return softmax(Z)

    # @a:predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# @a:boundary
def decision_boundary(model: LogisticRegressionScratch, x1: np.ndarray) -> np.ndarray:
    """For a binary model on two features: the x₂ where p = 0.5, i.e. w₁x₁ + w₂x₂ + b = 0."""
    (w1, w2), b = model.coef_[0], model.intercept_[0]
    return -(w1 * x1 + b) / w2
