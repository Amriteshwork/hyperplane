"""SGD classification from scratch with NumPy: one row at a time, one-vs-rest for many classes."""

import numpy as np  # @a:numpy


# @a:losses
def log_loss(score: float, sign: float) -> tuple[float, float]:
    """Logistic regression's loss log(1 + e^(−y·z)) and its slope, for a label y = ±1."""
    return float(np.logaddexp(0.0, -sign * score)), -sign / (1.0 + np.exp(sign * score))


def hinge(score: float, sign: float) -> tuple[float, float]:
    """A linear SVM's loss max(0, 1 − y·z): rows beyond the margin cost nothing and don't pull."""
    if sign * score >= 1.0:
        return 0.0, 0.0
    return 1.0 - sign * score, -sign


LOSSES = {"log_loss": log_loss, "hinge": hinge}


# @a:schedule
def step_size(schedule: str, t: int, alpha: float, eta0: float, offset: float) -> float:
    """The learning rate at update t (counting from 1)."""
    if schedule == "optimal":
        return 1.0 / (alpha * (offset + t - 1))  # 1/t decay, started where steps are sensible
    if schedule == "invscaling":
        return eta0 / np.sqrt(t)
    return eta0  # constant


def optimal_offset(alpha: float) -> float:
    """Where the optimal schedule starts: scikit-learn's heuristic, after Léon Bottou.

    The first step is about the size of a typical weight, 1/α^¼ (both losses' slopes are ≤ 1).
    """
    return 1.0 / (np.sqrt(1.0 / np.sqrt(alpha)) * alpha)


# @a:class
class SGDClassifierScratch:
    def __init__(
        self,
        loss: str = "log_loss",
        alpha: float = 0.0001,
        learning_rate: str = "optimal",
        eta0: float = 0.01,
        max_iter: int = 1000,
        tol: float | None = 0.001,
        n_iter_no_change: int = 5,
        fit_intercept: bool = True,
        random_state: int = 0,
    ) -> None:
        self.loss = loss
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.classes_: np.ndarray = np.empty(0)
        self.coef_: np.ndarray = np.empty((0, 0))
        self.intercept_: np.ndarray = np.empty(0)
        self.n_iter_: int = 0

    # @a:fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> SGDClassifierScratch:
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        # @a:one-vs-rest
        scored = self.classes_[1:] if len(self.classes_) == 2 else self.classes_
        fits = [self._fit_binary(X, np.where(y == c, 1.0, -1.0)) for c in scored]
        self.coef_ = np.array([w for w, _, _ in fits])
        self.intercept_ = np.array([b for _, b, _ in fits])
        self.n_iter_ = max(epochs for _, _, epochs in fits)
        return self

    # @a:binary
    def _fit_binary(self, X: np.ndarray, signs: np.ndarray) -> tuple[np.ndarray, float, int]:
        """SGD for one yes/no problem with labels ±1: the same loop as the regressor's."""
        rng = np.random.default_rng(self.random_state)
        n, p = X.shape
        w, b, t = np.zeros(p), 0.0, 1
        loss_and_slope = LOSSES[self.loss]
        offset = optimal_offset(self.alpha)
        best, stale = np.inf, 0
        for epoch in range(1, self.max_iter + 1):
            total = 0.0
            for i in rng.permutation(n):
                # @a:update
                eta = step_size(self.learning_rate, t, self.alpha, self.eta0, offset)
                loss, slope = loss_and_slope(X[i] @ w + b, signs[i])
                w -= eta * (slope * X[i] + self.alpha * w)
                if self.fit_intercept:
                    b -= eta * slope
                total += loss
                t += 1
            # @a:stop
            if self.tol is not None:
                stale = stale + 1 if total / n > best - self.tol else 0
                best = min(best, total / n)
                if stale >= self.n_iter_no_change:
                    break
        return w, b, epoch

    # @a:decision
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """The linear score w·x + b of each row (one column per class for one-vs-rest)."""
        scores = np.asarray(X, dtype=float) @ self.coef_.T + self.intercept_
        return scores[:, 0] if len(self.classes_) == 2 else scores

    # @a:predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        if len(self.classes_) == 2:
            return self.classes_[(scores > 0).astype(int)]  # positive score: classes_[1]
        return self.classes_[np.argmax(scores, axis=1)]  # the most confident "yes" wins

    # @a:predict-proba
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.loss != "log_loss":
            raise AttributeError("only the log_loss model has probabilities here")
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        if len(self.classes_) == 2:
            return np.column_stack([1 - p, p])
        return p / p.sum(axis=1, keepdims=True)  # one-vs-rest "yes" chances, rescaled to sum to 1
