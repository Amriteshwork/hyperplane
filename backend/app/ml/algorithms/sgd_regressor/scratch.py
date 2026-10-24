"""SGD regression from scratch with NumPy: visit the rows one at a time, step downhill on each."""

import numpy as np  # @a:numpy


# @a:losses
def squared_error(error: float, epsilon: float) -> tuple[float, float]:
    """½e² and its slope e: big errors cost a lot, and pull hard."""
    return 0.5 * error**2, error


def huber(error: float, epsilon: float) -> tuple[float, float]:
    """Squared up to ε, linear beyond: an outlier's pull is capped at ε."""
    if abs(error) <= epsilon:
        return 0.5 * error**2, error
    return epsilon * abs(error) - 0.5 * epsilon**2, epsilon * np.sign(error)


def epsilon_insensitive(error: float, epsilon: float) -> tuple[float, float]:
    """Free inside ±ε, linear outside: only errors bigger than ε count."""
    if abs(error) <= epsilon:
        return 0.0, 0.0
    return abs(error) - epsilon, float(np.sign(error))


LOSSES = {
    "squared_error": squared_error,
    "huber": huber,
    "epsilon_insensitive": epsilon_insensitive,
}


# @a:schedule
def step_size(schedule: str, t: int, eta0: float, power_t: float) -> float:
    """The learning rate at update t (counting from 1)."""
    if schedule == "constant":
        return eta0
    return eta0 / t**power_t  # invscaling: big steps first, smaller ones as we settle


# @a:class
class SGDRegressorScratch:
    def __init__(
        self,
        loss: str = "squared_error",
        alpha: float = 0.0001,
        epsilon: float = 0.1,
        learning_rate: str = "invscaling",
        eta0: float = 0.01,
        power_t: float = 0.25,
        max_iter: int = 1000,
        tol: float | None = 0.001,
        n_iter_no_change: int = 5,
        fit_intercept: bool = True,
        random_state: int = 0,
    ) -> None:
        self.loss = loss
        self.alpha = alpha
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.coef_: np.ndarray = np.empty(0)
        self.intercept_: float = 0.0
        self.n_iter_: int = 0
        self.loss_curve_: list[float] = []

    # @a:fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> SGDRegressorScratch:
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        # @a:init
        rng = np.random.default_rng(self.random_state)
        n, p = X.shape
        w, b, t = np.zeros(p), 0.0, 1
        loss_and_slope = LOSSES[self.loss]
        best, stale = np.inf, 0
        self.loss_curve_ = []
        # @a:epoch
        for epoch in range(1, self.max_iter + 1):
            total = 0.0
            for i in rng.permutation(n):  # @a:shuffle
                # @a:update
                eta = step_size(self.learning_rate, t, self.eta0, self.power_t)
                loss, slope = loss_and_slope(X[i] @ w + b - y[i], self.epsilon)
                w -= eta * (slope * X[i] + self.alpha * w)  # one row's gradient, plus the L2 pull
                if self.fit_intercept:
                    b -= eta * slope  # the intercept is not penalised
                total += loss
                t += 1
            # @a:stop
            self.loss_curve_.append(total / n)
            if self.tol is not None:
                stale = stale + 1 if self.loss_curve_[-1] > best - self.tol else 0
                best = min(best, self.loss_curve_[-1])
                if stale >= self.n_iter_no_change:
                    break
        self.coef_, self.intercept_, self.n_iter_ = w, b, epoch
        return self

    # @a:predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


# @a:sweep
def learning_rate_sweep(
    X: np.ndarray, y: np.ndarray, rates: np.ndarray, epochs: int = 5
) -> np.ndarray:
    """Training MSE after a few epochs at each constant learning rate: watch it fall, then explode."""
    mse = []
    for eta0 in rates:
        model = SGDRegressorScratch(learning_rate="constant", eta0=eta0, max_iter=epochs, tol=None)
        # A too-large rate overflows to inf/nan: that's the lesson, not an error.
        with np.errstate(over="ignore", invalid="ignore"):
            mse.append(np.mean((model.fit(X, y).predict(X) - y) ** 2))
    return np.array(mse)
