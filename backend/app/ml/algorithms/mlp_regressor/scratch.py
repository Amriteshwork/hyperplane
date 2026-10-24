"""An MLP regressor from scratch with NumPy: forward pass, backpropagation, Adam, mini-batches."""

from collections.abc import Callable
from itertools import pairwise

import numpy as np  # @a:numpy

type Fn = Callable[[np.ndarray], np.ndarray]

# @a:activations
ACTIVATIONS: dict[str, Fn] = {
    "relu": lambda z: np.maximum(z, 0.0),
    "tanh": np.tanh,
    "logistic": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "identity": lambda z: z,
}
# Each slope is written in terms of the activation's output a, which the forward pass kept.
SLOPES: dict[str, Fn] = {
    "relu": lambda a: (a > 0).astype(float),
    "tanh": lambda a: 1.0 - a**2,
    "logistic": lambda a: a * (1.0 - a),
    "identity": np.ones_like,
}


# @a:init
def glorot_uniform(
    fan_in: int, fan_out: int, activation: str, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    """Weights and biases uniform in ±√(6 / (fan_in + fan_out)): Glorot & Bengio's start."""
    bound = np.sqrt((2.0 if activation == "logistic" else 6.0) / (fan_in + fan_out))
    return rng.uniform(-bound, bound, (fan_in, fan_out)), rng.uniform(-bound, bound, fan_out)


# @a:adam
class Adam:
    """Per-weight step sizes from running averages of the gradient (m) and its square (v)."""

    def __init__(
        self,
        params: list[np.ndarray],
        learning_rate: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.learning_rate, self.beta_1, self.beta_2, self.epsilon = (
            learning_rate,
            beta_1,
            beta_2,
            epsilon,
        )
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        # Bias correction: m and v start at 0, so early averages are too small without it.
        lr = self.learning_rate * np.sqrt(1 - self.beta_2**self.t) / (1 - self.beta_1**self.t)
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * grad
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * grad**2
            param -= lr * self.m[i] / (np.sqrt(self.v[i]) + self.epsilon)  # in place


# @a:class
class MLPRegressorScratch:
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100,),
        activation: str = "relu",
        alpha: float = 0.0001,
        batch_size: int = 200,
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        tol: float = 1e-4,
        n_iter_no_change: int = 10,
        random_state: int = 0,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.coefs_: list[np.ndarray] = []
        self.intercepts_: list[np.ndarray] = []
        self.loss_curve_: list[float] = []
        self.n_iter_ = 0

    # @a:forward
    def _forward(self, X: np.ndarray) -> list[np.ndarray]:
        """Every layer's output, from the inputs to the prediction."""
        bend = ACTIVATIONS[self.activation]
        outputs = [X]
        for layer, (W, b) in enumerate(zip(self.coefs_, self.intercepts_)):
            z = outputs[-1] @ W + b  # each neuron: a weighted sum of the layer below
            last = layer == len(self.coefs_) - 1
            outputs.append(z if last else bend(z))  # the output stays linear: any number
        return outputs

    # @a:backprop
    def _backprop(self, X: np.ndarray, Y: np.ndarray) -> tuple[float, list[np.ndarray]]:
        """The batch's loss, and its gradient for every weight, by the chain rule."""
        outputs = self._forward(X)
        n = len(X)
        penalty = sum(np.dot(W.ravel(), W.ravel()) for W in self.coefs_)
        loss = 0.5 * np.mean((outputs[-1] - Y) ** 2) + 0.5 * self.alpha * penalty / n
        # @a:delta
        delta = outputs[-1] - Y  # ∂loss/∂z at the output (× n): just the error
        coef_grads = [np.empty(0)] * len(self.coefs_)
        intercept_grads = [np.empty(0)] * len(self.coefs_)
        for layer in reversed(range(len(self.coefs_))):
            coef_grads[layer] = (outputs[layer].T @ delta + self.alpha * self.coefs_[layer]) / n
            intercept_grads[layer] = delta.sum(axis=0) / n
            if layer > 0:  # pass the error back through the weights and the activation's slope
                delta = (delta @ self.coefs_[layer].T) * SLOPES[self.activation](outputs[layer])
        return loss, coef_grads + intercept_grads

    # @a:fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPRegressorScratch:
        X, Y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1, 1)
        n = len(X)
        # NumPy's legacy generator, drawn in scikit-learn's order, so both start alike.
        rng = np.random.RandomState(self.random_state)
        sizes = [X.shape[1], *self.hidden_layer_sizes, 1]
        layers = [glorot_uniform(a, b, self.activation, rng) for a, b in pairwise(sizes)]
        self.coefs_ = [W for W, _ in layers]
        self.intercepts_ = [b for _, b in layers]
        params = self.coefs_ + self.intercepts_  # the same arrays: Adam updates them in place
        adam = Adam(params, self.learning_rate_init)
        batch = min(self.batch_size, n)
        order, best, stale = np.arange(n), np.inf, 0
        self.loss_curve_ = []
        for epoch in range(1, self.max_iter + 1):
            # @a:epoch
            order = order[rng.permutation(n)]  # a fresh shuffle every epoch
            total = 0.0
            for start in range(0, n, batch):
                rows = order[start : start + batch]
                loss, grads = self._backprop(X[rows], Y[rows])
                adam.step(params, grads)
                total += loss * len(rows)
            self.loss_curve_.append(total / n)
            # @a:stop
            stale = stale + 1 if self.loss_curve_[-1] > best - self.tol else 0
            best = min(best, self.loss_curve_[-1])
            if stale > self.n_iter_no_change:
                break
        self.n_iter_ = epoch
        return self

    # @a:predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._forward(np.asarray(X, dtype=float))[-1].ravel()
