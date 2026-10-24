"""Epochs in NumPy: pass over the data several times, and stop when the loss stops improving."""

import numpy as np

# @a:reuse
from mean_squared_error_numpy import mse
from stochastic_gradient_descent_numpy import sgd_epoch


# @a:stalled
def stalled(losses, tol=1e-3, patience=5):
    """scikit-learn's rule: the last `patience` epochs each failed to beat the best so far by tol."""
    best, stale = np.inf, 0
    for loss in losses:
        stale = stale + 1 if loss > best - tol else 0
        best = min(best, loss)
    return stale >= patience


# @a:demo
if __name__ == "__main__":
    # @a:data
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(size=(n, 2))
    y = 4.0 + X @ np.array([2.0, -3.0]) + rng.normal(scale=0.5, size=n)
    X_train, y_train, X_val, y_val = X[:1600], y[:1600], X[1600:], y[1600:]  # rows are already random

    # @a:loop
    w, b = np.zeros(2), 0.0
    history = []
    for epoch in range(1, 101):
        w, b = sgd_epoch(X_train, y_train, w, b, learning_rate=0.0005 / np.sqrt(epoch), rng=rng)
        history.append(mse(y_val, X_val @ w + b))  # judged on rows it never trained on
        print(f"epoch {epoch:3d}: {epoch * len(y_train):>6} updates, validation MSE {history[-1]:.4f}")
        # @a:stop
        if stalled(history):
            break
    print(f"stopped after {epoch} epochs; w = {w.round(3)}, b = {b:.3f}")
    assert epoch < 100 and history[-1] < 0.3  # stopped by itself, at the noise floor
    assert history[0] > history[-1]  # the first pass alone wasn't enough
