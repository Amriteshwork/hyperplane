"""Decision boundaries in NumPy: logistic regression separates two classes with a straight line."""

import numpy as np

# @a:reuse
from log_loss_numpy import log_loss
from sigmoid_numpy import sigmoid


# @a:fit-logistic
def fit_logistic(X, y, weights=None, C=1.0, tol=1e-8, max_iter=100_000):
    """Binary logistic regression (y in {0, 1}) by gradient descent; returns (w, b).

    Minimises the (row-weighted) mean log-loss + ‖w‖²/(2Cn): scikit-learn's objective, rescaled.
    """
    n, p = X.shape
    s = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    X1 = np.column_stack([X, np.ones(n)])  # the last weight is the intercept...
    penalised = np.r_[np.ones(p), 0.0]  # ...which is not penalised
    theta = np.zeros(p + 1)
    step = 1 / (0.25 * s.max() * np.linalg.norm(X1, 2) ** 2 / n + 1 / (C * n))  # never overshoots
    for _ in range(max_iter):
        gradient = X1.T @ (s * (sigmoid(X1 @ theta) - y)) / n + penalised * theta / (C * n)
        if np.abs(gradient).max() < tol:
            break
        theta -= step * gradient
    return theta[:-1], theta[-1]


# @a:demo
if __name__ == "__main__":
    # @a:data
    rng = np.random.default_rng(0)
    n = 200
    X = np.vstack([rng.normal([-1.0, -1.0], 1.0, (n, 2)), rng.normal([1.5, 1.0], 1.0, (n, 2))])
    y = np.r_[np.zeros(n), np.ones(n)]

    # @a:fit
    w, b = fit_logistic(X, y)
    p = sigmoid(X @ w + b)
    print(f"w = {w.round(3)}, b = {b:.3f}, log-loss {log_loss(y, p):.3f}, "
          f"accuracy {np.mean((p >= 0.5) == y):.3f}")

    # @a:line
    x1 = np.linspace(-3, 3, 7)
    x2 = -(w[0] * x1 + b) / w[1]  # solve w₁x₁ + w₂x₂ + b = 0 for x₂
    on_line = np.column_stack([x1, x2])
    assert np.allclose(sigmoid(on_line @ w + b), 0.5)  # on the boundary: a coin flip
    assert np.array_equal(p >= 0.5, X @ w + b >= 0)  # p ≥ 0.5 exactly where the score ≥ 0

    # @a:distance
    unit = w / np.linalg.norm(w)  # perpendicular to the line
    for d in (-2, -1, 0, 1, 2):
        print(f"{d:+} from the line: p = {sigmoid((on_line[3] + d * unit) @ w + b):.3f}")
    assert np.isclose((on_line[3] + 2 * unit) @ w + b, 2 * np.linalg.norm(w))  # log-odds = d‖w‖

    # @a:xor
    Q = rng.uniform(-1, 1, size=(400, 2))
    yq = (Q[:, 0] * Q[:, 1] > 0).astype(float)  # class 1 where the signs agree: XOR
    wq, bq = fit_logistic(Q, yq)
    flat = np.mean((Q @ wq + bq >= 0) == yq)
    expanded = np.column_stack([Q, Q[:, 0] * Q[:, 1]])  # add the product as a feature
    we, be = fit_logistic(expanded, yq, C=100.0, tol=1e-6)
    curved = np.mean((expanded @ we + be >= 0) == yq)
    print(f"XOR accuracy: straight line {flat:.2f}, with an x₁·x₂ feature {curved:.2f}")
    assert flat < 0.7 and curved > 0.95
