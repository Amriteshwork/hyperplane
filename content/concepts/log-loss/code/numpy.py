"""Log-loss in NumPy: score probabilities by how likely they made the true labels."""

import numpy as np

# @a:reuse
from sigmoid_numpy import sigmoid


# @a:log-loss
def log_loss(y, p, eps=1e-15):
    """Mean of −[y log p + (1 − y) log(1 − p)]: the negative log-likelihood per row."""
    p = np.clip(p, eps, 1 - eps)  # log(0) would be −∞
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# @a:demo
if __name__ == "__main__":
    # @a:confidence
    for p in (0.9, 0.6, 0.4, 0.1, 0.01):
        print(f"true label 1, predicted {p:<4}: log-loss {log_loss(1, p):.3f}, "
              f"squared error {(1 - p) ** 2:.3f}")
    assert log_loss(1, 0.01) > 4 > (1 - 0.01) ** 2  # confident and wrong: punished hard

    # @a:likelihood
    y = np.array([1, 0, 1, 1, 0])
    p = np.array([0.8, 0.3, 0.6, 0.9, 0.2])
    likelihood = np.prod(np.where(y == 1, p, 1 - p))  # chance of exactly these labels
    assert np.isclose(log_loss(y, p), -np.log(likelihood) / len(y))  # = −log(likelihood)/n

    # @a:data
    rng = np.random.default_rng(0)
    n = 500
    X = rng.normal(size=(n, 2))
    y = (rng.uniform(size=n) < sigmoid(X @ np.array([2.0, -1.0]) + 0.5)).astype(float)

    # @a:gradient
    w, b = np.array([0.3, 0.2]), -0.1
    error = sigmoid(X @ w + b) - y  # predicted probability minus truth
    grad_w, grad_b = X.T @ error / n, error.mean()
    h = 1e-6
    numeric = [(log_loss(y, sigmoid(X @ (w + h * e) + b)) - log_loss(y, sigmoid(X @ (w - h * e) + b)))
               / (2 * h) for e in np.eye(2)]
    assert np.allclose(grad_w, numeric, atol=1e-6)  # ∂/∂w = Xᵀ(p − y)/n, like MSE's gradient
    print("gradient:", grad_w.round(4), grad_b.round(4))
