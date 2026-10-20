"""Decision boundaries in PyTorch: logistic regression separates two classes with a straight line."""

import torch

# @a:reuse
from log_loss_pytorch import log_loss
from sigmoid_pytorch import sigmoid


# @a:fit-logistic
def fit_logistic(X, y, weights=None, C=1.0, tol=1e-8, max_iter=100_000):
    """Binary logistic regression (y in {0, 1}) by gradient descent; returns (w, b).

    Minimises the (row-weighted) mean log-loss + ‖w‖²/(2Cn): scikit-learn's objective, rescaled.
    """
    n, p = X.shape
    s = torch.ones(n, dtype=X.dtype) if weights is None else weights
    X1 = torch.cat([X, torch.ones(n, 1, dtype=X.dtype)], dim=1)  # the last weight is the intercept...
    penalised = torch.cat([torch.ones(p, dtype=X.dtype), torch.zeros(1, dtype=X.dtype)])  # ...unpenalised
    theta = torch.zeros(p + 1, dtype=X.dtype)
    step = 1 / (0.25 * s.max() * torch.linalg.matrix_norm(X1, 2) ** 2 / n + 1 / (C * n))
    for _ in range(max_iter):
        gradient = X1.T @ (s * (sigmoid(X1 @ theta) - y)) / n + penalised * theta / (C * n)
        if gradient.abs().max() < tol:
            break
        theta -= step * gradient
    return theta[:-1], theta[-1]


# @a:demo
if __name__ == "__main__":
    # @a:data
    torch.manual_seed(0)
    f64 = torch.float64
    n = 200
    X = torch.cat([torch.randn(n, 2, dtype=f64) + torch.tensor([-1.0, -1.0], dtype=f64),
                   torch.randn(n, 2, dtype=f64) + torch.tensor([1.5, 1.0], dtype=f64)])
    y = torch.cat([torch.zeros(n, dtype=f64), torch.ones(n, dtype=f64)])

    # @a:fit
    w, b = fit_logistic(X, y)
    p = sigmoid(X @ w + b)
    print(f"w = {w.round(decimals=3).tolist()}, b = {b.item():.3f}, "
          f"log-loss {log_loss(y, p).item():.3f}, accuracy {((p >= 0.5) == y).double().mean().item():.3f}")

    # @a:line
    x1 = torch.linspace(-3, 3, 7, dtype=f64)
    x2 = -(w[0] * x1 + b) / w[1]  # solve w₁x₁ + w₂x₂ + b = 0 for x₂
    on_line = torch.stack([x1, x2], dim=1)
    assert torch.allclose(sigmoid(on_line @ w + b), torch.full((7,), 0.5, dtype=f64))  # a coin flip
    assert torch.equal(p >= 0.5, X @ w + b >= 0)  # p ≥ 0.5 exactly where the score ≥ 0

    # @a:distance
    unit = w / torch.linalg.norm(w)  # perpendicular to the line
    for d in (-2, -1, 0, 1, 2):
        print(f"{d:+} from the line: p = {sigmoid((on_line[3] + d * unit) @ w + b).item():.3f}")
    assert torch.isclose((on_line[3] + 2 * unit) @ w + b, 2 * torch.linalg.norm(w))  # log-odds = d‖w‖

    # @a:xor
    Q = 2 * torch.rand(400, 2, dtype=f64) - 1
    yq = (Q[:, 0] * Q[:, 1] > 0).to(f64)  # class 1 where the signs agree: XOR
    wq, bq = fit_logistic(Q, yq)
    flat = ((Q @ wq + bq >= 0) == yq).double().mean().item()
    expanded = torch.cat([Q, (Q[:, 0] * Q[:, 1])[:, None]], dim=1)  # add the product as a feature
    we, be = fit_logistic(expanded, yq, C=100.0, tol=1e-6)
    curved = ((expanded @ we + be >= 0) == yq).double().mean().item()
    print(f"XOR accuracy: straight line {flat:.2f}, with an x₁·x₂ feature {curved:.2f}")
    assert flat < 0.7 and curved > 0.95
