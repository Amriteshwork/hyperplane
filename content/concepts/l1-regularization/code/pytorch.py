"""L1 regularisation (Lasso) in PyTorch: a penalty that sets weights exactly to zero."""

import torch

# @a:reuse
from normal_equation_pytorch import least_squares


# @a:soft-threshold
def soft_threshold(z, t):
    """Move z towards 0 by t, and stop at 0."""
    return torch.sign(z) * torch.clamp(z.abs() - t, min=0.0)


# @a:lasso
def lasso(X, y, alpha, tol=1e-10, max_sweeps=1000):
    """w minimising ‖y − Xw‖²/2n + α‖w‖₁, one weight at a time. Centre X and y first."""
    n, p = X.shape
    w = torch.zeros(p, dtype=X.dtype)
    for _ in range(max_sweeps):
        previous = w.clone()
        for j in range(p):
            partial = y - X @ w + X[:, j] * w[j]  # the residual without feature j
            w[j] = soft_threshold(X[:, j] @ partial / n, alpha) / (X[:, j] @ X[:, j] / n)
        if (w - previous).abs().max() < tol:  # a whole sweep changed nothing: done
            break
    return w


# @a:demo
if __name__ == "__main__":
    # @a:data
    torch.manual_seed(0)
    f64 = torch.float64
    n, p = 200, 8
    X = torch.randn(n, p, dtype=f64)
    true_w = torch.tensor([3.0, -2.0, 1.5, 0, 0, 0, 0, 0], dtype=f64)  # 3 features matter
    y = X @ true_w + torch.randn(n, dtype=f64)
    X, y = X - X.mean(dim=0), y - y.mean()  # centred, so there is no intercept to fit

    # @a:fit
    alpha = 0.25
    ols = least_squares(X, y)
    l1 = lasso(X, y, alpha)
    l2 = torch.linalg.solve(X.T @ X + 50.0 * torch.eye(p, dtype=f64), X.T @ y)  # Ridge
    print("least squares:", ols.round(decimals=3).tolist())
    print("lasso (L1):   ", l1.round(decimals=3).tolist())
    print("ridge (L2):   ", l2.round(decimals=3).tolist())

    # @a:optimality
    correlation = X.T @ (y - X @ l1) / n  # how much each feature could still explain
    kept = l1 != 0
    assert torch.allclose(correlation[kept], alpha * torch.sign(l1[kept]))  # kept: exactly ±α
    assert torch.all(correlation[~kept].abs() <= alpha)  # dropped: not worth α

    # @a:checks
    assert torch.all(l1[3:] == 0) and torch.all(l1[:3] != 0)  # exactly the useless ones are 0
    assert torch.all(l2 != 0)  # Ridge shrinks everything, but zeroes nothing
    assert torch.all(l1[:3].abs() < ols[:3].abs())  # the kept weights are shrunk, too
