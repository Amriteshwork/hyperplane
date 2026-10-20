"""Combined penalties in PyTorch: L1 drops features, L2 keeps correlated ones together."""

import torch

# @a:reuse
from l1_regularization_pytorch import lasso, soft_threshold


# @a:elastic-net
def elastic_net(X, y, alpha, l1_ratio=0.5, tol=1e-10, max_sweeps=1000):
    """w minimising ‖y − Xw‖²/2n + αρ‖w‖₁ + α(1 − ρ)‖w‖²/2, ρ = l1_ratio. Centre X and y first."""
    n, p = X.shape
    l1, l2 = alpha * l1_ratio, alpha * (1 - l1_ratio)
    w = torch.zeros(p, dtype=X.dtype)
    for _ in range(max_sweeps):
        previous = w.clone()
        for j in range(p):
            partial = y - X @ w + X[:, j] * w[j]  # the residual without feature j
            w[j] = soft_threshold(X[:, j] @ partial / n, l1) / (X[:, j] @ X[:, j] / n + l2)
        if (w - previous).abs().max() < tol:  # a whole sweep changed nothing: done
            break
    return w


# @a:demo
if __name__ == "__main__":
    # @a:data
    torch.manual_seed(0)
    f64 = torch.float64
    n = 200
    signal = torch.randn(n, dtype=f64)
    group = signal[:, None] + 0.1 * torch.randn(n, 3, dtype=f64)  # three near-copies of one signal
    X = torch.hstack([group, torch.randn(n, 5, dtype=f64)])  # ...plus five useless features
    y = 3 * signal + torch.randn(n, dtype=f64)
    X, y = X - X.mean(dim=0), y - y.mean()  # centred, so there is no intercept to fit

    # @a:fit
    l1_part, l2_part = 0.5, 0.5
    only_l1 = lasso(X, y, l1_part)
    alpha = l1_part + l2_part  # same L1 part as the Lasso, plus an L2 part
    mixed = elastic_net(X, y, alpha, l1_ratio=l1_part / alpha)
    print("lasso:      ", only_l1.round(decimals=2).tolist())
    print("elastic net:", mixed.round(decimals=2).tolist())

    # @a:optimality
    kept = mixed != 0
    pull = X.T @ (y - X @ mixed) / len(y) - l2_part * mixed  # data's pull minus the L2 pull
    assert torch.allclose(pull[kept], l1_part * torch.sign(mixed[kept]))  # kept: balanced by L1
    assert torch.all(pull[~kept].abs() <= l1_part)  # dropped: not worth the L1 price

    # @a:checks
    spread = (mixed[:3].max() - mixed[:3].min(), only_l1[:3].max() - only_l1[:3].min())
    assert torch.all(only_l1[3:] == 0) and torch.all(mixed[3:] == 0)  # both drop the noise...
    assert torch.all(mixed[:3] > 0)  # ...Elastic Net keeps the whole group...
    assert spread[0] < spread[1] / 3  # ...and shares its weight evenly
