"""Sparsity in PyTorch: as the L1 penalty falls, features join the model one at a time."""

import torch

# @a:reuse
from l1_regularization_pytorch import lasso

# @a:data
torch.manual_seed(0)
f64 = torch.float64
n, p = 300, 6
X = torch.randn(n, p, dtype=f64)
y = X @ torch.tensor([4.0, -3.0, 2.0, 1.0, 0.5, 0.0], dtype=f64) + torch.randn(n, dtype=f64)
X, y = X - X.mean(dim=0), y - y.mean()  # centred: no intercept

# @a:alpha-max
alpha_max = ((X.T @ y).abs().max() / n).item()  # every weight is 0 from here up
print(f"alpha_max = {alpha_max:.3f}")

# @a:path
alphas = alpha_max * torch.logspace(0, -3, 13, dtype=f64)  # from all-zero to almost none
path = torch.stack([lasso(X, y, alpha.item()) for alpha in alphas])
for alpha, w in zip(alphas, path):
    print(f"α = {alpha:7.4f}  non-zero: {torch.count_nonzero(w)}  w = {w.round(decimals=2)}")

# @a:entry
first_used = [torch.nonzero(path[:, j])[:1] for j in range(p)]  # first α where j is used
order = sorted(range(p), key=lambda j: first_used[j].item() if len(first_used[j]) else len(alphas))
print("features enter in order:", order)

# @a:ridge
ridge = torch.linalg.solve(X.T @ X + 1e4 * torch.eye(p, dtype=f64), X.T @ y)  # crushing L2

# @a:checks
counts = [torch.count_nonzero(w).item() for w in path]
assert counts[0] == 0 and counts[-1] >= 5  # from nothing to (almost) everything
assert counts == sorted(counts)  # lowering α only ever added features here
assert order[:5] == [0, 1, 2, 3, 4]  # the biggest true effects enter first
assert torch.all(ridge != 0)  # L2 shrinks weights to tiny values, but never to 0
