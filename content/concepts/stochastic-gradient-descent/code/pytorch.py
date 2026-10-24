"""Stochastic gradient descent in PyTorch: torch.optim.SGD fed one mini-batch at a time."""

import torch

# @a:data
torch.manual_seed(0)
n = 10_000
X = torch.randn(n, 2)
y = 4.0 + X @ torch.tensor([2.0, -3.0]) + 0.5 * torch.randn(n)

# @a:batches
results = {}
for batch_size in (1, 32, n):
    w, b = torch.zeros(2, requires_grad=True), torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.SGD([w, b], lr=0.01)
    for rows in torch.randperm(n).split(batch_size):  # one shuffled pass, in batches
        loss = ((X[rows] @ w + b - y[rows]) ** 2).mean()  # MSE on this batch only
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        results[batch_size] = ((X @ w + b - y) ** 2).mean().item()
    print(f"batch {batch_size:>5}: {-(-n // batch_size):>5} steps in one pass, MSE {results[batch_size]:.4f}")
assert results[1] < 0.3 and results[32] < 0.3
assert results[n] > 10
