"""Epochs in PyTorch: a DataLoader reshuffles every pass; stop when validation stops improving."""

import torch
from torch.utils.data import DataLoader, TensorDataset

# @a:data
torch.manual_seed(0)
n = 2000
X = torch.randn(n, 2)
y = 4.0 + X @ torch.tensor([2.0, -3.0]) + 0.5 * torch.randn(n)
train = DataLoader(TensorDataset(X[:1600], y[:1600]), batch_size=16, shuffle=True)
X_val, y_val = X[1600:], y[1600:]

# @a:loop
w, b = torch.zeros(2, requires_grad=True), torch.zeros(1, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.002)
history, best, stale = [], float("inf"), 0
for epoch in range(1, 101):
    for X_batch, y_batch in train:  # one epoch: every training row once, in a new order
        loss = ((X_batch @ w + b - y_batch) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        history.append(((X_val @ w + b - y_val) ** 2).mean().item())
    print(f"epoch {epoch:3d}: {epoch * len(train)} updates, validation MSE {history[-1]:.4f}")
    # @a:stop
    stale = stale + 1 if history[-1] > best - 1e-3 else 0
    best = min(best, history[-1])
    if stale >= 5:
        break
print(f"stopped after {epoch} epochs; w = {w.detach().numpy().round(3)}, b = {b.item():.3f}")
assert epoch < 100 and history[-1] < 0.3
assert history[0] > history[-1]
