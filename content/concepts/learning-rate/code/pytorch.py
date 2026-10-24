"""Learning rates in PyTorch: an optimizer's `lr`, and a scheduler that shrinks it over time."""

import torch

# @a:data
torch.manual_seed(0)
n = 500
X = torch.randn(n, 2)
y = 4.0 + X @ torch.tensor([2.0, -3.0]) + 0.5 * torch.randn(n)
X1 = torch.cat([torch.ones(n, 1), X], dim=1)  # with the intercept's column of 1s
limit = n / torch.linalg.eigvalsh(X1.T @ X1).max()  # above this, gradient descent diverges

# @a:compare
final = {}
for factor in (0.002, 0.5, 1.05):
    w, b = torch.zeros(2, requires_grad=True), torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.SGD([w, b], lr=float(factor * limit))  # full batch: plain gradient descent
    for step in range(100):
        loss = ((X @ w + b - y) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final[factor] = ((X @ w + b - y) ** 2).mean().item()
    print(f"lr = {factor} × limit: MSE after 100 steps {final[factor]:.4g}")
assert final[0.5] < 0.3 < final[0.002]
assert final[1.05] > (y**2).mean().item()

# @a:schedule
w, b = torch.zeros(2, requires_grad=True), torch.zeros(1, requires_grad=True)
optimizer = torch.optim.SGD([w, b], lr=0.5)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: 1 / (t + 1) ** 0.5)  # η₀ / √t
for i in torch.randint(n, (20_000,)):  # one random row per step
    loss = (X[i] @ w + b - y[i]).pow(2).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()  # shrink the learning rate after every update
print(f"final lr {scheduler.get_last_lr()[0]:.4f}, w = {w.detach().numpy().round(3)}, b = {b.item():.3f}")
assert torch.allclose(w.detach(), torch.tensor([2.0, -3.0]), atol=0.1)
