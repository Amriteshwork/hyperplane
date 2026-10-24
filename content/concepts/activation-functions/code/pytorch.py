"""Activation functions in PyTorch: autograd finds each slope; check them against the formulas."""

import torch

# @a:check-slopes
z = torch.tensor([-3.0, -0.5, 0.7, 2.0], requires_grad=True)
for name, f, slope in [
    ("relu", torch.relu, lambda a: (a > 0).float()),
    ("tanh", torch.tanh, lambda a: 1 - a**2),
    ("logistic", torch.sigmoid, lambda a: a * (1 - a)),
]:
    a = f(z)
    (grad,) = torch.autograd.grad(a.sum(), z)  # ∂a/∂z for every entry
    assert torch.allclose(grad, slope(a.detach())), name
    print(f"{name:>8}: f'(z) = {grad.numpy().round(3)}")

# @a:collapse
torch.manual_seed(0)
X = torch.randn(6, 3)
first, second = torch.nn.Linear(3, 4), torch.nn.Linear(4, 2)
with torch.no_grad():
    merged = X @ (first.weight.T @ second.weight.T) + (first.bias @ second.weight.T + second.bias)
    assert torch.allclose(second(first(X)), merged, atol=1e-6)  # no bend: one linear layer
    assert not torch.allclose(second(torch.relu(first(X))), merged, atol=1e-6)

# @a:vanishing
x = torch.tensor(0.0, requires_grad=True)
out = x
for _ in range(10):
    out = torch.sigmoid(out)  # ten sigmoid "layers" in a row
out.backward()
print(f"gradient through 10 sigmoids: {x.grad.item():.2e}")
assert x.grad.item() < 0.25**10 * 2  # each layer multiplies the gradient by at most ¼
