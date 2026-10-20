"""Derivatives and gradients in PyTorch: numerical slopes, checked against autograd."""

import math

import torch


# @a:derivative
def numerical_derivative(f, x, h=1e-6):
    """f′(x) ≈ (f(x + h) − f(x − h)) / 2h: the slope across a tiny interval around x."""
    return (f(x + h) - f(x - h)) / (2 * h)


# @a:gradient
def numerical_gradient(f, w, h=1e-6):
    """∇f(w): one partial derivative per parameter, nudging one coordinate at a time."""
    w = w.detach().to(torch.float64)
    grad = torch.zeros_like(w)
    for i in range(w.numel()):
        step = torch.zeros_like(w)
        step[i] = h
        grad[i] = (f(w + step) - f(w - step)) / (2 * h)
    return grad


# @a:demo
if __name__ == "__main__":
    # @a:rules
    assert math.isclose(numerical_derivative(lambda x: x**2, 3.0), 6.0, rel_tol=1e-6)
    assert math.isclose(numerical_derivative(math.exp, 1.0), math.e, rel_tol=1e-6)
    assert math.isclose(numerical_derivative(math.sin, 0.5), math.cos(0.5), rel_tol=1e-6)

    # @a:chain
    def f(x):
        return math.sin(x**2)  # sin(g(x)) with g(x) = x²

    x = 1.3
    chain_rule = math.cos(x**2) * 2 * x  # f′(g(x)) · g′(x)
    assert math.isclose(numerical_derivative(f, x), chain_rule, rel_tol=1e-6)

    # @a:partials
    def bowl(w):
        return w[0] ** 2 + 3 * w[0] * w[1] + 2 * w[1] ** 2

    w = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
    bowl(w).backward()  # autograd: PyTorch applies the chain rule for us
    print("numerical:", numerical_gradient(bowl, w).tolist(), " autograd:", w.grad.tolist())
    assert torch.allclose(numerical_gradient(bowl, w), w.grad, atol=1e-6)

    # @a:downhill
    with torch.no_grad():
        step = 0.01 * w.grad
        assert bowl(w - step) < bowl(w) < bowl(w + step)  # against the gradient goes down

    # @a:mse-gradient
    torch.manual_seed(0)
    X, y = torch.randn(50, 3, dtype=torch.float64), torch.randn(50, dtype=torch.float64)

    def mse(w):
        return ((X @ w - y) ** 2).mean()

    w = torch.randn(3, dtype=torch.float64, requires_grad=True)
    mse(w).backward()
    analytic = 2 / len(y) * X.T @ (X @ w.detach() - y)  # the formula gradient descent uses
    assert torch.allclose(w.grad, analytic) and torch.allclose(numerical_gradient(mse, w), analytic)
