"""Derivatives and gradients in NumPy: slopes measured numerically, checked against calculus."""

import numpy as np


# @a:derivative
def numerical_derivative(f, x, h=1e-6):
    """f′(x) ≈ (f(x + h) − f(x − h)) / 2h: the slope across a tiny interval around x."""
    return (f(x + h) - f(x - h)) / (2 * h)


# @a:gradient
def numerical_gradient(f, w, h=1e-6):
    """∇f(w): one partial derivative per parameter, nudging one coordinate at a time."""
    w = np.asarray(w, dtype=float)
    grad = np.zeros_like(w)
    for i in range(w.size):
        step = np.zeros_like(w)
        step[i] = h
        grad[i] = (f(w + step) - f(w - step)) / (2 * h)
    return grad


# @a:demo
if __name__ == "__main__":
    # @a:rules
    assert np.isclose(numerical_derivative(lambda x: x**2, 3.0), 6.0)  # (x²)′ = 2x
    assert np.isclose(numerical_derivative(np.exp, 1.0), np.e)  # (eˣ)′ = eˣ
    assert np.isclose(numerical_derivative(np.sin, 0.5), np.cos(0.5))

    # @a:chain
    def f(x):
        return np.sin(x**2)  # sin(g(x)) with g(x) = x²

    x = 1.3
    chain_rule = np.cos(x**2) * 2 * x  # f′(g(x)) · g′(x)
    assert np.isclose(numerical_derivative(f, x), chain_rule)

    # @a:partials
    def bowl(w):
        return w[0] ** 2 + 3 * w[0] * w[1] + 2 * w[1] ** 2

    w = np.array([1.0, -2.0])
    by_hand = np.array([2 * w[0] + 3 * w[1], 3 * w[0] + 4 * w[1]])  # ∂/∂w₀, ∂/∂w₁
    print("gradient at", w, "=", numerical_gradient(bowl, w), " by hand:", by_hand)
    assert np.allclose(numerical_gradient(bowl, w), by_hand, atol=1e-6)

    # @a:downhill
    step = 0.01 * numerical_gradient(bowl, w)
    assert bowl(w - step) < bowl(w) < bowl(w + step)  # against the gradient goes down

    # @a:mse-gradient
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(50, 3)), rng.normal(size=50)

    def mse(w):
        return np.mean((X @ w - y) ** 2)

    w = rng.normal(size=3)
    analytic = 2 / len(y) * X.T @ (X @ w - y)  # the formula gradient descent uses
    assert np.allclose(numerical_gradient(mse, w), analytic, atol=1e-6)
