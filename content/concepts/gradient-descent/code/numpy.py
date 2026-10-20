"""Gradient descent in NumPy: fit a linear regression by repeatedly stepping downhill."""

import numpy as np

# @a:data
rng = np.random.default_rng(0)
n = 500
X = rng.normal(size=(n, 2))  # already standardised: mean 0, variance 1
y = 4.0 + X @ np.array([2.0, -3.0]) + rng.normal(scale=0.5, size=n)

# @a:init
w = np.zeros(2)
b = 0.0
learning_rate = 0.05  # η: too big and the loss explodes, too small and it crawls
# @a:loop
for step in range(300):
    error = X @ w + b - y  # prediction minus truth
    loss = np.mean(error**2)  # MSE
    # @a:gradient
    grad_w = 2 * X.T @ error / n  # ∂MSE/∂w
    grad_b = 2 * error.mean()  # ∂MSE/∂b
    # @a:update
    w -= learning_rate * grad_w  # step against the gradient
    b -= learning_rate * grad_b
    if step % 30 == 0:
        print(f"step {step:3d}  loss {loss:.4f}")

# @a:compare
# It lands where the normal equation jumps to in one step.
X1 = np.column_stack([np.ones(n), X])
exact = np.linalg.lstsq(X1, y, rcond=None)[0]
print("gradient descent:", np.round([b, *w], 4))
print("normal equation: ", exact.round(4))
assert np.allclose([b, *w], exact, atol=1e-4)
