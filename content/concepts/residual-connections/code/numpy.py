"""Residual connections in NumPy: x + f(x) gives the gradient a highway through a deep network."""

import numpy as np

# @a:reuse
from activation_functions_numpy import ACTIVATIONS
from derivatives_numpy import numerical_gradient

tanh, tanh_slope = ACTIVATIONS["tanh"]


# @a:residual
def residual(x, f):
    """Add the layer's output to its input: the layer only has to learn a correction."""
    return x + f(x)


# @a:deep
def forward(X, weights, skip):
    """Every layer's output through a stack of tanh layers, with or without residual skips."""
    outputs = [X]
    for W in weights:
        h = outputs[-1]
        outputs.append(residual(h, lambda h: tanh(h @ W)) if skip else tanh(h @ W))
    return outputs


def input_gradient(outputs, weights, skip):
    """Backprop ∂(sum of outputs)/∂input; a skip passes the gradient through untouched."""
    grad = np.ones_like(outputs[-1])
    for W, h_in, h_out in zip(reversed(weights), reversed(outputs[:-1]), reversed(outputs[1:])):
        a = h_out - h_in if skip else h_out  # the tanh branch's own output
        through_layer = (grad * tanh_slope(a)) @ W.T
        grad = grad + through_layer if skip else through_layer
    return grad


# @a:demo
if __name__ == "__main__":
    # @a:identity
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 8))
    assert np.array_equal(residual(x, lambda h: 0 * h), x)  # a silent layer is a perfect identity

    # @a:check
    few = [rng.normal(size=(8, 8)) / np.sqrt(8) for _ in range(3)]
    for skip in (False, True):
        exact = input_gradient(forward(x, few, skip), few, skip)
        numeric = numerical_gradient(lambda v: forward(v.reshape(x.shape), few, skip)[-1].sum(), x.ravel())
        assert np.allclose(exact, numeric.reshape(x.shape), atol=1e-6)

    # @a:vanishing
    d = 32
    X = rng.normal(size=(16, d))
    top = np.linalg.norm(np.ones_like(X))  # the gradient's size at the output
    for depth in (2, 10, 50):
        weights = [rng.normal(size=(d, d)) * 0.3 / np.sqrt(d) for _ in range(depth)]
        plain = np.linalg.norm(input_gradient(forward(X, weights, False), weights, False)) / top
        skip = np.linalg.norm(input_gradient(forward(X, weights, True), weights, True)) / top
        print(f"{depth:2d} layers: gradient reaching the input, plain {plain:8.1e} | residual {skip:.2f}")
    assert plain < 1e-20 and 0.5 < skip < 10  # plain: the first layers can't learn; residual: fine
