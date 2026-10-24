"""Activation functions in NumPy: the bend in each neuron, and the slope backpropagation needs."""

import numpy as np

# @a:reuse
from derivatives_numpy import numerical_derivative
from neurons_numpy import layer
from sigmoid_numpy import sigmoid


# @a:activations
def relu(z):
    return np.maximum(z, 0.0)


def identity(z):
    return z


# @a:slopes
# Name → (f, f′ written in terms of the output a = f(z)), as scikit-learn's MLP stores them.
ACTIVATIONS = {
    "relu": (relu, lambda a: (a > 0).astype(float)),
    "tanh": (np.tanh, lambda a: 1.0 - a**2),
    "logistic": (sigmoid, lambda a: a * (1.0 - a)),
    "identity": (identity, np.ones_like),
}


# @a:demo
if __name__ == "__main__":
    # @a:check-slopes
    z = np.array([-3.0, -0.5, 0.7, 2.0])  # avoid z = 0, where ReLU has a kink
    for name, (f, slope) in ACTIVATIONS.items():
        numeric = np.array([numerical_derivative(f, zi) for zi in z])
        assert np.allclose(slope(f(z)), numeric, atol=1e-6), name
        print(f"{name:>8}: f(z) = {f(z).round(3)}, f'(z) = {slope(f(z)).round(3)}")

    # @a:collapse
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 3))
    W1, b1, W2, b2 = rng.normal(size=(3, 4)), rng.normal(size=4), rng.normal(size=(4, 2)), rng.normal(size=2)
    two_layers = layer(layer(X, W1, b1, identity), W2, b2, identity)
    one_layer = layer(X, W1 @ W2, b1 @ W2 + b2, identity)
    assert np.allclose(two_layers, one_layer)  # no bend: two layers are one linear layer
    assert not np.allclose(layer(layer(X, W1, b1, relu), W2, b2, identity), one_layer)

    # @a:vanishing
    sigmoid_slope = ACTIVATIONS["logistic"][1]
    assert sigmoid_slope(sigmoid(0.0)) == 0.25  # the steepest the sigmoid ever gets
    for depth in (1, 5, 10):
        print(f"{depth:2d} sigmoid layers: gradient shrinks by at least {0.25 ** -depth:,.0f}×")
    assert sigmoid_slope(sigmoid(10.0)) < 1e-4  # saturated: deep in the tail, almost flat

    # @a:dead-relu
    z = X @ np.array([-1.0, -1.0, -1.0]) - 10.0  # this neuron's sum is negative for every row
    relu_slope = ACTIVATIONS["relu"][1]
    assert np.all(relu(z) == 0) and np.all(relu_slope(relu(z)) == 0)  # dead: no output, no gradient
