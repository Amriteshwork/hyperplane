"""Neurons and layers in NumPy: weighted sums and a bend, one matrix product per layer."""

import numpy as np

# @a:reuse
from sigmoid_numpy import sigmoid


# @a:layer
def layer(A, W, b, activation=sigmoid):
    """Every neuron of a layer at once, for every row: f(A @ W + b), one column per neuron."""
    return activation(A @ W + b)


# @a:demo
if __name__ == "__main__":
    # @a:one-neuron
    x = np.array([2.0, -1.0, 0.5])
    w, b = np.array([0.4, 0.3, -0.2]), 0.1
    a = sigmoid(w @ x + b)  # one neuron: weighted sum, plus bias, then the bend
    print(f"one neuron: w·x + b = {w @ x + b:.2f}, output {a:.3f}")

    # @a:shapes
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3))  # 5 rows, 3 features
    W1, b1 = rng.normal(size=(3, 4)), np.zeros(4)  # 3 inputs → 4 hidden neurons
    W2, b2 = rng.normal(size=(4, 1)), np.zeros(1)  # 4 hidden → 1 output
    H = layer(X, W1, b1)
    out = layer(H, W2, b2)
    print("hidden layer:", H.shape, "| output:", out.shape)
    assert H.shape == (5, 4) and out.shape == (5, 1)
    assert np.allclose(H[2, 1], sigmoid(X[2] @ W1[:, 1] + b1[1]))  # column j is neuron j
    assert W1.size + b1.size + W2.size + b2.size == 21  # (3 + 1)·4 + (4 + 1)·1 parameters

    # @a:xor-data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])  # XOR: 1 when exactly one input is 1

    # @a:one-line
    best = 0.0
    for w1, w2, b in rng.uniform(-5, 5, size=(20_000, 3)):  # many random straight lines
        best = max(best, np.mean((X @ np.array([w1, w2]) + b > 0) == y))
    print(f"best single neuron: {best:.0%} of the XOR points right")
    assert best == 0.75  # no straight line gets all four

    # @a:xor-network
    W1 = np.array([[20.0, 20.0], [20.0, 20.0]])  # hidden neuron 0: OR, neuron 1: AND
    b1 = np.array([-10.0, -30.0])
    W2, b2 = np.array([[20.0], [-20.0]]), np.array([-10.0])  # OR and not AND
    H = layer(X, W1, b1)
    out = layer(H, W2, b2)[:, 0]
    print("hidden (OR, AND):", H.round(2).tolist(), "| XOR:", out.round(2))
    assert np.array_equal((out > 0.5).astype(int), y)  # two layers: all four right
