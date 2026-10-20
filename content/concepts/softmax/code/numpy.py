"""Softmax in NumPy: turn one score per class into probabilities that add up to 1."""

import numpy as np

# @a:reuse
from logarithms_exponentials_numpy import logsumexp
from sigmoid_numpy import sigmoid


# @a:softmax
def softmax(Z):
    """Row-wise e^z / Σ e^z, via log-softmax so that no exponent overflows."""
    return np.exp(Z - logsumexp(Z, axis=1)[:, None])


# @a:demo
if __name__ == "__main__":
    # @a:scores
    Z = np.array([[2.0, 1.0, 0.1], [0.0, 0.0, 0.0], [1000.0, 1001.0, 999.0]])
    P = softmax(Z)
    print(P.round(3))

    # @a:checks
    assert np.allclose(P.sum(axis=1), 1) and np.all(P > 0)  # a probability per class
    assert np.array_equal(P.argmax(axis=1), Z.argmax(axis=1))  # biggest score, biggest share
    assert np.allclose(P[1], 1 / 3)  # equal scores, equal chances
    assert np.allclose(softmax(Z + 5.0), P)  # only differences between scores matter

    # @a:two-classes
    z = np.linspace(-4, 4, 9)
    two = softmax(np.column_stack([np.zeros_like(z), z]))
    assert np.allclose(two[:, 1], sigmoid(z))  # two classes: softmax is the sigmoid
