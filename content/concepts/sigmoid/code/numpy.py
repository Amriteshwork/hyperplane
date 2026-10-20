"""The sigmoid in NumPy: squash any score into a probability, and undo it with the log-odds."""

import numpy as np


# @a:sigmoid
def sigmoid(z):
    """1 / (1 + e^−z), computed so that large |z| never overflows."""
    return np.exp(-np.logaddexp(0.0, -z))


# @a:demo
if __name__ == "__main__":
    # @a:shape
    z = np.array([-10.0, -2.0, 0.0, 2.0, 10.0])
    print("z:         ", z)
    print("sigmoid(z):", sigmoid(z).round(4))
    assert sigmoid(0.0) == 0.5  # a score of 0 means a coin flip
    assert np.all((0 < sigmoid(z)) & (sigmoid(z) < 1))  # always a valid probability
    assert np.allclose(sigmoid(-z), 1 - sigmoid(z))  # symmetric around (0, 0.5)

    # @a:log-odds
    p = sigmoid(z)
    log_odds = np.log(p / (1 - p))  # the logit: sigmoid's inverse
    assert np.allclose(log_odds, z)

    # @a:derivative
    h = 1e-6
    numeric = (sigmoid(z + h) - sigmoid(z - h)) / (2 * h)
    assert np.allclose(numeric, sigmoid(z) * (1 - sigmoid(z)))  # σ' = σ(1 − σ)
    assert np.isclose(numeric.max(), 0.25)  # steepest at z = 0, slope ¼

    # @a:stable
    big = np.array([-800.0, 800.0])
    with np.errstate(over="ignore"):
        naive = 1 / (1 + np.exp(-big))  # e⁸⁰⁰ overflows (to inf; the answer survives here...)
    assert np.allclose(sigmoid(big), [0.0, 1.0]) and np.allclose(naive, [0.0, 1.0])
    print("sigmoid(±800):", sigmoid(big))  # ...but the stable form never warns
