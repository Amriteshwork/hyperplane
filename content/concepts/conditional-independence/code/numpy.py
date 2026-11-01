"""Conditional independence in NumPy: when multiplying evidence is exact, and when it overstates."""

import numpy as np

# @a:reuse
from bayes_theorem_numpy import posterior
from likelihood_numpy import log_normal_pdf


# @a:naive
def naive_posterior(x, priors, means, variances):
    """P(class | row), treating features as independent given the class: logs add, then Bayes."""
    joint = np.log(priors) + log_normal_pdf(x[:, None, :], means, variances).sum(axis=2)
    scaled = np.exp(joint - joint.max(axis=1, keepdims=True))
    return posterior(np.ones_like(scaled), scaled)


# @a:demo
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # @a:coins
    n = 2_000_000
    two_headed = rng.random(n) < 0.5
    first = two_headed | (rng.random(n) < 0.5)
    second = two_headed | (rng.random(n) < 0.5)
    print(f"overall: P(both heads) = {np.mean(first & second):.3f}, P(heads)² = {np.mean(first) * np.mean(second):.3f}")
    fair = ~two_headed
    print(f"fair coin: P(both heads) = {np.mean((first & second)[fair]):.3f}, P(heads)² = {np.mean(first[fair]) * np.mean(second[fair]):.3f}")
    assert abs(np.mean(first & second) - 5 / 8) < 0.002  # dependent: 5/8 is not (3/4)²
    assert abs(np.mean((first & second)[fair]) - 1 / 4) < 0.002  # independent given the coin

    # @a:data
    priors = np.array([0.5, 0.5])
    means, variances = np.array([[-0.5] * 2, [0.5] * 2]), np.ones((2, 2))

    def sample(rows, copies=1):
        """Rows of one informative feature, repeated `copies` times."""
        y = rng.integers(0, 2, rows)
        x = rng.normal(means[y, 0], 1.0)[:, None]
        return np.repeat(x, copies, axis=1), y

    # @a:exact
    y = rng.integers(0, 2, 5)
    x = rng.normal(means[y], 1.0)
    product = np.exp(log_normal_pdf(x[:, None, :], means, variances).sum(axis=2))
    independent_density = np.array(
        [[np.linalg.det(2 * np.pi * np.eye(2)) ** -0.5 * np.exp(-0.5 * np.sum((row - m) ** 2)) for m in means] for row in x]
    )
    assert np.allclose(product, independent_density)  # with truly independent features it's exact

    # @a:double-count
    for copies in [1, 10]:
        x, y = sample(100_000, copies)
        m, v = np.repeat(means[:, :1], copies, axis=1), np.ones((2, copies))
        probabilities = naive_posterior(x, priors, m, v)
        confidence, accuracy = probabilities.max(axis=1).mean(), np.mean(probabilities.argmax(axis=1) == y)
        print(f"{copies:2d} copies: accuracy {accuracy:.3f}, average confidence {confidence:.3f}")
        if copies == 1:
            honest, baseline = confidence, accuracy
            assert abs(confidence - accuracy) < 0.01  # one copy: the probabilities are right
    assert confidence > accuracy + 0.1 and confidence > honest + 0.1  # sure of itself, no better
    assert abs(accuracy - baseline) < 0.01  # the labels barely change

    # @a:xor
    x = rng.choice([-1.0, 1.0], size=(100_000, 2)) + rng.normal(0, 0.2, size=(100_000, 2))
    y = (x[:, 0] * x[:, 1] > 0).astype(int)
    m = np.array([x[y == c].mean(axis=0) for c in (0, 1)])
    v = np.array([x[y == c].var(axis=0) for c in (0, 1)])
    accuracy = np.mean(naive_posterior(x, priors, m, v).argmax(axis=1) == y)
    print(f"XOR: accuracy {accuracy:.3f}: each feature alone says nothing about the class")
    assert abs(accuracy - 0.5) < 0.02
