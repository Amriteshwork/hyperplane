"""Bayes' theorem in NumPy: update a prior with a likelihood, checked by counting simulated people."""

import numpy as np

# @a:reuse
from probability_numpy import normal_pdf


# @a:posterior
def posterior(prior, likelihood):
    """P(class | evidence) from P(class) and P(evidence | class); the last axis is the classes."""
    joint = prior * likelihood
    return joint / joint.sum(axis=-1, keepdims=True)


# @a:demo
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # @a:test
    prior = np.array([0.99, 0.01])  # healthy, sick
    positive = np.array([0.05, 0.90])  # P(positive | healthy), P(positive | sick)
    sick_given_positive = posterior(prior, positive)[1]
    print(f"P(sick | positive) = {sick_given_positive:.3f}, though P(positive | sick) = {positive[1]}")
    assert np.isclose(sick_given_positive, 0.009 / (0.009 + 0.0495))

    # @a:simulate
    n = 2_000_000
    sick = rng.random(n) < prior[1]
    tested_positive = rng.random(n) < positive[sick.astype(int)]
    counted = sick[tested_positive].mean()
    print(f"share of simulated positives who are sick: {counted:.3f}")
    assert abs(counted - sick_given_positive) < 0.003

    # @a:update
    twice = posterior(posterior(prior, positive), positive)[1]
    print(f"after a second independent positive: {twice:.3f}")
    assert np.isclose(twice, posterior(prior, positive**2)[1])  # the same as both at once

    # @a:continuous
    prior = np.array([0.5, 0.5])
    means, sigma = np.array([-1.0, 1.0]), 1.0
    x = np.concatenate([rng.normal(means[0], sigma, n // 2), rng.normal(means[1], sigma, n // 2)])
    labels = np.repeat([0, 1], n // 2)
    near = np.abs(x - 0.5) < 0.01  # rows that landed at about x = 0.5
    at_half = posterior(prior, normal_pdf(0.5, means, sigma))[1]
    print(f"P(class 1 | x = 0.5) = {at_half:.3f}; share among rows near it: {labels[near].mean():.3f}")
    assert abs(labels[near].mean() - at_half) < 0.01
