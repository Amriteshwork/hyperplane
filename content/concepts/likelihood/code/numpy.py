"""Likelihood in NumPy: score parameters against fixed data, and check the closed-form best fit."""

import numpy as np

# @a:reuse
from probability_numpy import normal_pdf


# @a:log-pdf
def log_normal_pdf(x, mu=0.0, variance=1.0):
    """The log of the bell curve's height: safe where the height itself would round to 0."""
    return -0.5 * np.log(2 * np.pi * variance) - (x - mu) ** 2 / (2 * variance)


# @a:log-likelihood
def log_likelihood(x, mu, variance):
    """How well N(mu, variance) explains every row of x, assuming the rows are independent."""
    return np.sum(log_normal_pdf(x, mu, variance))


# @a:fit
def fit_gaussian(x):
    """The maximum-likelihood mean and variance: just the sample's own."""
    return np.mean(x), np.mean((x - np.mean(x)) ** 2)


# @a:demo
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(10.0, 2.0, size=200)

    # @a:same-formula
    assert np.isclose(log_normal_pdf(11.0, 10.0, 4.0), np.log(normal_pdf(11.0, 10.0, 2.0)))

    # @a:compare
    mu, variance = fit_gaussian(x)
    print(f"best fit: mu = {mu:.3f}, variance = {variance:.3f}, log likelihood = {log_likelihood(x, mu, variance):.1f}")
    print(f"a worse guess, mu = 8: log likelihood = {log_likelihood(x, 8.0, variance):.1f}")

    # @a:search
    mus, variances = np.linspace(8, 12, 201), np.linspace(2, 7, 201)
    grid = [[log_likelihood(x, m, v) for v in variances] for m in mus]
    i, j = np.unravel_index(np.argmax(grid), (201, 201))
    print(f"grid search found: mu = {mus[i]:.2f}, variance = {variances[j]:.3f}")
    assert abs(mus[i] - mu) <= 0.02 and abs(variances[j] - variance) <= 0.025

    # @a:underflow
    many = rng.normal(10.0, 2.0, size=5_000)
    product = np.prod(normal_pdf(many, 10.0, 2.0))
    print(f"product of {len(many):,} densities: {product}; sum of their logs: {log_likelihood(many, 10.0, 4.0):.1f}")
    assert product == 0.0 and np.isfinite(log_likelihood(many, 10.0, 4.0))

    # @a:mse
    guesses = np.array([8.0, 9.0, 10.5, 12.0])
    errors = np.array([np.mean((x - m) ** 2) for m in guesses])
    scores = np.array([log_likelihood(x, m, 4.0) for m in guesses])
    assert np.allclose(scores, -len(x) * (0.5 * np.log(2 * np.pi * 4.0) + errors / (2 * 4.0)))

    # @a:spike
    constant = np.full(50, 3.0)
    for floor in [1e-9, 0.01]:
        fitted = fit_gaussian(constant)[1] + floor  # a variance of 0 would divide by zero
        print(f"variance {fitted:g}: at the value {log_normal_pdf(3.0, 3.0, fitted):.1f}, one away {log_normal_pdf(4.0, 3.0, fitted):.1f}")
    assert log_normal_pdf(4.0, 3.0, 1e-9) < -1e8 < -100 < log_normal_pdf(4.0, 3.0, 0.01)
