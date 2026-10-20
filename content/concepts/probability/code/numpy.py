"""Probability in NumPy: mean, variance and the normal distribution, checked by simulation."""

import numpy as np


# @a:moments
def mean(x):
    """The sample mean: the centre of the data."""
    return np.sum(x) / len(x)


def variance(x, ddof=0):
    """The average squared distance from the mean; ddof=1 gives the unbiased sample estimate."""
    return np.sum((x - mean(x)) ** 2) / (len(x) - ddof)


# @a:pdf
def normal_pdf(x, mu=0.0, sigma=1.0):
    """The bell curve: how dense the normal distribution N(μ, σ²) is at x."""
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


# @a:demo
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # @a:sample
    mu, sigma = 10.0, 2.0
    x = rng.normal(mu, sigma, size=100_000)
    print(f"sample mean {mean(x):.3f} (true {mu}),  sample std {np.sqrt(variance(x)):.3f} (true {sigma})")
    assert abs(mean(x) - mu) < 0.05 and abs(np.sqrt(variance(x)) - sigma) < 0.05
    assert np.isclose(mean(x), x.mean()) and np.isclose(variance(x, ddof=1), x.var(ddof=1))

    # @a:rule
    for k, expected in [(1, 0.683), (2, 0.954), (3, 0.997)]:
        share = np.mean(np.abs(x - mu) < k * sigma)
        print(f"within {k}σ: {share:.3f}")
        assert abs(share - expected) < 0.01

    # @a:density
    grid = np.linspace(mu - 6 * sigma, mu + 6 * sigma, 10_001)
    area = np.sum(normal_pdf(grid, mu, sigma)) * (grid[1] - grid[0])
    assert np.isclose(area, 1.0, atol=1e-6)  # a density's total area is 1
    counts, edges = np.histogram(x, bins=40, range=(mu - 4 * sigma, mu + 4 * sigma))
    centres, width = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
    assert np.allclose(counts / (len(x) * width), normal_pdf(centres, mu, sigma), atol=0.01)

    # @a:large-numbers
    for n in [10, 1_000, 100_000]:
        print(f"mean of {n:>7,} dice rolls: {mean(rng.integers(1, 7, size=n)):.3f}  (true 3.5)")
    assert abs(mean(rng.integers(1, 7, size=100_000)) - 3.5) < 0.03

    # @a:clt
    sums = rng.uniform(size=(20_000, 30)).sum(axis=1)  # each a sum of 30 flat random numbers
    z = (sums - 15) / np.sqrt(30 / 12)  # standardise with the sum's true mean and std
    assert abs(np.mean(np.abs(z) < 1) - 0.683) < 0.01  # behaves like a normal already
