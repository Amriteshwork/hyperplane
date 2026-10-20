"""Bias and variance in NumPy: refit on many fresh samples and see how predictions spread."""

import numpy as np

# @a:setup
rng = np.random.default_rng(0)
x_test = np.linspace(-0.9, 0.9, 50)
truth = np.sin(3 * x_test)  # the function every training sample comes from


# @a:refit
def predictions(degree, samples=300):
    preds = []
    for _ in range(samples):  # a fresh, noisy training set each time
        x = rng.uniform(-1, 1, size=40)
        y = np.sin(3 * x) + rng.normal(scale=0.3, size=40)
        beta = np.linalg.lstsq(np.vander(x, degree + 1), y, rcond=None)[0]
        preds.append(np.vander(x_test, degree + 1) @ beta)
    return np.array(preds)  # shape (samples, points)


# @a:decompose
def bias_variance(degree):
    preds = predictions(degree)
    bias2 = np.mean((preds.mean(axis=0) - truth) ** 2)  # how wrong on average
    variance = np.mean(preds.var(axis=0))  # how much it swings between samples
    print(f"degree {degree}: bias² {bias2:.4f}  variance {variance:.4f}")
    return bias2, variance


simple, balanced, flexible = (bias_variance(degree) for degree in (1, 3, 8))

# @a:checks
assert simple[0] > 10 * balanced[0]  # the line is wrong the same way every time
assert flexible[1] > 3 * balanced[1]  # the wiggly fit changes with every sample
assert sum(balanced) < min(sum(simple), sum(flexible))  # the sweet spot in between
