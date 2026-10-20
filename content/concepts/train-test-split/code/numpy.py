"""Train/test split in NumPy: hold out rows to measure the model on data it never saw."""

import numpy as np

# @a:data
rng = np.random.default_rng(42)  # fixed seed: same split, same numbers, every run
n = 30  # small on purpose: flexible models overfit small datasets
x = rng.uniform(-1, 1, size=n)
y = np.sin(3 * x) + rng.normal(scale=0.2, size=n)


# @a:split
def train_test_split(n_rows, test_size, rng):
    order = rng.permutation(n_rows)  # shuffle, so both parts look like the whole dataset
    n_test = round(n_rows * test_size)
    return order[n_test:], order[:n_test]


# @a:check
train, test = train_test_split(n, test_size=0.3, rng=rng)
assert len(test) == 9
assert not set(train) & set(test)  # no row is in both


# @a:score
def mse(coefficients, rows):
    return np.mean((np.polyval(coefficients, x[rows]) - y[rows]) ** 2)


# @a:fit
# Fit on the training rows only; score on both.
scores = {}
for degree in (1, 3, 12):
    coefficients = np.polyfit(x[train], y[train], degree)
    scores[degree] = (mse(coefficients, train), mse(coefficients, test))
    print(f"degree {degree:2d}: train MSE {scores[degree][0]:.3f}   test MSE {scores[degree][1]:.3f}")

# @a:verdict
# More flexible models always fit the training rows better...
assert scores[1][0] > scores[3][0] > scores[12][0]
# ...but only the test rows reveal that degree 12 memorised noise.
assert scores[12][1] > scores[3][1]
