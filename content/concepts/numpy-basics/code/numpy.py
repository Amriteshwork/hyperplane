"""NumPy basics: arrays, shapes, vectorised maths, axis reductions and broadcasting."""

import numpy as np

# @a:arrays
x = np.array([1.0, 2.0, 3.0, 4.0])
X = np.arange(12, dtype=float).reshape(4, 3)  # 4 rows, 3 columns: a tiny dataset
print("x.shape =", x.shape, "  X.shape =", X.shape)

# @a:vectorised
doubled = x * 2  # every element at once, no loop
looped = np.array([value * 2 for value in x])  # the same, the slow way
assert np.array_equal(doubled, looped)

# @a:axis
column_means = X.mean(axis=0)  # down the rows: one mean per column
row_sums = X.sum(axis=1)  # across each row: one sum per row
print("column means:", column_means, "  row sums:", row_sums)
assert column_means.shape == (3,) and row_sums.shape == (4,)

# @a:broadcasting
centred = X - column_means  # (4, 3) − (3,): the means are reused for every row
assert np.allclose(centred.mean(axis=0), 0)

# @a:mask
big = X[X[:, 0] > 4]  # keep rows whose first feature exceeds 4
print("rows with first feature > 4:\n", big)
assert big.shape == (2, 3)

# @a:matmul
w = np.array([0.5, -1.0, 2.0])
predictions = X @ w  # one weighted sum per row: (4, 3) @ (3,) → (4,)
assert np.allclose(predictions, [(row * w).sum() for row in X])
