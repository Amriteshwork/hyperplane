"""Mean squared error in NumPy: the average squared difference between truth and prediction."""

import numpy as np

# @a:data
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))  # 200 rows, 3 random features
y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(scale=0.3, size=200)
y_pred = X @ np.array([1.8, -0.9, 0.6])  # a model with slightly wrong weights

# @a:mse
errors = y - y_pred
mse = np.mean(errors**2)
rmse = np.sqrt(mse)  # back in the target's units, so easier to read
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")

# @a:formula
# The same number, written exactly like the formula (1/n) Σ (yᵢ − ŷᵢ)².
assert np.isclose(mse, sum(e * e for e in errors) / len(y))

# @a:baseline
# Predicting the mean for every row scores MSE = variance of y: the baseline to beat.
baseline = np.mean((y - y.mean()) ** 2)
print(f"baseline (always predict the mean) = {baseline:.4f}")
assert np.isclose(baseline, y.var())
assert mse < baseline
