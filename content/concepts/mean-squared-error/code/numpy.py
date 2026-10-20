"""Mean squared error in NumPy: the average squared difference between truth and prediction."""

import numpy as np


# @a:mse
def mse(y, y_pred):
    """(1/n) Σ (yᵢ − ŷᵢ)². Later concepts reuse it: `from mean_squared_error_numpy import mse`."""
    return np.mean((y - y_pred) ** 2)


# @a:demo
if __name__ == "__main__":
    # @a:data
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))  # 200 rows, 3 random features
    y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(scale=0.3, size=200)
    y_pred = X @ np.array([1.8, -0.9, 0.6])  # a model with slightly wrong weights

    # @a:rmse
    error = mse(y, y_pred)
    rmse = np.sqrt(error)  # back in the target's units, so easier to read
    print(f"MSE  = {error:.4f}")
    print(f"RMSE = {rmse:.4f}")

    # @a:formula
    # The same number, written exactly like the formula (1/n) Σ (yᵢ − ŷᵢ)².
    assert np.isclose(error, sum((a - b) ** 2 for a, b in zip(y, y_pred)) / len(y))

    # @a:baseline
    # Predicting the mean for every row scores MSE = variance of y: the baseline to beat.
    baseline = mse(y, np.full_like(y, y.mean()))
    print(f"baseline (always predict the mean) = {baseline:.4f}")
    assert np.isclose(baseline, y.var())
    assert error < baseline
