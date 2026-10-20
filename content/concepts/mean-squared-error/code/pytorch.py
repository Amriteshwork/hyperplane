"""Mean squared error in PyTorch: the average squared difference between truth and prediction."""

import torch
import torch.nn.functional as F


# @a:mse
def mse(y, y_pred):
    """(1/n) Σ (yᵢ − ŷᵢ)². Later concepts reuse it: `from mean_squared_error_pytorch import mse`."""
    return ((y - y_pred) ** 2).mean()


# @a:demo
if __name__ == "__main__":
    # @a:data
    torch.manual_seed(0)
    X = torch.randn(200, 3)  # 200 rows, 3 random features
    y = X @ torch.tensor([2.0, -1.0, 0.5]) + 0.3 * torch.randn(200)
    y_pred = X @ torch.tensor([1.8, -0.9, 0.6])  # a model with slightly wrong weights

    # @a:rmse
    error = mse(y, y_pred)
    rmse = error.sqrt()  # back in the target's units, so easier to read
    print(f"MSE  = {error:.4f}")
    print(f"RMSE = {rmse:.4f}")

    # @a:builtin
    # The built-in loss used to train networks is the same number.
    assert torch.isclose(error, F.mse_loss(y_pred, y))

    # @a:baseline
    # Predicting the mean for every row scores MSE = variance of y: the baseline to beat.
    baseline = mse(y, torch.full_like(y, y.mean().item()))
    print(f"baseline (always predict the mean) = {baseline:.4f}")
    assert torch.isclose(baseline, y.var(correction=0))
    assert error < baseline
