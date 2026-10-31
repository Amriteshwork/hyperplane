"""Lazy learning in PyTorch: k-nearest neighbours as `cdist` + `topk`; nothing here has a gradient."""

import torch


# @a:neighbours
def nearest(X_train, X_query, k):
    """The row numbers of each query's k closest training rows."""
    return torch.cdist(X_query, X_train).topk(k, dim=1, largest=False).indices


# @a:classify
def knn_classify(X_train, y_train, X_query, k):
    """The most common label among each query's k nearest rows."""
    return y_train[nearest(X_train, X_query, k)].mode(dim=1).values


# @a:regress
def knn_regress(X_train, y_train, X_query, k):
    """The average target of each query's k nearest rows."""
    return y_train[nearest(X_train, X_query, k)].mean(dim=1)


# @a:demo
if __name__ == "__main__":
    # @a:data
    torch.manual_seed(0)

    def two_blobs(n):
        y = torch.randint(0, 2, (n,))
        return torch.randn(n, 2) + 1.2 * y[:, None], y

    X_train, y_train = two_blobs(300)
    X_test, y_test = two_blobs(300)

    # @a:sweep
    for k in (1, 15):
        train = (knn_classify(X_train, y_train, X_train, k) == y_train).float().mean().item()
        test = (knn_classify(X_train, y_train, X_test, k) == y_test).float().mean().item()
        print(f"k = {k:2d}: train {train:.3f}, test {test:.3f}")
    assert (knn_classify(X_train, y_train, X_train, 1) == y_train).all()  # k = 1 memorises
    assert test > 0.7

    # @a:regression
    x = torch.linspace(-3, 3, 200)[:, None]
    y = torch.sin(x[:, 0]) + 0.2 * torch.randn(200)
    grid = torch.linspace(-2.5, 2.5, 50)[:, None]
    error = (knn_regress(x, y, grid, 10) - torch.sin(grid[:, 0])).pow(2).mean().item()
    print(f"regression MSE against sin(x), k = 10: {error:.4f}")
    assert error < 0.03
