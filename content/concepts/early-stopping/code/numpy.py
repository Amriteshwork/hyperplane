"""Early stopping in NumPy: watch a validation loss, stop when it stalls, keep the best weights."""

import numpy as np

# @a:reuse
from adam_optimizer_numpy import adam_state, adam_step
from backpropagation_numpy import backward, forward, init_network
from mean_squared_error_numpy import mse


# @a:early-stopping
class EarlyStopping:
    """Remembers the best validation loss and its weights; says stop after `patience` bad epochs."""

    def __init__(self, patience=10, tol=0.0):
        self.patience, self.tol = patience, tol
        self.best, self.best_epoch, self.best_params, self.stale = np.inf, 0, None, 0

    def update(self, epoch, loss, params):
        """Record one epoch's validation loss; True once it has stalled for `patience` epochs."""
        if loss < self.best - self.tol:
            self.best, self.best_epoch, self.stale = loss, epoch, 0
            self.best_params = [p.copy() for p in params]  # a copy: training keeps changing params
        else:
            self.stale += 1
        return self.stale >= self.patience


# @a:demo
if __name__ == "__main__":
    # @a:data
    rng = np.random.default_rng(0)

    def noisy_sine(n):
        x = rng.uniform(-3, 3, size=n)
        return x[:, None], np.sin(x) + rng.normal(scale=0.3, size=n)

    X_train, y_train = noisy_sine(30)  # few rows ...
    X_val, y_val = noisy_sine(200)  # ... judged on rows it never trains on

    # @a:network
    def as_layers(flat):
        return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]

    flat = [a for W, b in init_network([1, 32, 32, 1], rng) for a in (W, b)]  # ~1,100 weights
    state = adam_state(flat)

    # @a:loop
    stopper, stopped_at = EarlyStopping(patience=100), None
    for epoch in range(1, 3001):
        grads = [g for dW, db in backward(forward(X_train, as_layers(flat)), y_train, as_layers(flat))
                 for g in (dW, db)]
        flat, state = adam_step(flat, grads, state, learning_rate=0.02)
        val_loss = mse(y_val, forward(X_val, as_layers(flat))[-1][:, 0])
        # @a:check
        if stopper.update(epoch, val_loss, flat) and stopped_at is None:
            stopped_at = epoch  # early stopping would break here; we carry on to see why
        if epoch in (1, 100, 300, 1000, 3000):
            train_loss = mse(y_train, forward(X_train, as_layers(flat))[-1][:, 0])
            print(f"epoch {epoch:4d}: training MSE {train_loss:.3f}, validation MSE {val_loss:.3f}")

    # @a:compare
    kept = mse(y_val, forward(X_val, as_layers(stopper.best_params))[-1][:, 0])
    print(f"best epoch {stopper.best_epoch}, early stopping at {stopped_at}: "
          f"validation MSE {kept:.3f} with its weights vs {val_loss:.3f} after 3,000 epochs")
    assert stopped_at < 1000  # it would have stopped long before the end ...
    assert kept < 0.15 < 0.25 < val_loss  # ... with far better weights than training on gave
