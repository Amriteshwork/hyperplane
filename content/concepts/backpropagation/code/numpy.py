"""Backpropagation in NumPy: the chain rule, layer by layer, for every weight of a network."""

import numpy as np

# @a:reuse
from activation_functions_numpy import ACTIVATIONS
from derivatives_numpy import numerical_gradient
from mean_squared_error_numpy import mse


# @a:init
def init_network(sizes, rng):
    """A (W, b) pair per layer; small random weights, scaled to each layer (Glorot)."""
    return [
        (rng.uniform(-1, 1, (m, k)) * np.sqrt(6 / (m + k)), np.zeros(k))
        for m, k in zip(sizes, sizes[1:])
    ]


# @a:forward
def forward(X, params, activation="tanh"):
    """Every layer's output, inputs first; the last layer is linear (regression)."""
    bend = ACTIVATIONS[activation][0]
    outputs = [X]
    for i, (W, b) in enumerate(params):
        z = outputs[-1] @ W + b
        outputs.append(z if i == len(params) - 1 else bend(z))
    return outputs


# @a:backward
def backward(outputs, y, params, activation="tanh"):
    """∂MSE/∂W and ∂MSE/∂b for every layer, from the top down."""
    slope = ACTIVATIONS[activation][1]
    delta = 2 * (outputs[-1] - y[:, None]) / len(y)  # ∂MSE/∂ŷ: the scaled error
    grads = [None] * len(params)
    for i in reversed(range(len(params))):
        W, _ = params[i]
        grads[i] = (outputs[i].T @ delta, delta.sum(axis=0))  # this layer's weights and biases
        if i > 0:
            delta = (delta @ W.T) * slope(outputs[i])  # chain rule: back through W, then f
    return grads


# @a:demo
if __name__ == "__main__":
    # @a:data
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, size=200)
    y = np.sin(2 * x) + rng.normal(scale=0.1, size=200)  # a curve no straight line fits
    X = x[:, None]

    # @a:gradient-check
    params = init_network([1, 16, 1], rng)
    grads = backward(forward(X, params), y, params)
    for i, (W, b) in enumerate(params):
        def loss_with(w, i=i, shape=W.shape):  # the MSE as a function of layer i's weights only
            trial = [(w.reshape(shape), b_) if j == i else (W_, b_) for j, (W_, b_) in enumerate(params)]
            return mse(y, forward(X, trial)[-1][:, 0])

        numeric = numerical_gradient(loss_with, W.ravel()).reshape(W.shape)
        assert np.allclose(grads[i][0], numeric, atol=1e-6)
    print("backpropagation matches the numerical gradient for every weight")

    # @a:train
    for step in range(5001):
        outputs = forward(X, params)
        grads = backward(outputs, y, params)
        params = [(W - 0.1 * dW, b - 0.1 * db) for (W, b), (dW, db) in zip(params, grads)]
        if step % 1000 == 0:
            print(f"step {step:4d}: MSE {mse(y, outputs[-1][:, 0]):.4f}")

    # @a:compare
    network_mse = mse(y, forward(X, params)[-1][:, 0])
    slope, intercept = np.polyfit(x, y, 1)
    line_mse = mse(y, slope * x + intercept)
    print(f"network MSE {network_mse:.4f} vs straight line {line_mse:.4f}")
    assert network_mse < 0.1 < 0.4 < line_mse  # the hidden layer bent itself around the curve
