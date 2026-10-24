"""Adam in NumPy: momentum plus a per-weight step size, with bias correction."""

import numpy as np

# @a:reuse
from backpropagation_numpy import backward, forward, init_network
from mean_squared_error_numpy import mse


# @a:adam
def adam_step(params, grads, state, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    """One Adam update of a list of arrays. `state` is (t, ms, vs); returns (params, state)."""
    t, ms, vs = state
    t += 1
    ms = [beta_1 * m + (1 - beta_1) * g for m, g in zip(ms, grads)]  # average gradient
    vs = [beta_2 * v + (1 - beta_2) * g**2 for v, g in zip(vs, grads)]  # average squared gradient
    # @a:bias-correction
    m_hats = [m / (1 - beta_1**t) for m in ms]  # both averages start at 0: scale them up early on
    v_hats = [v / (1 - beta_2**t) for v in vs]
    # @a:step
    params = [p - learning_rate * m / (np.sqrt(v) + epsilon) for p, m, v in zip(params, m_hats, v_hats)]
    return params, (t, ms, vs)


def adam_state(params):
    return 0, [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]


# @a:demo
if __name__ == "__main__":
    # @a:scale-free
    w = [np.zeros(3)]
    grad = [np.array([1e-4, 1.0, 1e4])]  # gradients of wildly different sizes
    new, _ = adam_step(w, grad, adam_state(w), learning_rate=0.01)
    print("first Adam step per weight:", new[0])
    assert np.allclose(new[0], -0.01, rtol=1e-3)  # every weight moves by about η, whatever its scale

    # @a:ravine
    def gradient(w):
        return np.array([100.0, 1.0]) * w  # f(w) = ½(100 w₀² + w₁²): steep one way, flat the other

    gd = np.array([1.0, 1.0])
    for _ in range(100):
        gd = gd - 0.019 * gradient(gd)  # any larger step makes w₀ diverge (needs η < 2/100)
    params = [np.array([1.0, 1.0])]
    state = adam_state(params)
    for _ in range(100):
        params, state = adam_step(params, [gradient(params[0])], state, learning_rate=0.1)
    print(f"after 100 steps: gradient descent {gd.round(3)}, Adam {params[0].round(3)}")
    assert abs(gd[1]) > 0.1 > abs(params[0][1])  # GD still crawling along the flat direction

    # @a:network
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, size=200)
    y = np.sin(2 * x) + rng.normal(scale=0.1, size=200)  # the backpropagation page's data
    X = x[:, None]
    layers = init_network([1, 16, 1], rng)
    flat = [a for W, b in layers for a in (W, b)]  # Adam sees a flat list of arrays
    state = adam_state(flat)
    for step in range(1001):
        layers = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
        grads = [g for dW, db in backward(forward(X, layers), y, layers) for g in (dW, db)]
        flat, state = adam_step(flat, grads, state, learning_rate=0.01)
        if step % 200 == 0:
            print(f"step {step:4d}: MSE {mse(y, forward(X, layers)[-1][:, 0]):.4f}")
    final = mse(y, forward(X, layers)[-1][:, 0])
    assert final < 0.02  # near the noise floor (0.01); 5,000 plain steps only reached 0.047
