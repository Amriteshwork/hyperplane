"""Five tricks for training a big language model, each a few lines of NumPy with a check that it does its job."""

import numpy as np


# @a:schedule
def learning_rate(step, peak, warmup, total):
    """Climb linearly to `peak` over `warmup` steps, then glide down a half cosine to 0."""
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * 0.5 * (1 + np.cos(np.pi * (step - warmup) / (total - warmup)))


# @a:clip
def clip_by_norm(gradients, max_norm):
    """If the whole gradient is longer than `max_norm`, shrink it. Its direction never changes."""
    norm = np.sqrt(sum((g**2).sum() for g in gradients))
    scale = min(1.0, max_norm / norm)
    return [g * scale for g in gradients], norm


# @a:dropout
def dropout(x, rate, rng):
    """Zero a random share of the numbers, and scale the rest so the average stays the same."""
    keep = rng.random(x.shape) >= rate
    return x * keep / (1 - rate)


# @a:smooth
def smooth_targets(token, V, epsilon):
    """Give the right token 1 − ε and share ε across every token, so the model can't be endlessly sure."""
    target = np.full(V, epsilon / V)
    target[token] += 1 - epsilon
    return target


# @a:demo
if __name__ == "__main__":
    # @a:schedule-check
    rates = [learning_rate(s, 1e-3, 100, 1000) for s in range(1000)]
    assert np.isclose(max(rates), 1e-3) and int(np.argmax(rates)) == 99
    assert rates[0] < 2e-5 and rates[-1] < 1e-7
    print(f"learning rate: step 0 {rates[0]:.1e}, peak {max(rates):.1e}, last {rates[-1]:.1e}")

    # @a:clip-check
    grads = [np.array([30.0, 40.0]), np.array([0.0])]
    clipped, norm = clip_by_norm(grads, 1.0)
    assert norm == 50 and np.allclose(clipped[0], [0.6, 0.8])  # same direction, length 1
    small, _ = clip_by_norm([np.array([0.3, 0.4])], 1.0)
    assert np.allclose(small[0], [0.3, 0.4])  # a normal gradient isn't touched
    print(f"clipping: a gradient of length {norm:.0f} becomes {clipped[0]}, a small one is unchanged")

    # @a:dropout-check
    rng = np.random.default_rng(0)
    x = np.ones(100_000)
    dropped = dropout(x, 0.2, rng)
    assert abs((dropped == 0).mean() - 0.2) < 0.01  # about 20% silenced
    assert abs(dropped.mean() - 1) < 0.01  # the expected value is unchanged
    print(f"dropout: {(dropped == 0).mean():.1%} zeroed, mean still {dropped.mean():.3f}")

    # @a:tying
    V, d = 50257, 768
    print(f"weight tying: GPT-2 saves {V * d:,} parameters by using one table in both directions")
    assert V * d == 38_597_376  # the embedding table, about 31% of GPT-2 small

    # @a:smooth-check
    target = smooth_targets(2, 5, 0.1)
    assert np.isclose(target.sum(), 1) and np.isclose(target[2], 0.92)
    # A perfect prediction now has loss above 0 because the target itself is unsure.
    print(f"label smoothing: targets {np.round(target, 2)}")
