"""Logarithms and exponentials in NumPy: products become sums, and how to add up huge exponentials."""

import numpy as np


# @a:logsumexp
def logsumexp(z, axis=None):
    """log(Σ e^z) without overflow: factor out the largest term first."""
    top = np.max(z, axis=axis, keepdims=True)
    total = top + np.log(np.sum(np.exp(z - top), axis=axis, keepdims=True))
    return np.squeeze(total, axis=axis) if axis is not None else total.item()


# @a:demo
if __name__ == "__main__":
    # @a:inverse
    x = np.linspace(-3, 3, 7)
    assert np.allclose(np.log(np.exp(x)), x)  # log undoes exp...
    assert np.allclose(np.exp(np.log(x[x > 0])), x[x > 0])  # ...and exp undoes log (for x > 0)

    # @a:products
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 1.0, size=2000)  # 2000 probabilities, e.g. one per training row
    print("product of probabilities:", np.prod(p))  # underflows to exactly 0.0
    print("sum of their logs:       ", np.log(p).sum())  # perfectly representable
    assert np.prod(p) == 0.0 and np.isfinite(np.log(p).sum())
    assert np.isclose(np.log(np.prod(p[:20])), np.log(p[:20]).sum())  # log(ab) = log a + log b

    # @a:derivative
    h = 1e-6
    assert np.isclose((np.exp(1 + h) - np.exp(1)) / h, np.exp(1), rtol=1e-5)  # (eˣ)' = eˣ
    assert np.isclose((np.log(2 + h) - np.log(2)) / h, 1 / 2, rtol=1e-5)  # (log x)' = 1/x

    # @a:stable
    z = np.array([1000.0, 1001.0, 1002.0])
    with np.errstate(over="ignore"):
        naive = np.log(np.sum(np.exp(z)))  # e¹⁰⁰⁰ overflows to inf
    print(f"naive: {naive}, logsumexp: {logsumexp(z):.4f}")
    assert np.isinf(naive)
    assert np.isclose(logsumexp(z), 1002 + np.log(1 + np.exp(-1) + np.exp(-2)))
