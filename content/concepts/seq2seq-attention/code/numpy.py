"""Seq2seq with attention in NumPy: an LSTM encoder, Bahdanau's additive score, and why one vector isn't enough."""

import numpy as np

# @a:reuse
from lstm_numpy import init_lstm, lstm_forward
from softmax_numpy import softmax


# @a:additive
def additive_attention(s, H, W_s, W_h, v):
    """Score each encoder state against decoder state `s`, softmax, and blend: (d,), (S, d) → (d,), (S,)."""
    scores = np.tanh(s @ W_s + H @ W_h) @ v  # one number per English word
    weights = softmax(scores[None])[0]
    return weights @ H, weights


# @a:demo
if __name__ == "__main__":
    # @a:encode
    rng = np.random.default_rng(0)
    V, d = 20, 8
    params = init_lstm(V, d, rng)
    W_s, W_h, v = rng.normal(size=(d, d)), rng.normal(size=(d, d)), rng.normal(size=d)
    short = rng.integers(V, size=4)
    long = rng.integers(V, size=40)
    H_short = lstm_forward(short, params)[0]
    H_long = lstm_forward(long, params)[0]
    print(f"encoder states: {H_short.shape} for 4 words, {H_long.shape} for 40")

    # @a:bottleneck
    # Without attention the decoder gets only the last row, the same size either way.
    assert H_short[-1].shape == H_long[-1].shape == (d,)
    print(f"no attention: the decoder's input is {d} numbers, for 4 words or for 40")

    # @a:look-back
    s = rng.normal(size=d)
    context, weights = additive_attention(s, H_long, W_s, W_h, v)
    assert weights.shape == (40,) and np.isclose(weights.sum(), 1)  # it spreads 100% over the English
    assert np.allclose(context, sum(w * h for w, h in zip(weights, H_long)))  # a weighted sum
    print(f"attention: weights {weights.shape}, the context is a blend of all 40 states")

    # @a:recomputed
    # A new decoder state asks a new question, so the same memory yields a new context.
    other_context, other_weights = additive_attention(rng.normal(size=d), H_long, W_s, W_h, v)
    assert not np.allclose(weights, other_weights)
    assert other_context.shape == context.shape == (d,)
    print("a different decoder state reads the English differently")
