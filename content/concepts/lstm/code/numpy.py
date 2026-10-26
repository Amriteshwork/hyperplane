"""An LSTM in NumPy: gates that decide what to keep, write and show, and a memory that survives the trip back in time."""

import numpy as np

# @a:reuse
from embeddings_numpy import embed
from language_modeling_numpy import make_corpus
from sigmoid_numpy import sigmoid
from tokenization_numpy import build_vocab, encode


# @a:params
def init_lstm(V, d, rng):
    """Embeddings E; weights Wx, Wh and biases b for four gates side by side; read-out U."""
    return [rng.normal(size=(V, d)) * 0.1, rng.normal(size=(d, 4 * d)) / np.sqrt(d),
            rng.normal(size=(d, 4 * d)) / np.sqrt(d), np.zeros(4 * d), rng.normal(size=(d, V)) * 0.1]


# @a:step
def lstm_step(x, h, c, Wx, Wh, b):
    """One token: forget part of the cell c, write new content into it, show part of it as h."""
    i, f, g, o = np.split(x @ Wx + h @ Wh + b, 4)
    i, f, o = sigmoid(i), sigmoid(f), sigmoid(o)  # gates: each entry between 0 (shut) and 1 (open)
    c = f * c + i * np.tanh(g)  # the memory is updated by adding, not by multiplying by W
    return o * np.tanh(c), c, f


# @a:forward
def lstm_forward(ids, params):
    """Run over the tokens; returns hidden states and cells (T, d), forget gates (T, d), logits (T, V)."""
    E, Wx, Wh, b, U = params
    h = c = np.zeros(Wh.shape[0])
    H, C, F = [], [], []
    for x in embed(ids, E):
        h, c, f = lstm_step(x, h, c, Wx, Wh, b)
        H.append(h)
        C.append(c)
        F.append(f)
    return np.array(H), np.array(C), np.array(F), np.array(H) @ U


# @a:demo
if __name__ == "__main__":
    # @a:gates
    rng = np.random.default_rng(0)
    text = make_corpus(1000)
    vocab = build_vocab(text)
    ids = encode(text, vocab)[:25]
    V, d = len(vocab), 32
    params = init_lstm(V, d, rng)
    H, C, F, logits = lstm_forward(ids, params)
    print(f"{len(ids)} tokens -> h, c {H.shape}, logits {logits.shape}; forget gates average {F.mean():.2f}")
    assert logits.shape == (len(ids), V) and np.all((0 < F) & (F < 1)) and np.all(np.abs(H) < 1)

    # @a:forget-path
    def with_forget_bias(value):
        """The same LSTM with every forget gate's bias set to `value`."""
        b = params[3].copy()
        b[d : 2 * d] = value
        return [*params[:3], b, params[4]]

    for bias in (0.0, 5.0):
        F = lstm_forward(ids, with_forget_bias(bias))[2]
        path = np.prod(F[1:], axis=0)  # ∂c_T/∂c_1 along the cell: one forget gate per step
        print(f"forget bias {bias}: the cell path keeps {path.mean():.1e} of a gradient over {len(ids) - 1} steps")
    assert np.prod(lstm_forward(ids, with_forget_bias(5.0))[2][1:], axis=0).min() > 0.5

    # @a:remembers
    other = ids.copy()
    other[0] = (other[0] + 1) % V  # a different first character, the same 24 after it
    gaps = []
    for bias in (0.0, 5.0):
        p = with_forget_bias(bias)
        gaps.append(np.abs(lstm_forward(ids, p)[1][-1] - lstm_forward(other, p)[1][-1]).max())
        print(f"forget bias {bias}: 24 steps later the cells still differ by {gaps[-1]:.1e}")
    assert gaps[1] > 100 * gaps[0]  # with the forget gates open, the first token is still there
