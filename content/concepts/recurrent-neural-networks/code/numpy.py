"""A recurrent network in NumPy: one hidden state carried through the text, trained by backpropagation through time."""

import numpy as np

# @a:reuse
from adam_optimizer_numpy import adam_state, adam_step
from embeddings_numpy import embed
from language_modeling_numpy import cross_entropy, fit_bigram, make_corpus
from softmax_numpy import softmax
from tokenization_numpy import build_vocab, encode


# @a:params
def init_rnn(V, d, rng):
    """Input embeddings E, recurrent weights W and bias b, read-out U and c."""
    return [rng.normal(size=(V, d)) * 0.1, rng.normal(size=(d, d)) / np.sqrt(d), np.zeros(d),
            rng.normal(size=(d, V)) * 0.1, np.zeros(V)]


# @a:forward
def rnn_forward(ids, params):
    """Read one token at a time, h_t = tanh(E[x_t] + h_{t-1} W + b); returns states (T, d), logits (T, V)."""
    E, W, b, U, c = params
    h, states = np.zeros(W.shape[0]), []
    for x in embed(ids, E):
        h = np.tanh(x + h @ W + b)  # the same weights at every step: only h changes
        states.append(h)
    H = np.array(states)
    return H, H @ U + c


# @a:bptt
def bptt(ids, targets, params):
    """Loss and gradients: the forward pass, then the chain rule from the last step back to the first."""
    E, W, b, U, c = params
    H, logits = rnn_forward(ids, params)
    probs = softmax(logits)
    d_logits = probs.copy()
    d_logits[np.arange(len(ids)), targets] -= 1  # softmax + cross-entropy: p − one_hot(target)
    d_logits /= len(ids)
    dE, dW, db = np.zeros_like(E), np.zeros_like(W), np.zeros_like(b)
    d_h_next = np.zeros_like(b)  # gradient arriving from the future
    for t in reversed(range(len(ids))):
        dz = (d_logits[t] @ U.T + d_h_next) * (1 - H[t] ** 2)  # now's loss + later losses, through tanh
        dW += np.outer(H[t - 1] if t else np.zeros_like(b), dz)
        db += dz
        dE[ids[t]] += dz
        d_h_next = dz @ W.T  # hand it one step further back
    return cross_entropy(probs, targets), [dE, dW, db, H.T @ d_logits, d_logits.sum(axis=0)]


# @a:demo
if __name__ == "__main__":
    # @a:grad-check
    rng = np.random.default_rng(0)
    text = make_corpus(1000)
    vocab = build_vocab(text)
    ids = encode(text, vocab)
    V, d, T = len(vocab), 48, 32
    params = init_rnn(V, d, rng)
    x, y = ids[:T], ids[1 : T + 1]
    _, grads = bptt(x, y, params)
    eps, (i, j) = 1e-6, (3, 5)
    params[1][i, j] += eps
    up = bptt(x, y, params)[0]
    params[1][i, j] -= 2 * eps
    down = bptt(x, y, params)[0]
    params[1][i, j] += eps
    assert np.isclose((up - down) / (2 * eps), grads[1][i, j], rtol=1e-4)
    print("backpropagation through time matches a finite difference")

    # @a:vanishing
    def reach(W, k=24):
        """Size of ∂h_T/∂h_{T−k}: how strongly a gradient k steps back still arrives."""
        H, _ = rnn_forward(x, [params[0], W, *params[2:]])
        J = np.eye(d)
        for t in range(T - 1, T - 1 - k, -1):
            J = J @ (np.diag(1 - H[t] ** 2) @ W.T)  # ∂h_t/∂h_{t−1}, chained backwards
        return np.linalg.norm(J, 2)

    for scale in (0.5, 1, 3):
        print(f"W × {scale}: a gradient 24 steps back is scaled by {reach(params[1] * scale):.1e}")
    assert reach(params[1] * 0.5) < 1e-4 < 1e3 < reach(params[1] * 3)  # it vanishes or explodes

    # @a:train
    split = int(0.9 * len(ids))
    state = adam_state(params)
    for step in range(2001):
        start = rng.integers(split - T - 1)
        loss, grads = bptt(ids[start : start + T], ids[start + 1 : start + T + 1], params)
        grads = [np.clip(g, -5, 5) for g in grads]  # clip: one huge gradient can't wreck training
        params, state = adam_step(params, grads, state, learning_rate=0.01)
        if step % 1000 == 0:
            print(f"step {step:4d}: loss {loss:.3f}")

    # @a:evaluate
    val = ids[split:]
    val_loss = np.mean([bptt(val[s : s + T], val[s + 1 : s + T + 1], params)[0]
                        for s in range(0, len(val) - T - 1, T)])
    bigram_loss = cross_entropy(fit_bigram(ids[:split], V)[val[:-1]], val[1:])
    print(f"validation loss: RNN {val_loss:.3f} vs bigram {bigram_loss:.3f}")
    assert val_loss < bigram_loss / 3  # memory of more than one character pays off
