"""Masked language modelling in NumPy: hide 15% of the tokens and guess them from both sides."""

import numpy as np

# @a:reuse
from language_modeling_numpy import cross_entropy, fit_bigram, make_corpus
from tokenization_numpy import build_vocab, encode


# @a:add-tokens
def add_tokens(vocab, *tokens):
    """Extend a vocabulary with special tokens such as '[MASK]', numbered after the real ones."""
    return {**vocab, **{t: len(vocab) + i for i, t in enumerate(tokens)}}


# @a:mask
def mask_tokens(ids, mask_id, V, rng, rate=0.15):
    """Pick `rate` of the real tokens (ids < V); of those, 80% → [MASK], 10% → random, 10% kept."""
    picked = (rng.random(ids.shape) < rate) & (ids < V)  # special tokens are never picked
    roll = rng.random(ids.shape)
    corrupted = ids.copy()
    corrupted[picked & (roll < 0.8)] = mask_id
    swap = picked & (roll >= 0.8) & (roll < 0.9)
    corrupted[swap] = rng.integers(V, size=swap.sum())
    return corrupted, picked


# @a:loss
def mlm_loss(probs, targets, picked):
    """Cross-entropy at the picked positions only: the rest of the text is given, not asked."""
    return cross_entropy(probs[picked], targets[picked])


# @a:demo
if __name__ == "__main__":
    # @a:recipe
    rng = np.random.default_rng(0)
    vocab = build_vocab(make_corpus(1000))
    V = len(vocab)
    vocab = add_tokens(vocab, "[MASK]")
    ids = encode(make_corpus(1000), vocab)
    corrupted, picked = mask_tokens(ids, vocab["[MASK]"], V, rng)
    masked = corrupted[picked] == vocab["[MASK]"]
    kept = corrupted[picked] == ids[picked]
    print(f"picked {picked.mean():.1%} of {len(ids)} tokens: {masked.mean():.0%} → [MASK], "
          f"{kept.mean():.0%} left as they were, the rest swapped for a random token")
    assert abs(picked.mean() - 0.15) < 0.01 and abs(masked.mean() - 0.8) < 0.02
    assert np.array_equal(corrupted[~picked], ids[~picked])  # everything else is untouched

    # @a:both-sides
    split = int(0.9 * len(ids))
    P = fit_bigram(ids[:split], V)  # P[a, b] = P(next is b | current is a)
    val, hide = ids[split:], picked[split:].copy()
    hide[[0, -1]] = False  # keep a neighbour on each side
    t = np.flatnonzero(hide)
    left_only = np.zeros((len(val), V))
    left_only[t] = P[val[t - 1]]  # what usually follows the token on the left
    both = np.zeros((len(val), V))
    both[t] = P[val[t - 1]] * P[:, val[t + 1]].T  # ...and usually precedes the one on the right
    both[t] /= both[t].sum(axis=1, keepdims=True)
    loss_left, loss_both = mlm_loss(left_only, val, hide), mlm_loss(both, val, hide)
    print(f"guessing hidden characters: left only {loss_left:.2f} nats, both sides {loss_both:.2f}")
    assert loss_both < loss_left / 2  # the right-hand side carries as much again

    # @a:uniform
    uniform = np.full((len(val), V), 1 / V)
    assert np.isclose(mlm_loss(uniform, val, hide), np.log(V))  # the baseline is still log V
