"""Text generation in NumPy: one function for temperature, top-k and top-p, then beam search beating greedy."""

import numpy as np

# @a:reuse
from language_modeling_numpy import fit_bigram, make_corpus
from softmax_numpy import softmax
from tokenization_numpy import build_vocab, encode


# @a:filter
def next_token_probs(logits, temperature=1.0, top_k=None, top_p=None):
    """Scores → the distribution to draw from: reshape it with temperature, then cut the tail."""
    p = softmax((logits / temperature)[None])[0]
    order = np.argsort(-p)  # most likely first
    keep = np.ones(len(p), dtype=bool)
    if top_k is not None:
        keep[order[top_k:]] = False
    if top_p is not None:
        mass_before = np.cumsum(p[order]) - p[order]  # keep a token if the ones above it fall short
        keep[order[mass_before >= top_p]] = False
    p = np.where(keep, p, 0.0)
    return p / p.sum()  # what's left must still sum to 1


# @a:beam
def beam_search(log_probs, start, width, steps):
    """Keep the `width` best partial sequences at every step, ranked by total log-probability."""
    beams = [([start], 0.0)]
    for _ in range(steps):
        grown = [(seq + [t], score + lp) for seq, score in beams for t, lp in log_probs(seq).items()]
        beams = sorted(grown, key=lambda beam: -beam[1])[:width]
    return beams[0]


# @a:demo
if __name__ == "__main__":
    # @a:temperature
    logits = np.log(np.array([0.5, 0.3, 0.15, 0.04, 0.01]))
    for t in (0.2, 1.0, 5.0):
        print(f"temperature {t}: {np.round(next_token_probs(logits, t), 3)}")
    assert np.allclose(next_token_probs(logits), [0.5, 0.3, 0.15, 0.04, 0.01])  # τ = 1 changes nothing
    assert next_token_probs(logits, 0.05)[0] > 0.99  # τ → 0 is greedy
    assert next_token_probs(logits, 1000).max() < 0.21  # τ → ∞ is uniform

    # @a:top-k-p
    top_k = next_token_probs(logits, top_k=2)
    top_p = next_token_probs(logits, top_p=0.9)
    print(f"top-k 2:   {np.round(top_k, 3)}")
    print(f"top-p 0.9: {np.round(top_p, 3)}")
    assert (top_k > 0).sum() == 2  # always 2 tokens, however sure the model is
    assert (top_p > 0).sum() == 3  # 0.5 + 0.3 falls short of 0.9, so a third token gets in
    sure = np.log(np.array([0.97, 0.01, 0.01, 0.005, 0.005]))
    assert (next_token_probs(sure, top_p=0.9) > 0).sum() == 1  # a sure model keeps only one

    # @a:greedy-trap
    tree = {
        "<s>": {"the": 0.5, "a": 0.4, "one": 0.1},
        "the": {"cat": 0.4, "dog": 0.35, "end": 0.25},
        "a": {"cat": 0.9, "dog": 0.1},
        "one": {"cat": 1.0},
    }
    def options(seq):
        return {t: np.log(p) for t, p in tree.get(seq[-1], {}).items()}
    for width in (1, 2):
        seq, score = beam_search(options, "<s>", width, 2)
        print(f"beam width {width}: {' '.join(seq[1:]):6}  probability {np.exp(score):.2f}")
    assert np.isclose(np.exp(beam_search(options, "<s>", 1, 2)[1]), 0.20)  # width 1 is greedy
    assert np.isclose(np.exp(beam_search(options, "<s>", 2, 2)[1]), 0.36)  # "a cat" wins, from a worse start

    # @a:real-model
    text = make_corpus(1000)
    vocab = build_vocab(text)
    P = fit_bigram(encode(text, vocab), len(vocab))
    def bigram(seq):
        return {t: np.log(p) for t, p in enumerate(P[seq[-1]]) if p > 0}
    greedy = beam_search(bigram, vocab["z"], 1, 8)[1]
    beam = beam_search(bigram, vocab["z"], 5, 8)[1]
    print(f"bigram, 8 tokens: greedy log-probability {greedy:.2f}, beam of 5 {beam:.2f}")
    assert beam >= greedy
