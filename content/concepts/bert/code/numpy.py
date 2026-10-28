"""BERT in NumPy: the transformer's block with the causal mask taken off, reading [CLS] A [SEP] B [SEP]."""

import re

import numpy as np

# @a:reuse
from embeddings_numpy import embed
from language_modeling_numpy import make_corpus
from layer_normalization_numpy import layer_norm
from masked_language_modeling_numpy import add_tokens, mask_tokens, mlm_loss
from positional_encoding_numpy import sinusoidal
from softmax_numpy import softmax
from tokenization_numpy import build_vocab, encode
from transformer_numpy import block, init_params


# @a:words
def words(text):
    """Word-level tokens: 'zoe likes jam, so' → ['zoe', 'likes', 'jam', ',', 'so']."""
    return re.findall(r"\w+|[,.]", text)


# @a:pack
def pack(a, b, vocab):
    """Two token lists as one input, [CLS] a [SEP] b [SEP], with segment ids: 0 for a, 1 for b."""
    ids = [vocab["[CLS]"], *a, vocab["[SEP]"], *b, vocab["[SEP]"]]
    return np.array(ids), np.array([0] * (len(a) + 2) + [1] * (len(b) + 1))


# @a:padding-mask
def padding_mask(ids, pad_id):
    """Every token may look at every token, before or after it, except padding."""
    return np.broadcast_to(ids != pad_id, (len(ids), len(ids)))


# @a:params
def init_bert(V, d, n_layers, rng):
    """GPT's weights plus a segment table: one learned vector for sentence A, one for B."""
    return {**init_params(V, d, n_layers, rng), "S": rng.normal(size=(2, d))}


# @a:encoder
def bert(ids, segments, params, n_heads, pad_id):
    """Token + segment + position embeddings, then the blocks without a causal mask: (T, d)."""
    X = embed(ids, params["E"]) + embed(segments, params["S"]) + sinusoidal(len(ids), params["E"].shape[1])
    for p in params["blocks"]:
        X = block(X, p, n_heads, padding_mask(ids, pad_id))
    return layer_norm(X, *params["ln_f"])  # one context-aware vector per token


# @a:demo
if __name__ == "__main__":
    # @a:input
    rng = np.random.default_rng(0)
    vocab = build_vocab(words(make_corpus(1000)))
    V = len(vocab)  # real words; the special tokens come after them
    vocab = add_tokens(vocab, "[PAD]", "[CLS]", "[SEP]", "[MASK]")
    a, b = (encode(words(s), vocab) for s in ("zoe likes jam, so zoe eats jam.", "raj likes figs."))
    ids, segments = pack(a, b, vocab)
    itos = {i: t for t, i in vocab.items()}
    print(" ".join(f"{itos[i]}:{s}" for i, s in zip(ids, segments)))  # token:segment
    assert len(ids) == len(a) + len(b) + 3 and segments.sum() == len(b) + 1

    # @a:both-ways
    params = init_bert(len(vocab), 32, 2, rng)
    H = bert(ids, segments, params, 4, vocab["[PAD]"])
    changed = ids.copy()
    changed[-3] = vocab["jam"]  # "raj likes figs." → "raj likes jam."
    assert not np.allclose(bert(changed, segments, params, 4, vocab["[PAD]"])[1], H[1])
    print(f"{H.shape[0]} vectors of {H.shape[1]}; editing the last word changes the first word's vector")

    # @a:padding
    padded = np.concatenate([ids, [vocab["[PAD]"]] * 5])
    H_pad = bert(padded, np.concatenate([segments, [0] * 5]), params, 4, vocab["[PAD]"])
    assert np.allclose(H_pad[: len(ids)], H)  # padding changes nothing for the real tokens
    print("5 [PAD] tokens added: the real tokens' vectors are identical")

    # @a:mlm
    corrupted, picked = mask_tokens(ids, vocab["[MASK]"], V, rng, rate=0.3)
    probs = softmax(bert(corrupted, segments, params, 4, vocab["[PAD]"]) @ params["W_out"])
    loss = mlm_loss(probs, ids, picked)
    print(f"masked {picked.sum()} words; untrained loss {loss:.2f} vs log V = {np.log(len(vocab)):.2f}")
    assert abs(loss - np.log(len(vocab))) < 0.1

    # @a:fine-tune
    K = 2  # e.g. "does the second sentence follow the first?"
    W_cls = rng.normal(size=(params["E"].shape[1], K)) * 0.01  # the only new weights
    p_cls = softmax(H[:1] @ W_cls)  # read the [CLS] vector, the first row of H
    print(f"fine-tuning adds {W_cls.size} new weights (d × K); P(classes) = {p_cls.round(2)}")
    assert p_cls.shape == (1, K) and np.isclose(p_cls.sum(), 1)
