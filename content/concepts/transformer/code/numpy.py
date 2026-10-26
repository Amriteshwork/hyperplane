"""A transformer in NumPy: the forward pass of a tiny GPT, assembled from every earlier concept."""

import numpy as np

# @a:reuse
from activation_functions_numpy import ACTIVATIONS
from causal_masking_numpy import causal_mask
from embeddings_numpy import embed
from language_modeling_numpy import cross_entropy, make_corpus
from layer_normalization_numpy import layer_norm
from multi_head_attention_numpy import multi_head_attention
from positional_encoding_numpy import sinusoidal
from residual_connections_numpy import residual
from softmax_numpy import softmax
from tokenization_numpy import build_vocab, encode

relu = ACTIVATIONS["relu"][0]


# @a:params
def init_params(V, d, n_layers, rng):
    """An embedding table, then per block: two layer norms, attention weights and an MLP."""
    def weight(rows, cols):
        return rng.normal(size=(rows, cols)) / np.sqrt(rows)

    def norm():
        return np.ones(d), np.zeros(d)  # γ = 1, β = 0: start as plain standardisation

    blocks = [
        {"ln1": norm(), "attn": [weight(d, d) for _ in range(4)], "ln2": norm(),
         "W1": weight(d, 4 * d), "b1": np.zeros(4 * d), "W2": weight(4 * d, d), "b2": np.zeros(d)}
        for _ in range(n_layers)
    ]
    return {"E": rng.normal(size=(V, d)), "blocks": blocks, "ln_f": norm(),
            "W_out": weight(d, V) * 0.01}  # tiny outputs: start out unsure about everything


# @a:block
def block(X, p, n_heads):
    """Tokens talk (attention), then each token thinks on its own (MLP); both as residuals."""
    def talk(h):
        return multi_head_attention(layer_norm(h, *p["ln1"]), *p["attn"], n_heads, causal_mask(len(h)))[0]

    def think(h):
        return relu(layer_norm(h, *p["ln2"]) @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]

    return residual(residual(X, talk), think)


# @a:gpt
def gpt(ids, params, n_heads):
    """Token ids (T,) → next-token scores (T, V): one prediction per position."""
    X = embed(ids, params["E"]) + sinusoidal(len(ids), params["E"].shape[1])
    for p in params["blocks"]:
        X = block(X, p, n_heads)
    return layer_norm(X, *params["ln_f"]) @ params["W_out"]


# @a:demo
if __name__ == "__main__":
    # @a:forward
    rng = np.random.default_rng(0)
    vocab = build_vocab(make_corpus(1000))
    ids = encode(make_corpus(2, seed=1), vocab)[:40]
    V, d, n_layers, n_heads = len(vocab), 32, 2, 4
    params = init_params(V, d, n_layers, rng)
    logits = gpt(ids, params, n_heads)
    print(f"{len(ids)} tokens in -> logits {logits.shape}: a next-token guess at every position")
    assert logits.shape == (len(ids), V)

    # @a:causal
    changed = ids.copy()
    changed[25:] = (changed[25:] + 1) % V  # rewrite the end of the text
    assert np.allclose(gpt(changed, params, n_heads)[:25], logits[:25])  # the past can't see it
    print("through every layer, no position sees the future")

    # @a:untrained
    loss = cross_entropy(softmax(logits[:-1]), ids[1:])
    print(f"untrained loss {loss:.3f} vs log V = {np.log(V):.3f}: it starts out guessing uniformly")
    assert abs(loss - np.log(V)) < 0.05

    # @a:count
    def n_params(tree):
        """Count the numbers in every array inside nested dicts, lists and tuples."""
        if isinstance(tree, np.ndarray):
            return tree.size
        return sum(n_params(t) for t in (tree.values() if isinstance(tree, dict) else tree))

    per_block = 4 * d * d + 8 * d * d + 5 * d + 4 * d  # attention, MLP weights, MLP biases, norms
    assert n_params(params) == 2 * V * d + n_layers * per_block + 2 * d
    print(f"{n_params(params):,} parameters; per block ≈ 12·d² = {12 * d * d:,} "
          f"(GPT-2 small: 12·768² ≈ 7.1M per block)")
