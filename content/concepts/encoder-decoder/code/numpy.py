"""An encoder-decoder transformer in NumPy: read the English sentence once, then write German one word at a time."""

import re

import numpy as np

# @a:reuse
from activation_functions_numpy import ACTIVATIONS
from causal_masking_numpy import causal_mask
from cross_attention_numpy import cross_attention
from embeddings_numpy import embed
from language_modeling_numpy import FOODS, NAMES
from layer_normalization_numpy import layer_norm
from multi_head_attention_numpy import multi_head_attention
from positional_encoding_numpy import sinusoidal
from residual_connections_numpy import residual
from softmax_numpy import softmax
from tokenization_numpy import build_vocab, encode
from transformer_numpy import block, init_params

relu = ACTIVATIONS["relu"][0]

# @a:pairs
GERMAN = {"likes": "mag", "so": "also", "eats": "isst", "figs": "feigen", "plums": "pflaumen",
          "rice": "reis", "nuts": "nüsse", "soup": "suppe", "kale": "grünkohl", "eggs": "eier",
          "jam": "marmelade"}


def words(text):
    """Word-level tokens: 'zoe likes jam, so' → ['zoe', 'likes', 'jam', ',', 'so']."""
    return re.findall(r"\w+|[,.]", text)


def to_german(english):
    """'zoe likes jam, so raj eats figs.' → 'zoe mag marmelade, also isst raj feigen.' as tokens."""
    out = [GERMAN.get(w, w) for w in words(english)]
    i = out.index("also")
    out[i + 1], out[i + 2] = out[i + 2], out[i + 1]  # after "also", German puts the verb first
    return out


def make_pairs(n, seed=0):
    """`n` English sentences with their German translations, as word lists."""
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        a, b = rng.choice(NAMES, 2)
        f, g = rng.choice(FOODS, 2)
        english = f"{a} likes {f}, so {b} eats {g}."
        pairs.append((words(english), to_german(english)))
    return pairs


# @a:params
def init_seq2seq(V_src, V_tgt, d, n_layers, rng):
    """Two transformers' weights; each decoder block also gets cross-attention weights and a norm."""
    encoder = init_params(V_src, d, n_layers, rng)
    del encoder["W_out"]  # the encoder predicts nothing: its vectors are the product
    decoder = init_params(V_tgt, d, n_layers, rng)
    for p in decoder["blocks"]:
        p["ln_x"] = (np.ones(d), np.zeros(d))
        p["cross"] = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4)]
    return {"encoder": encoder, "decoder": decoder}


# @a:encoder
def encoder(src, p, n_heads):
    """Source ids (S,) → one vector per source token (S, d). No mask: the whole sentence is known."""
    X = embed(src, p["E"]) + sinusoidal(len(src), p["E"].shape[1])
    for b in p["blocks"]:
        X = block(X, b, n_heads, None)
    return layer_norm(X, *p["ln_f"])


# @a:decoder-block
def decoder_block(Y, memory, p, n_heads):
    """Look back at the German so far, then read the English, then think: three residual steps."""
    def talk(h):
        return multi_head_attention(layer_norm(h, *p["ln1"]), *p["attn"], n_heads, causal_mask(len(h)))[0]

    def read(h):
        return cross_attention(layer_norm(h, *p["ln_x"]), memory, *p["cross"], n_heads)[0]

    def think(h):
        return relu(layer_norm(h, *p["ln2"]) @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]

    return residual(residual(residual(Y, talk), read), think)


# @a:decoder
def decoder(tgt, memory, p, n_heads):
    """Target ids so far (T,) and the encoder's vectors → next-token scores (T, V_tgt)."""
    Y = embed(tgt, p["E"]) + sinusoidal(len(tgt), p["E"].shape[1])
    for b in p["blocks"]:
        Y = decoder_block(Y, memory, b, n_heads)
    return layer_norm(Y, *p["ln_f"]) @ p["W_out"]


# @a:translate
def translate(src, params, n_heads, bos, eos, max_len=20):
    """Greedy decoding: encode once, then append the likeliest next word until [EOS]."""
    memory = encoder(src, params["encoder"], n_heads)  # computed once, read at every step
    out = [bos]
    while len(out) < max_len and out[-1] != eos:
        out.append(int(decoder(np.array(out), memory, params["decoder"], n_heads)[-1].argmax()))
    return out[1:]


# @a:demo
if __name__ == "__main__":
    # @a:data
    pairs = make_pairs(1000)
    src_vocab = build_vocab([w for en, _ in pairs for w in en])
    tgt_vocab = build_vocab([w for _, de in pairs for w in de])
    tgt_vocab |= {"[BOS]": len(tgt_vocab), "[EOS]": len(tgt_vocab) + 1}  # start and stop signals
    english, german = pairs[0]
    print(" ".join(english), "→", " ".join(german))
    src = encode(english, src_vocab)
    tgt_in = encode(["[BOS]", *german], tgt_vocab)  # what the decoder is shown
    tgt_out = encode([*german, "[EOS]"], tgt_vocab)  # what it must predict: the same, one step ahead

    # @a:forward
    rng = np.random.default_rng(0)
    d, n_layers, h = 32, 2, 4
    params = init_seq2seq(len(src_vocab), len(tgt_vocab), d, n_layers, rng)
    logits = decoder(tgt_in, encoder(src, params["encoder"], h), params["decoder"], h)
    print(f"{len(src)} English tokens, {len(tgt_in)} German inputs → logits {logits.shape}")
    assert logits.shape == (len(tgt_in), len(tgt_vocab))

    # @a:who-sees-what
    def run(s, t):
        return decoder(t, encoder(s, params["encoder"], h), params["decoder"], h)

    later_src = src.copy()
    later_src[-2] = src_vocab["soup"]  # change the last English word
    assert not np.allclose(run(later_src, tgt_in)[0], logits[0])  # the first German word sees it
    later_tgt = tgt_in.copy()
    later_tgt[5:] = tgt_vocab["reis"]  # change the end of the German input
    assert np.allclose(run(src, later_tgt)[:5], logits[:5])  # earlier German words don't
    print("every German position sees all the English, but only the German before it")

    # @a:untrained
    loss = -np.log(softmax(logits)[np.arange(len(tgt_out)), tgt_out]).mean()
    print(f"untrained loss {loss:.2f} vs log V = {np.log(len(tgt_vocab)):.2f}")
    assert abs(loss - np.log(len(tgt_vocab))) < 0.1
    itos = {i: w for w, i in tgt_vocab.items()}
    out = translate(src, params, h, tgt_vocab["[BOS]"], tgt_vocab["[EOS]"])
    print("untrained translation:", " ".join(itos[i] for i in out))

    # @a:count
    per_encoder_block, per_decoder_block = 12 * d * d, 16 * d * d  # weights only, no biases or norms
    print(f"encoder block ≈ 12·d² = {per_encoder_block:,}; decoder block ≈ 16·d² = {per_decoder_block:,}"
          " (4·d² more, for cross-attention)")
    assert sum(W.size for W in params["decoder"]["blocks"][0]["cross"]) == 4 * d * d
