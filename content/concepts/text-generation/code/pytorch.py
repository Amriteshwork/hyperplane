"""Text generation in PyTorch: train a tiny GPT, measure what temperature costs, then speed it up with a KV cache."""

import re

import torch
import torch.nn.functional as F

# @a:reuse
from language_modeling_pytorch import FOODS, NAMES, make_corpus
from layer_normalization_pytorch import layer_norm
from tokenization_pytorch import build_vocab, decode, encode, windows
from transformer_pytorch import TinyGPT


# @a:filter
def next_token_probs(logits, temperature=1.0, top_k=None, top_p=None):
    """Scores (…, V) → the distribution to draw from: temperature first, then cut the tail."""
    p = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
    sorted_p, order = p.sort(dim=-1, descending=True)
    keep = torch.ones_like(p, dtype=torch.bool)
    if top_k is not None:
        keep &= torch.zeros_like(keep).scatter(-1, order[..., :top_k], True)
    if top_p is not None:
        keep &= (sorted_p.cumsum(-1) - sorted_p < top_p).gather(-1, order.argsort(-1))
    p = p * keep
    return p / p.sum(-1, keepdim=True)


# @a:generate
@torch.no_grad()
def generate(model, ids, n, generator=None, **knobs):
    """Batched generation: ids (B, T) → (B, T + n), drawing each token with the given knobs."""
    for _ in range(n):
        p = next_token_probs(model(ids)[:, -1], **knobs)
        ids = torch.cat([ids, torch.multinomial(p, 1, generator=generator)], dim=1)
    return ids


# @a:cache
@torch.no_grad()
def cached_step(model, new_id, position, cache):
    """One new token → its logits, reading old tokens' keys and values from `cache` (one per block)."""
    x = model.embed(new_id) + model.positions[position]
    for block, past in zip(model.blocks, cache):
        W_Q, W_K, W_V, W_O = block.attn
        h, size = block.n_heads, x.shape[-1] // block.n_heads
        n = layer_norm(x, *block.ln1)
        Q, K, V = ((n @ W).view(h, 1, size) for W in (W_Q, W_K, W_V))
        past["K"] = torch.cat([past["K"], K], dim=1) if "K" in past else K
        past["V"] = torch.cat([past["V"], V], dim=1) if "V" in past else V
        # No mask: the future hasn't been written, so it isn't in the cache to peek at.
        weights = torch.softmax(Q @ past["K"].transpose(-1, -2) / size**0.5, dim=-1)
        x = x + (weights @ past["V"]).view(-1) @ W_O
        x = x + block.mlp(layer_norm(x, *block.ln2))
    return model.unembed(model.ln_f(x))


def is_valid(sentence):
    """Does this follow the corpus's rule: real name and food, then copied correctly?"""
    names, foods = "|".join(NAMES), "|".join(FOODS)
    return re.fullmatch(rf"({names}) likes ({foods}), so \1 eats \2\.", sentence) is not None


# @a:demo
if __name__ == "__main__":
    # @a:train
    torch.manual_seed(0)
    text = make_corpus(1000)
    vocab = build_vocab(text)
    x, y = windows(encode(text, vocab), 24)
    model = TinyGPT(len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for step in range(401):
        batch = torch.randint(len(x), (32,))
        loss = F.cross_entropy(model(x[batch]).reshape(-1, len(vocab)), y[batch].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"trained: loss {loss.item():.3f}")

    # @a:tradeoff
    start = encode(". ", vocab).repeat(200, 1)
    results = {}
    for temperature in (0.01, 0.3, 0.6, 1.0, 1.5, 2.5):
        rng = torch.Generator().manual_seed(0)
        out = generate(model, start, 44, generator=rng, temperature=temperature)
        said = [decode(row, vocab)[2:].split(". ")[0] + "." for row in out]
        valid = sum(map(is_valid, said)) / len(said)
        results[temperature] = (valid, len(set(said)))
        print(f"τ={temperature:<4}  valid {valid:4.0%}   distinct {len(set(said)):3d}   e.g. {said[0]!r}")
    assert results[0.01][1] <= 8  # near-greedy: the same few sentences
    assert results[0.6][0] > results[2.5][0]  # hotter: more variety, less sense

    # @a:cache-check
    ids = encode("zoe likes jam, so", vocab)
    cache = [{} for _ in model.blocks]
    with torch.no_grad():
        for t in range(len(ids)):
            logits = cached_step(model, ids[t], t, cache)
        full = model(ids)[-1]
    assert torch.allclose(logits, full, atol=1e-5)  # same answer, far less work
    print(f"cached and full logits match (largest gap {(logits - full).abs().max():.1e})")

    # @a:cache-work
    T = 256  # a short reply from a real model
    rerun = sum(t**2 for t in range(1, T + 1))  # every token against every earlier one, each time
    cached = sum(range(1, T + 1))  # only the new token's row
    print(f"{T} tokens: {rerun:,} attention scores rerunning, {cached:,} with the cache ({rerun // cached}× fewer)")
    assert rerun / cached > T / 2
