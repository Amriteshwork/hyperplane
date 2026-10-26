"""A transformer in PyTorch: a tiny GPT built from the earlier concepts' functions, trained to beat the bigram."""

import torch
import torch.nn.functional as F

# @a:reuse
from causal_masking_pytorch import causal_mask
from language_modeling_pytorch import fit_bigram, generate, make_corpus
from layer_normalization_pytorch import layer_norm
from multi_head_attention_pytorch import multi_head_attention
from positional_encoding_pytorch import sinusoidal
from tokenization_pytorch import build_vocab, decode, encode, windows


# @a:block
class Block(torch.nn.Module):
    """Pre-LN block: x + attention(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.attn = torch.nn.Parameter(torch.randn(4, d, d) / d**0.5)  # W_Q, W_K, W_V, W_O
        gamma_beta = torch.stack([torch.ones(d), torch.zeros(d)])  # each norm's (γ, β)
        self.ln1, self.ln2 = torch.nn.Parameter(gamma_beta.clone()), torch.nn.Parameter(gamma_beta.clone())
        self.mlp = torch.nn.Sequential(torch.nn.Linear(d, 4 * d), torch.nn.GELU(), torch.nn.Linear(4 * d, d))

    def forward(self, x):
        mask = causal_mask(x.shape[-2])
        x = x + multi_head_attention(layer_norm(x, *self.ln1), *self.attn, self.n_heads, mask)[0]
        return x + self.mlp(layer_norm(x, *self.ln2))


# @a:gpt
class TinyGPT(torch.nn.Module):
    """Token ids (…, T) → next-token logits (…, T, V)."""

    def __init__(self, V, d=32, n_heads=4, n_layers=2, context=24):
        super().__init__()
        self.context = context
        self.embed = torch.nn.Embedding(V, d)
        self.register_buffer("positions", sinusoidal(context, d))
        self.blocks = torch.nn.Sequential(*[Block(d, n_heads) for _ in range(n_layers)])
        self.ln_f = torch.nn.LayerNorm(d)
        self.unembed = torch.nn.Linear(d, V)

    def forward(self, ids):
        ids = ids[..., -self.context :]  # it only ever sees the last `context` tokens
        x = self.embed(ids) + self.positions[: ids.shape[-1]]
        return self.unembed(self.ln_f(self.blocks(x)))


# @a:demo
if __name__ == "__main__":
    # @a:data
    torch.manual_seed(0)
    text = make_corpus(1000)
    vocab = build_vocab(text)
    ids = encode(text, vocab)
    split = int(0.9 * len(ids))
    x_train, y_train = windows(ids[:split], 24)
    x_val, y_val = windows(ids[split:], 24)
    bigram = fit_bigram(ids[:split], len(vocab))
    bigram_loss = -bigram[ids[split:-1], ids[split + 1 :]].mean().item()

    # @a:train
    model = TinyGPT(len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for step in range(601):
        batch = torch.randint(len(x_train), (32,))  # 32 random windows of 24 characters
        logits = model(x_train[batch])
        loss = F.cross_entropy(logits.reshape(-1, len(vocab)), y_train[batch].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"step {step:3d}: loss {loss.item():.3f}")

    # @a:evaluate
    with torch.no_grad():
        every = slice(None, None, 24)  # non-overlapping validation windows
        val_loss = F.cross_entropy(model(x_val[every]).reshape(-1, len(vocab)),
                                   y_val[every].reshape(-1)).item()
    print(f"validation loss: transformer {val_loss:.3f} vs bigram {bigram_loss:.3f}")
    assert val_loss < bigram_loss / 3

    # @a:copy
    rng = torch.Generator().manual_seed(0)
    copied = 0
    for name, food in [("zoe", "jam"), ("raj", "figs"), ("kim", "soup"), ("ian", "nuts")]:
        prompt = encode(f"{name} likes {food}, so ", vocab)
        out = decode(generate(model, prompt, 14, temperature=0.5, generator=rng), vocab)
        print(f"  {out!r}")
        copied += out.startswith(f"{name} likes {food}, so {name} eats {food}")
    assert copied >= 3  # it looks back and copies: no bigram could
