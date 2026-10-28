"""BERT in PyTorch: pretrain a tiny encoder by filling in blanks, and watch it read the answer from the right."""

import re

import torch
import torch.nn.functional as F

# @a:reuse
from language_modeling_pytorch import make_corpus
from masked_language_modeling_pytorch import add_tokens, mask_tokens
from positional_encoding_pytorch import sinusoidal
from tokenization_pytorch import build_vocab, encode
from transformer_pytorch import Block


# @a:words
def words(text):
    """Word-level tokens: 'zoe likes jam, so' → ['zoe', 'likes', 'jam', ',', 'so']."""
    return re.findall(r"\w+|[,.]", text)


# @a:model
class TinyBERT(torch.nn.Module):
    """Token ids (…, T) → one vector per token (…, T, d), every token reading the whole input."""

    def __init__(self, V, d=32, n_heads=4, n_layers=2, context=16):
        super().__init__()
        self.embed = torch.nn.Embedding(V, d)
        self.segment = torch.nn.Embedding(2, d)  # sentence A or B
        self.register_buffer("positions", sinusoidal(context, d))
        self.blocks = torch.nn.Sequential(*[Block(d, n_heads, causal=False) for _ in range(n_layers)])
        self.ln_f = torch.nn.LayerNorm(d)
        self.mlm_head = torch.nn.Linear(d, V)  # pretraining only: a score per vocabulary word

    def forward(self, ids, segments=None):
        segments = torch.zeros_like(ids) if segments is None else segments
        x = self.embed(ids) + self.segment(segments) + self.positions[: ids.shape[-1]]
        return self.ln_f(self.blocks(x))


# @a:demo
if __name__ == "__main__":
    # @a:data
    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(0)
    sentences = [s + "." for s in make_corpus(2000).split(". ") if s]
    vocab = build_vocab(sorted({w for s in sentences for w in words(s)}))
    V = len(vocab)
    vocab = add_tokens(vocab, "[CLS]", "[SEP]", "[MASK]")

    def batch(texts):
        """[CLS] words [SEP], one row per sentence (all sentences here have the same length)."""
        return torch.stack([encode(["[CLS]", *words(t), "[SEP]"], vocab) for t in texts])

    data = batch(sentences)

    # @a:pretrain
    model = TinyBERT(len(vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for step in range(501):
        x, y = mask_tokens(data[torch.randint(len(data), (64,), generator=rng)], vocab["[MASK]"], V,
                           generator=rng)
        loss = F.cross_entropy(model.mlm_head(model(x)).reshape(-1, len(vocab)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"step {step:4d}: masked-word loss {loss.item():.3f}")

    # @a:fill
    itos = {i: w for w, i in vocab.items()}
    filled = 0
    for name, food in [("zoe", "jam"), ("raj", "figs"), ("kim", "soup"), ("ian", "nuts")]:
        x = batch([f"{name} likes {food}, so {name} eats {food}."])
        x[0, 1] = vocab["[MASK]"]  # the first word: nothing on its left to go on
        with torch.no_grad():
            guess = itos[model.mlm_head(model(x))[0, 1].argmax().item()]
        print(f"  [MASK] likes {food}, so {name} eats {food}.  →  {guess}")
        filled += guess == name
    assert filled >= 3  # it reads the answer from the right; a left-to-right model can't

    # @a:both-hidden
    x = batch(["zoe likes jam, so zoe eats jam."])
    x[0, [1, 6]] = vocab["[MASK]"]  # hide both copies of the name
    with torch.no_grad():
        p = torch.softmax(model.mlm_head(model(x))[0, 1], dim=-1)
    print(f"both names hidden: its best guess, {itos[p.argmax().item()]!r}, gets only {p.max():.0%}")
    assert p.max() < 0.5  # no copy to read, so it can only spread its bets over the names
