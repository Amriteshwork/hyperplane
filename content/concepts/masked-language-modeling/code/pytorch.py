"""Masked language modelling in PyTorch: the masking recipe, `ignore_index`, and a model that reads both sides."""

import torch
import torch.nn.functional as F

# @a:reuse
from language_modeling_pytorch import make_corpus
from tokenization_pytorch import build_vocab, encode


# @a:add-tokens
def add_tokens(vocab, *tokens):
    """Extend a vocabulary with special tokens such as '[MASK]', numbered after the real ones."""
    return {**vocab, **{t: len(vocab) + i for i, t in enumerate(tokens)}}


# @a:mask
def mask_tokens(ids, mask_id, V, rate=0.15, generator=None):
    """BERT's recipe on real tokens (ids < V). Targets: the original id where picked, −100 elsewhere."""
    picked = (torch.rand(ids.shape, generator=generator) < rate) & (ids < V)
    roll = torch.rand(ids.shape, generator=generator)
    corrupted = ids.clone()
    corrupted[picked & (roll < 0.8)] = mask_id
    swap = picked & (roll >= 0.8) & (roll < 0.9)
    corrupted[swap] = torch.randint(V, (int(swap.sum()),), generator=generator)
    return corrupted, torch.where(picked, ids, -100)  # −100: F.cross_entropy's "ignore me"


# @a:neighbours
class Neighbours(torch.nn.Module):
    """Guess token t from the (possibly corrupted) token on its left, or from both neighbours."""

    def __init__(self, V, d=16, both_sides=True):
        super().__init__()
        self.both_sides = both_sides
        self.embed = torch.nn.Embedding(V, d)
        self.out = torch.nn.Linear(d * (2 if both_sides else 1), V)

    def forward(self, ids):
        sides = [ids[:-2], ids[2:]] if self.both_sides else [ids[:-2]]
        return self.out(torch.cat([self.embed(s) for s in sides], dim=-1))  # for positions 1..T−2


# @a:demo
if __name__ == "__main__":
    # @a:recipe
    rng = torch.Generator().manual_seed(0)
    text = make_corpus(300)
    vocab = build_vocab(text)
    V = len(vocab)
    vocab = add_tokens(vocab, "[MASK]")
    ids = encode(text, vocab)
    corrupted, targets = mask_tokens(ids, vocab["[MASK]"], V, generator=rng)
    picked = targets != -100
    print(f"picked {picked.float().mean():.1%}; of those, "
          f"{(corrupted[picked] == vocab['[MASK]']).float().mean():.0%} became [MASK]")
    assert torch.equal(corrupted[~picked], ids[~picked])

    # @a:ignore-index
    logits = torch.randn(len(ids), V + 1, generator=rng)
    assert torch.isclose(F.cross_entropy(logits, targets), F.cross_entropy(logits[picked], ids[picked]))
    print("ignore_index=-100 scores the picked positions only")

    # @a:train
    torch.manual_seed(0)
    split = int(0.9 * len(ids))
    val_in, val_targets = mask_tokens(ids[split:], vocab["[MASK]"], V, generator=rng)
    for both_sides in (False, True):
        model = Neighbours(V + 1, both_sides=both_sides)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        for step in range(200):
            x, y = mask_tokens(ids[:split], vocab["[MASK]"], V, generator=rng)  # fresh holes each step
            loss = F.cross_entropy(model(x), y[1:-1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val_loss = F.cross_entropy(model(val_in), val_targets[1:-1]).item()
        print(f"{'both sides' if both_sides else 'left only '}: validation loss {val_loss:.2f} nats")
        if not both_sides:
            left_loss = val_loss
    assert val_loss < left_loss * 0.6  # reading both ways fills the blanks far better
