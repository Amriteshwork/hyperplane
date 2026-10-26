"""Embeddings in PyTorch: nn.Embedding is a trainable lookup table, and usage shapes meaning."""

import torch

# @a:reuse
from tokenization_pytorch import build_vocab, encode

# @a:lookup
torch.manual_seed(0)
vocab = build_vocab("abcdefghijklmnopqrstuvwxyz ")
ids = encode("the cat sat", vocab)
table = torch.nn.Embedding(len(vocab), 4)  # 27 rows of 4 learnable numbers
X = table(ids)
one_hot = torch.nn.functional.one_hot(ids, len(vocab)).float()
assert torch.allclose(X, one_hot @ table.weight)  # a lookup is a one-hot matmul
assert torch.equal(X[0], X[6])  # both t's get the same vector: no context yet

# @a:gradient
X.sum().backward()
used = torch.zeros(len(vocab), dtype=torch.bool)
used[ids] = True
assert torch.all(table.weight.grad[~used] == 0)  # rows not in the batch get no gradient
print(f"{int(used.sum())} of {len(vocab)} rows have a gradient")

# @a:corpus
sentences = ["the cat eats fish", "the dog eats meat", "the cat sleeps", "the dog sleeps",
             "a car needs fuel", "a bus needs fuel", "a car drives", "a bus drives"]
words = [s.split() for s in sentences]
vocab = build_vocab(sorted({w for s in words for w in s}))
pairs = torch.tensor([(vocab[a], vocab[b]) for s in words for a, b in zip(s, s[1:])])

# @a:train
table = torch.nn.Embedding(len(vocab), 8)
predict_next = torch.nn.Linear(8, len(vocab))
params = [*table.parameters(), *predict_next.parameters()]
optimizer = torch.optim.Adam(params, lr=0.05, weight_decay=1e-3)
for step in range(500):
    loss = torch.nn.functional.cross_entropy(predict_next(table(pairs[:, 0])), pairs[:, 1])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# @a:similarity
def similarity(a, b):
    """Cosine similarity of two words' learned vectors."""
    return torch.cosine_similarity(table.weight[vocab[a]], table.weight[vocab[b]], dim=0).item()


print(f"cat~dog {similarity('cat', 'dog'):.2f}   car~bus {similarity('car', 'bus'):.2f}   "
      f"cat~car {similarity('cat', 'car'):.2f}")
assert similarity("cat", "dog") > 0.9 and similarity("car", "bus") > 0.9  # same company kept
assert similarity("cat", "car") < 0.5  # different company, different direction
