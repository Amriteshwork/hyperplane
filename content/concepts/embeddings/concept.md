---
title: Embeddings
summary: Give every token a learned vector, so that tokens with similar meanings end up pointing in similar directions.
area: sequence-models
prerequisites: [tokenization, vectors, matrix-multiplication]
---

Tokenization turned "cat" into id 1 and "dog" into id 7. But a network that multiplies by
those numbers would believe dog = 7 × cat. Ids are **names**, not quantities.

The classic fix is **one-hot encoding**: token $i$ becomes a vector of $V$ zeros with a 1 at
position $i$. Now nothing is "bigger" than anything else. But every pair of tokens is
equally different (all one-hot vectors are perpendicular), so "cat" is as far from "dog" as
from "democracy". And with $V = 50{,}000$ each vector is enormous and almost all zeros.

**An embedding** gives every token a short, dense vector instead, say $d = 768$ numbers, and
lets the network **learn** those numbers. Stack them as the rows of a matrix:

$$E \in \mathbb{R}^{V \times d}, \qquad \text{embed}(i) = E_{i,:}\ \ (\text{row } i).$$

That's the whole mechanism: **a lookup table**. Here is the neat part. Looking up row $i$ is
exactly the one-hot vector times $E$:

$$\text{onehot}(i)^\top E = E_{i,:}.$$

So an embedding layer is an ordinary linear layer whose inputs happen to be one-hot. It
trains with ordinary backpropagation. The library just skips multiplying by thousands of
zeros. One consequence shows up in the code: a training step only changes the rows of the
tokens that appeared in the batch.

**Meaning becomes geometry.** Nobody tells the network what the $d$ numbers should mean.
But to predict the next word well, it pays to put words that behave alike near each
other, because then they can share what has been learnt about them. After training,
similarity is measured by the angle between vectors (**cosine similarity**), and
*directions* start to carry meaning. The famous example from word2vec:

$$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \;\approx\; \vec{\text{queen}}.$$

The step from "man" to "woman" points roughly the same way as the step from "king" to
"queen". The code builds a tiny hand-made embedding where you can see this exactly. In real
models the directions are messier, but the idea holds.

**Where this goes next.** In a transformer, the embedding of "bank" is the same whether the
sentence is about rivers or money: the table has one row per token, so it ignores context.
Fixing that, by letting each token's vector absorb information from the tokens around it,
is the job of **self-attention**.
