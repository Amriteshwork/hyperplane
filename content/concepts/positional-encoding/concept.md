---
title: Positional encoding
summary: Attention sees a bag of tokens, not a sequence. Adding a unique "where am I" signal to every embedding puts word order back.
area: sequence-models
prerequisites: [embeddings, self-attention]
---

*"Dog bites man"* is not news. *"Man bites dog"* is. Same three words, completely
different meaning, and plain self-attention **can't tell them apart**. Attention compares
every token with every other token, but nothing in $\operatorname{softmax}(QK^\top/\sqrt
d)V$ mentions position. Shuffle the tokens and you shuffle the outputs, nothing more. To
attention, a sentence is a bag of words.

**The fix: stamp each token with its position.** Before the first layer, add a position
vector to each token's embedding:

$$x_t = \text{embed}(\text{token}_t) + p_t .$$

Now "dog" at position 0 and "dog" at position 2 are different vectors, and the queries and
keys can depend on where things are. Two ways to choose $p_t$:

- **Learn it.** Give each position its own row in a second embedding table, exactly like
  tokens. GPT-2 does this. Simple, but it can't handle positions it never saw in training.
- **Compute it.** The original transformer used **sinusoids**.

**Sinusoidal encoding: a row of clocks.** Pair up the $d$ dimensions. Pair $i$ is a clock
hand spinning at its own speed $\omega_i$, and $p_t$ records where every hand points at
time $t$:

$$p_{t,2i} = \sin(t\,\omega_i), \qquad p_{t,2i+1} = \cos(t\,\omega_i), \qquad
\omega_i = \frac{1}{10000^{2i/d}} .$$

The first pair spins fast (one radian per token), like a seconds hand. The last pairs creep
along over thousands of tokens, like an hour hand. Read together, they give every position
a unique fingerprint, the way a clock's hands give every moment of the day a unique
reading. It's binary counting, made smooth: the lowest bit flips fastest.

::figure[positional-encoding-explorer]

**Why sines and cosines, specifically?** Because moving $k$ steps forward is the **same
rotation** for every position. Each $(\sin, \cos)$ pair just turns by angle $k\omega_i$:

$$p_{t+k} = p_t\, R_k \qquad \text{for every } t,$$

where $R_k$ is a fixed block of $2\times2$ rotations that doesn't depend on $t$. So "the
token 3 to my left" looks the same everywhere in the sentence, and a linear layer can learn
to find it. Relatedly, $p_t \cdot p_{t+k}$ depends only on the distance $k$. The code
checks both.

**Modern models rotate instead of add.** Llama, Mistral and many other recent LLMs use
**rotary position embeddings (RoPE)**. Rather than adding $p_t$ to the embedding, they
*rotate* each query and key by angle $t\omega_i$, pair by pair. The score
$q_t \cdot k_s$ then depends only on $q$, $k$ and the **distance** $t - s$, never on
absolute position. Attention becomes relative by construction, and nothing is added to the
token's own vector. The PyTorch code implements RoPE in eight lines and checks that
property.

With word order restored, the transformer has everything it needs to read. What's left is
making it deep without it falling apart: **residual connections** and **layer
normalization**.
