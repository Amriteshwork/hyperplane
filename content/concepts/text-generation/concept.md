---
title: Text generation
summary: A model only gives you odds for the next token. How you pick from those odds decides whether you get dull, sensible or unhinged text. Temperature, top-k, top-p and beam search, and the cache that makes it fast.
area: sequence-models
prerequisites: [transformer]
videos:
  beam-vs-greedy:
    title: Why the best first word can lose
    youtube: LintbbjM7KQ
---

You've trained a GPT. Ask it for the next token and it doesn't reply with a word. It replies
with a **probability for every word it knows**. Something else has to choose one, append
it, and ask again:

$$\text{logits} \;\to\; \textbf{a rule that picks one token} \;\to\; \text{append} \;\to\; \text{repeat.}$$

That rule isn't part of the model. It has no weights and nothing is learnt. Yet the same
trained model can sound robotic or drunk depending on it. This page is about that rule.

### Four knobs on one distribution

Every sampling method does one of two things: it **reshapes** the odds, or it **cuts off**
the unlikely tail. Try them:

::figure[sampling-explorer]

- **Temperature** $\tau$ divides the scores before the softmax,
  $p_k \propto \exp(z_k/\tau)$. Low sharpens the odds; high flattens them. $\tau \to 0$ is
  **greedy**: always take the top token.
- **Top-k** keeps the $k$ likeliest tokens and drops the rest. The cut is the same size
  whether the model is sure or lost.
- **Top-p** (nucleus sampling) keeps the smallest set whose probabilities add up to $p$.
  When the model is sure that may be a single token, and when it's unsure it may be
  hundreds. The cut adapts. That's why it's a common default.

After a cut, the survivors are divided by their total so they still sum to 1.

### What temperature really costs

"Creativity slider" is a nice phrase. We can measure it. Train the tiny GPT on *"zoe likes
jam, so zoe eats jam."*, generate 200 sentences at each temperature, and ask two questions:
does the sentence follow the rule, and how many different sentences did we get?

::figure[temperature-tradeoff]

These are real numbers from this page's PyTorch code. Near $\tau=0$, every one of the 200
sentences is valid and **all 200 are the same sentence**. At $\tau = 2.5$ they're all
different and **none is valid**. There's no free lunch: you're choosing a point on that curve.

### Greedy can walk past the best answer

For a task with one right answer, such as translation, you don't want variety. You want the
**most likely whole sentence**. Greedy seems to give you that, but it doesn't, because the
best first word can lead somewhere bad. Watch it happen:

::video[beam-vs-greedy]

**Beam search** keeps the $B$ best partial sentences at every step, grows each by one token,
and keeps the best $B$ of the results. Scores are added logs:
$\log P(\text{sequence}) = \sum_t \log P(x_t \mid x_{<t})$. Step through it, then change
the width:

::figure[beam-search-tree]

$B=1$ is exactly greedy. A $B$ as large as every possible sequence would be exhaustive,
and that's hopeless at $V^{T}$ sequences. A small $B$ is a good bargain, not a guarantee:
it can still miss the best answer.

**So why doesn't your chatbot use beam search?** For open-ended writing, the most likely
text turns out to be bland and repetitive; Holtzman et al. (2019) showed that human writing
isn't the highest-probability text. So translation tends to use beam search, and chat
tends to use sampling with temperature and top-p.

### Making it fast: the KV cache

The obvious loop reruns the whole model for every new token, and most of that repeats old
work. Earlier tokens can't see later ones because of the causal mask, so **their keys and
values never change**. Save them, and each new token only needs to compute its own vectors
and compare itself against the saved ones.

::figure[kv-cache]

The saving grows with length. For $T$ tokens, rerunning computes $\sum_t t^2 \approx T^3/3$
attention scores and caching computes $\sum_t t \approx T^2/2$. The price is memory, which
is one reason long conversations cost more to serve.

**What the code shows.** The NumPy file writes one `next_token_probs` for all three knobs,
checks that $\tau=1$ changes nothing, and has a beam of 2 find the 0.36 path that greedy
misses at 0.20. The PyTorch file trains the tiny GPT in about 15 seconds and produces the
temperature numbers above. It then adds `cached_step` and checks that its logits match a
full forward pass to within rounding.
