---
title: Training a large language model
summary: The architecture is simple. Keeping a huge training run from blowing up is the hard part. Warmup, cosine decay, gradient clipping, dropout, weight tying and label smoothing, each tested honestly on a tiny GPT.
area: sequence-models
prerequisites: [transformer]
videos:
  warmup-rescue:
    title: The same model, the same peak, one schedule
    youtube: vjbmYZpvdwA
---

Your tiny GPT trains in 25 seconds with no special care. A real one trains for weeks on
thousands of GPUs at a cost of millions of dollars. Nobody wants to find out on day nine that
it diverged. So real training runs add a handful of small, cheap safety tricks.

This page tries each one on the tiny GPT and reports what happened, **including when it didn't
help**. Small models don't need most of these, and it's better to know that than to think
each trick is magic.

### Warmup: start gently

At the very start, the weights are random, so the gradients are noisy and point in poor
directions. A big step taken on that noise can land the model somewhere it never recovers from.
So start with a tiny learning rate, raise it steadily, then let it glide back to zero:

$$\eta(t) = \begin{cases}\eta_{\max}\,\dfrac{t+1}{t_w} & t < t_w \quad \textit{warmup}\\[6pt] \tfrac12\,\eta_{\max}\left(1 + \cos\dfrac{\pi\,(t - t_w)}{T - t_w}\right) & t \ge t_w \quad \textit{cosine decay}\end{cases}$$

::video[warmup-rescue]

Here's the result that surprised us. Set the learning rate to 0.1, ten times too high, and
training breaks and stays broken. Keep that same peak but add a warmup, and it trains fine:

::figure[lr-warmup]

### Gradient clipping: a seatbelt

Now and then a batch produces an enormous gradient. Clipping shrinks a gradient that's too long,
without changing its direction:

$$g \leftarrow g \cdot \min\!\left(1, \frac{c}{\lVert g \rVert}\right).$$

::figure[gradient-clip]

**In our test, clipping didn't rescue the too-high learning rate** (validation loss 1.55, against
1.58 without it). That's a real finding: Adam already scales its steps, so a shorter gradient
doesn't help as much as you might guess. Clipping protects against rare spikes. It doesn't fix a
learning rate that's wrong every step. It really earns its place in
[recurrent networks](/concepts/recurrent-neural-networks) and in long runs, where one bad batch
could cost days.

### Dropout: don't lean on any one neuron

During training, zero a random share of the numbers, and scale up the rest so the average stays
the same. The network can't rely on any single unit, so it has to spread what it knows around.
At test time it's switched off.

::figure[dropout-mask]

**On our tiny GPT, dropout made things worse.** Trained on only 120 windows, validation loss went
from 2.9 to 3.5 with a rate of 0.2. A small model on small data is short of capacity, not
oversupplied with it. GPT-2 used 0.1, while many recent large models use little or none,
because the flood of training data does most of the regularising.

### Weight tying: one table, two directions

The embedding table turns token ids into vectors. The final layer turns vectors back into token
scores. Those are mirror-image jobs, so use **one** matrix for both:

$$\text{logits} = x\,E^\top.$$

GPT-2 saves $50{,}257 \times 768 = 38{,}597{,}376$ parameters this way, about 31% of the model.
On our model it saved 768 parameters and validation loss went up a little (0.33 to 0.36). The
saving matters at scale: that's where the output table is a large share of the whole model.

### Label smoothing: don't be 100% sure

Instead of a target of 1 on the right token, aim for $1 - \epsilon$ and share $\epsilon$ across
all the tokens. The model never gets rewarded for infinite confidence. *Attention Is All You Need*
used $\epsilon = 0.1$. Many LLM training runs now leave it out.

### What really matters at scale

Those tricks fit in a few lines. Most of the real effort is everywhere else:

- **Data.** Cleaning, removing duplicates, and tokenizing a huge slice of the internet.
- **Mixed precision and distributed training**, to spread one model across thousands of GPUs.
- **Checkpointing**, so that a crash on day nine doesn't mean starting again.
- **Evaluation**, because a falling loss doesn't prove the model got better at what you want.

**What the code shows.** The NumPy file writes each trick in a few lines and checks its behaviour:
clipping keeps the direction, dropout keeps the mean, and the schedule peaks where it should. The
PyTorch file trains the tiny GPT five times from one seed, about 35 seconds in total, to produce
the numbers above.
