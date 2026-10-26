---
title: Layer normalization
summary: Rescale every token's vector to mean 0 and variance 1 before each layer reads it, so no layer is ever surprised by inputs that are far too big or too small.
area: sequence-models
prerequisites: [feature-scaling, neurons, residual-connections]
---

Feature scaling taught us that models learn best when their inputs are on a sensible
scale, and that standardising, $(x - \mu)/\sigma$, fixes it. Inside a deep network the same
problem comes back at every layer, and it changes during training. Layer 12's inputs are
layer 11's outputs. Every time layer 11's weights move, the scale of what layer 12 sees
shifts. And in a transformer the residual stream *grows* with depth, since every block adds
to it. The code shows it tripling in size over 48 blocks.

**Layer normalization standardises inside the network.** Take one token's vector
$x \in \mathbb R^d$ and standardise it across its own features:

$$\mu = \frac1d\sum_{j=1}^d x_j, \qquad \sigma^2 = \frac1d\sum_{j=1}^d (x_j - \mu)^2, \qquad
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta .$$

Every token comes out with mean 0 and variance 1, whatever the scale it went in with.
Then two learned vectors, a **gain** $\gamma$ and a **bias** $\beta$ (one number per
feature), let the network rescale and shift, in case raw standardised values aren't what
the next layer wants. $\epsilon$ (say $10^{-5}$) prevents dividing by zero when a vector is
constant.

**Rows, not columns. This is the crucial difference.** Feature scaling standardises each
*column* (one feature, across all examples), so its statistics depend on the whole
dataset. Layer norm standardises each *row* (one token, across its own features), so its
statistics depend on **nothing but that token**:

- the rest of the batch can't affect it, so a batch of 1 works fine;
- it does the same thing in training and at inference;
- it handles sequences of any length, since every token is normalised alone.

**Batch normalization**, its older cousin from image networks, uses *column* statistics
over the batch. A token's output then depends on which other sentences happened to share
its batch. The PyTorch code puts the same sentence in two different batches: layer norm's
output is identical, batch norm's isn't. That's why transformers use layer norm.

**Where it goes: pre-LN.** The original transformer normalised *after* each residual
addition (**post-LN**). Almost every model since normalises the branch's *input* instead:

$$x \leftarrow x + \text{Attention}(\text{LN}(x)), \qquad x \leftarrow x + \text{MLP}(\text{LN}(x)).$$

This **pre-LN** layout leaves the residual stream itself untouched (the gradient express
lane stays clean) while every layer still reads a standardised version of it. However big
the stream grows, what the next branch sees always has length $\sqrt d$. Pre-LN trains
more stably and doesn't need a long learning-rate warm-up. One more LN after the last block
tidies the stream before the final prediction.

(Many recent LLMs use **RMSNorm**, layer norm without subtracting the mean or adding
$\beta$: $x / \sqrt{\overline{x^2} + \epsilon}\cdot\gamma$. It's slightly cheaper and works as
well.)
