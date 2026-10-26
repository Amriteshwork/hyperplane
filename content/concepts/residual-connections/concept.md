---
title: Residual connections
summary: Instead of replacing its input, each layer adds a correction to it. The gradient gets an express lane back to the first layer, so very deep networks can train.
area: sequence-models
prerequisites: [backpropagation]
---

Here is a puzzle that stumped deep learning in 2015. Take a network that works, add more
layers, and it gets **worse**, even on the *training* data. That's not overfitting. A deeper
network could always copy the shallower one and make its extra layers do nothing, so it
should never be worse. Yet gradient descent couldn't find that solution. Something about
depth itself was breaking training.

**What breaks: a game of telephone.** Backpropagation multiplies one factor per layer on the
way back (the chain rule). With 50 layers, the gradient reaching layer 1 is a product of 50
factors. If they are a bit below 1, it **vanishes**; a bit above, it **explodes**. In the
code, 50 plain tanh layers deliver a gradient of about $10^{-27}$ to the input. The early
layers receive no signal, so they never learn.

**The fix is one plus sign.** Don't make a layer compute the new representation. Make it
compute a **correction** and add that to its input:

$$x_{\ell+1} = x_\ell + f_\ell(x_\ell).$$

The input skips over the layer (a **skip connection**) and the layer learns only the
**residual**: what's missing. Look at what this does to the gradient:

$$\frac{\partial x_{\ell+1}}{\partial x_\ell} = I + \frac{\partial f_\ell}{\partial x_\ell}.$$

That $I$ is an **express lane**. Whatever happens inside $f$, the gradient flows straight
through the addition untouched. Stack 50 blocks and the product of these factors always
contains the plain identity path, so layer 1 hears the loss as clearly as layer 50. In the
code, the same 50 layers with skips deliver a gradient about the size of the one at the
output.

**Doing nothing becomes easy.** If $f$'s weights are zero, the block is exactly the
identity. A new layer starts out harmless and only changes things once it has learnt
something useful. "Deeper can't be worse" becomes true in practice, not just in theory.
This idea (**ResNet**, He et al., 2015) took image networks from about 20 layers to over 150.

**In a transformer, think of a stream.** The vector $x$ flowing through the residual
connections is called the **residual stream**. Every attention layer and every MLP *reads*
from the stream and *writes* a small update back into it. Nothing is overwritten. Later
layers can still see what the embedding said, plus every update since. That's why research
on how transformers work talks about information being written to and read from the
stream.

One catch, which the PyTorch code hints at: every block *adds* to the stream, so its size
keeps growing with depth. Keeping each layer's input at a sensible scale is the job of the
last ingredient, **layer normalization**.
