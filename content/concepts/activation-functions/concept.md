---
title: Activation functions
summary: The bend each neuron applies to its weighted sum. Without it, any stack of layers is one linear model.
area: neural-networks
prerequisites: [neurons, derivatives]
---

A layer computes $f(XW + b)$. The **activation** $f$ is applied to every entry
separately, and it has to be **non-linear**: without it, two layers collapse into one,

$$(XW_1 + b_1)W_2 + b_2 = X\underbrace{(W_1W_2)}_{W} + \underbrace{(b_1W_2 + b_2)}_{b},$$

so a hundred linear layers are still a linear model. The bend is what lets a network fit
curves and interactions.

To see how, take one hidden layer of ReLUs: each neuron contributes one bent line, and
the output adds them up.

::figure[relu-sum]

The common choices, and their slopes (backpropagation multiplies by the slope at every
layer):

| name | $f(z)$ | $f'(z)$ | range |
|---|---|---|---|
| ReLU | $\max(0, z)$ | $1$ if $z > 0$, else $0$ | $[0, \infty)$ |
| tanh | $\tanh z$ | $1 - \tanh^2 z$ | $(-1, 1)$ |
| logistic (sigmoid) | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $\sigma(z)(1 - \sigma(z))$ | $(0, 1)$ |
| identity | $z$ | $1$ | any |

Try each one, and watch what happens to the slope it passes back:

::figure[activation-explorer]

**Saturation and vanishing gradients.** The sigmoid's slope is at most $\tfrac14$ (at
$z = 0$) and nearly 0 in its tails, tanh's at most 1. Backpropagation multiplies one slope
per layer, so through many sigmoid layers the gradient shrinks like $(\tfrac14)^L$ and the
early layers barely learn. Large inputs make it worse: a neuron whose weighted sum sits deep
in a tail has almost no slope. That is one reason features are standardised, and why weights
start small.

**ReLU** keeps a slope of exactly 1 wherever it is active, so gradients pass through deep
networks intact, and it is cheap. Its failure mode is the **dead ReLU**: a neuron whose
weighted sum is negative for every row outputs 0, has slope 0, and never recovers. It is the
default for hidden layers (scikit-learn's too).

The **output** layer's activation is chosen by the task, not tuned: identity for regression
(any number), sigmoid for two classes, softmax for more.
