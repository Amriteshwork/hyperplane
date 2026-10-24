---
title: Stochastic gradient descent (SGD)
summary: Gradient descent that steps on one row (or a small batch) at a time instead of the whole dataset.
area: optimisation
prerequisites: [learning-rate, mean-squared-error]
---

Gradient descent computes the gradient of the loss over **every** row before taking a single
step. On a million rows, that's a million gradient evaluations per step.

The loss is an average over rows, $L(\theta) = \frac1n\sum_i \ell_i(\theta)$, so its gradient
is the average of the rows' gradients. **Stochastic gradient descent** estimates it from one
randomly chosen row and steps right away:

$$\theta \leftarrow \theta - \eta \, \nabla_\theta \ell_i(\theta), \qquad i \text{ random.}$$

One row's gradient is noisy, but it is right **on average** (an unbiased estimate), so the
noise mostly cancels over many steps. In the time gradient descent takes one careful step,
SGD has taken $n$ rough ones, and is usually much closer to the minimum.

**Mini-batch** SGD averages the gradient over a small batch of rows (say 32): less noise
per step, and fast on vectorised hardware. It's how almost every neural network is trained.

The price of the noise: SGD never quite stops moving. To settle at the minimum it needs a
**learning rate that shrinks** over time, and its result depends a little on the random
order the rows were visited in.
