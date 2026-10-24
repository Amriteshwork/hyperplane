---
title: Adam optimiser
summary: Gradient descent with momentum and a separate, self-tuning step size for every weight. The default optimiser for neural networks.
area: optimisation
prerequisites: [backpropagation, stochastic-gradient-descent, learning-rate]
---

Plain (stochastic) gradient descent uses one learning rate $\eta$ for every weight. That is
a poor fit for networks: some weights have large gradients, others tiny ones, and the loss
surface has ravines, steep in one direction and nearly flat in another. A step small enough
not to bounce across the ravine crawls along it.

**Adam** (adaptive moment estimation, Kingma & Ba 2014) keeps two running averages per
weight, updated at every step $t$ from the mini-batch gradient $g_t$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\,g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2)\,g_t^2 .$$

- $m$ is **momentum**: an average direction, so consistent gradients build up speed and
  noisy ones cancel out;
- $v$ tracks how large this weight's gradients usually are.

Both start at 0, so early on they are too small; dividing by $1 - \beta^t$ corrects that
(**bias correction**). The update is

$$\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t}, \qquad
w \leftarrow w - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.$$

Dividing by $\sqrt{\hat v}$ makes the step roughly **scale-free**: a weight moves by about
$\eta$ per step whether its gradients are $10^{-4}$ or $10^4$. Steep directions get damped,
flat ones get relatively longer steps, and the ravine is no longer a problem.

See it on a ravine: all three start at the same point.

::figure[optimizer-race]

The defaults, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, rarely need changing;
$\eta$ (scikit-learn's `learning_rate_init`, 0.001) still does. Adam often finds a good
solution fast. Well-tuned SGD with momentum sometimes generalises a little better in the
end, which is why both remain in use.
