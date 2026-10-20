---
title: Gradient descent
summary: Minimise a loss by repeatedly stepping the parameters downhill along its gradient.
area: optimisation
prerequisites: [loss-function, derivatives]
---

The gradient of the loss points in the direction in which the loss increases fastest.
Gradient descent starts from some parameters and repeatedly takes a small step the other
way:

$$\theta \leftarrow \theta - \eta \, \nabla_\theta L(\theta)$$

The **learning rate** $\eta$ sets the step size. Too large and the steps overshoot and the
loss explodes; too small and training crawls. Standardising features helps, because it makes
the loss surface rounder and one learning rate suits every direction.

For linear regression gradient descent reaches the same answer as the normal equation. Its
value is that it works for almost any differentiable model, and its stochastic variant (SGD)
works on data too large to fit in memory.
