---
title: Bias–variance trade-off
summary: Test error = bias² + variance + noise. Simpler models err consistently; flexible ones err erratically.
area: evaluation
prerequisites: [overfitting-underfitting]
---

Imagine training the same kind of model many times, each time on a fresh sample of training
data, and predicting the same point $x$. The expected squared error splits into three parts:

$$\mathbb{E}\big[(y - \hat f(x))^2\big] = \underbrace{\big(f(x) - \mathbb{E}[\hat f(x)]\big)^2}_{\text{bias}^2} + \underbrace{\operatorname{Var}\big[\hat f(x)\big]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{noise}}$$

- **Bias**: how far the *average* prediction is from the truth. A straight line through a
  curve is wrong the same way every time: high bias, which is underfitting.
- **Variance**: how much predictions swing between training samples. A degree-15 polynomial
  wiggles differently for every sample: high variance, which is overfitting.
- **Noise**: randomness in $y$ itself. No model can remove it.

Making a model more flexible usually lowers bias and raises variance, so total test error
traces a U-shape and the best model sits in between. More data lowers variance without
raising bias; regularisation (Ridge, Lasso) trades a little bias for a large drop in variance.

Very large models can break the simple U-shape ("double descent"), but the decomposition
still holds and is the right first mental model.
