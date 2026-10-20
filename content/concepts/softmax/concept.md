---
title: Softmax
summary: Turn one score per class into probabilities that are positive and sum to 1. The multiclass sigmoid.
area: foundations
prerequisites: [sigmoid]
---

With $K$ classes, a model produces one score $z_k$ per class (often called **logits**). The
**softmax** turns them into probabilities:

$$\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}} .$$

Exponentiating makes every score positive; dividing by the total makes them sum to 1. The
largest score gets the largest probability, which is a "soft" version of picking the
maximum (argmax).

Two properties matter in practice:

- **Only differences matter.** Adding the same constant to every score changes nothing.
  So one class's scores could be fixed at 0, and multiclass coefficients are only
  meaningful *relative to each other*.
- **Two classes give the sigmoid.** With scores $(0, z)$, softmax gives
  $\frac{e^z}{1 + e^z} = \sigma(z)$. Multinomial logistic regression is the natural
  extension of binary logistic regression.

For numerical safety, compute the **log-softmax** $z_k - \log\sum_j e^{z_j}$ with the
log-sum-exp trick and exponentiate after; equivalently, subtract the row's maximum score
before exponentiating.
