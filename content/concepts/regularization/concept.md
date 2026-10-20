---
title: Regularisation
summary: Add a penalty for model complexity to the loss, so the fit prefers simpler explanations of the data.
area: features
prerequisites: [overfitting-underfitting, loss-function]
---

A flexible model can always lower its training error by bending to fit the noise. Its
weights grow huge as different features cancel each other out to hit every point.
**Regularisation** changes what "best fit" means by adding a penalty on the weights to the
loss:

$$\text{loss}(w) = \underbrace{\frac{1}{n}\sum_i (y_i - \hat y_i)^2}_{\text{fit the data}} + \underbrace{\lambda\, \Omega(w)}_{\text{stay simple}}$$

- $\Omega(w)$ measures size: $\sum_j w_j^2$ for **L2** (Ridge), $\sum_j |w_j|$ for **L1**
  (Lasso).
- $\lambda \ge 0$ (`alpha` in scikit-learn) sets the trade-off. At 0 you get the
  unregularised model; as it grows, weights shrink towards 0 and the model gets simpler.

In bias–variance terms, the penalty adds a little **bias** (the weights are pulled away
from the best training fit) to remove a lot of **variance** (they stop swinging with every
sample of noise). The best $\lambda$ sits in between and is chosen on held-out data, usually
with cross-validation.

Two practical rules: scale the features first, so the penalty treats them alike, and never
penalise the intercept.
