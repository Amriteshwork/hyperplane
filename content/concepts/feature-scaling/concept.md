---
title: Feature scaling
summary: Put features on comparable scales (e.g. mean 0, std 1) so penalties, distances and gradient descent treat them fairly.
area: features
prerequisites: [train-test-split, probability]
---

Features arrive in arbitrary units: income in dollars, age in years, a ratio between 0
and 1. Some models don't care (plain least squares, decision trees), but many do:

- **Penalised models** (Ridge, Lasso) penalise coefficient size, and a coefficient's size
  depends on its feature's units. A feature measured in kilometres needs a 1000× larger
  weight than the same feature in metres, so the penalty hits it 10⁶× harder.
- **Distance-based models** (k-nearest neighbours, SVMs, k-means): the feature with the
  biggest numbers dominates every distance.
- **Gradient descent** converges slowly when features have very different ranges.

The two common scalers:

$$\text{standardise: } z = \frac{x - \mu}{\sigma} \qquad\qquad \text{min–max: } z = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Standardisation (mean 0, standard deviation 1) is the default for linear models.

**Fit the scaler on the training set only**, then apply the same $\mu$ and $\sigma$ to the
test set and to new data. Computing them on all the data leaks information from the test
set. Saving the scaler inside the model pipeline guarantees it's applied consistently.
