---
title: Combined penalties (Elastic Net)
summary: Add an L1 and an L2 penalty together. L1 still drops features; L2 makes correlated features share the weight.
area: features
prerequisites: [l1-regularization, sparsity]
---

L1 and L2 penalties each fix something the other gets wrong. L2 (Ridge) keeps every
feature. L1 (Lasso) drops features, but from a group of correlated features it keeps one
almost at random. **Elastic Net** adds both, mixed by $\rho$ (`l1_ratio`):

$$\min_w \; \frac{1}{2n}\lVert y - Xw \rVert^2 + \alpha\rho\,\lVert w \rVert_1 + \frac{\alpha(1-\rho)}{2}\lVert w \rVert_2^2 .$$

- $\rho = 1$ is the Lasso; $\rho \to 0$ approaches Ridge. $\alpha$ sets the total strength.
- The coordinate-descent update changes in one place: the L2 part adds to the denominator,

  $$w_j = \frac{S\!\left(\tfrac{1}{n}x_j^\top r_{-j},\ \alpha\rho\right)}{\tfrac{1}{n}\lVert x_j\rVert^2 + \alpha(1-\rho)} .$$

  The soft threshold still decides *whether* a feature is used; the bigger denominator
  shrinks it further.

Why the mix keeps groups together: for a set of near-identical features, L1 charges the
same whatever way their total weight is split, so it has no reason to share. L2 charges
$\sum_j w_j^2$, which is smallest when the weight is split **evenly**. So Elastic Net gives
correlated features similar weights and keeps or drops them together: the **grouping
effect**. As a bonus, the squared term makes the loss strictly convex, so the answer is
unique.

The price is a second hyperparameter. $\alpha$ and $\rho$ interact, so they are tuned
together, by searching pairs with cross-validation.
