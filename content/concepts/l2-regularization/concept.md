---
title: L2 regularisation (Ridge)
summary: Penalise the sum of squared weights. Coefficients shrink smoothly, and correlated features share credit.
area: features
prerequisites: [regularization, normal-equation]
---

L2 regularisation adds the squared length of the weight vector to the squared error:

$$\min_w \; \lVert y - Xw \rVert^2 + \alpha \lVert w \rVert_2^2, \qquad \lVert w \rVert_2^2 = \sum_j w_j^2 .$$

Linear regression with this penalty is **Ridge regression**, and it keeps a closed form, the
normal equation with $\alpha$ added to the diagonal:

$$\hat w = (X^\top X + \alpha I)^{-1} X^\top y .$$

What that buys:

- **Always solvable.** $X^\top X$ is singular when features are collinear or outnumber the
  rows; $X^\top X + \alpha I$ never is for $\alpha > 0$.
- **Stable with correlated features.** Two near-identical features get roughly equal
  shares of the weight, instead of one huge positive and one huge negative coefficient that
  flip between samples.
- **Smooth shrinkage.** Every coefficient moves towards 0 as $\alpha$ grows, but none
  becomes exactly 0. To drop features entirely you need L1 (Lasso).

The penalty is on the coefficients' sizes, which depend on the features' units, so
standardise features first. The intercept is not penalised.
