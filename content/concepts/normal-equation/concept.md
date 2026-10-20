---
title: Normal equation
summary: The closed-form solution that gives the least-squares coefficients in one step.
area: optimisation
prerequisites: [mean-squared-error, matrix-multiplication, derivatives]
---

For a linear model $\hat{y} = X\beta$, the squared error $\lVert y - X\beta\rVert^2$ is a bowl
with a single lowest point. Setting its gradient to zero gives the **normal equation**:

$$X^\top X\,\beta = X^\top y \quad\Rightarrow\quad \beta = (X^\top X)^{-1} X^\top y$$

So linear regression needs no iterations or learning rate: the answer is computed directly.

Two practical caveats. The inverse doesn't exist when features are perfectly collinear, and
it is numerically fragile when they are nearly so; libraries therefore solve the same system
with QR or SVD decompositions. And solving it costs roughly $O(np^2 + p^3)$, so with very many
features or rows, **gradient descent** becomes the better tool.
