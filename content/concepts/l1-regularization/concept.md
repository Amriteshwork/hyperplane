---
title: L1 regularisation (Lasso)
summary: Penalise the sum of absolute weights. Small weights snap to exactly zero, so the model drops features.
area: features
prerequisites: [l2-regularization, derivatives]
---

L1 regularisation adds the sum of the weights' absolute values to the squared error:

$$\min_w \; \frac{1}{2n}\lVert y - Xw \rVert^2 + \alpha \lVert w \rVert_1, \qquad \lVert w \rVert_1 = \sum_j |w_j| .$$

Linear regression with this penalty is the **Lasso**. The change from L2 looks small, but it
changes the answer qualitatively:

- **Exact zeros.** L2's penalty $w_j^2$ is flat near 0, so it barely pushes on small weights
  and never quite reaches zero. L1's $|w_j|$ pushes with the same force $\alpha$ all the way
  down, so a weight that helps the fit by less than $\alpha$ lands **exactly** on 0.
- **No closed form.** $|w_j|$ has a corner at 0 where the derivative is undefined, so there
  is no normal-equation-style formula. Instead, solve for one weight at a time: with the
  others fixed, the best $w_j$ is a **soft threshold**,

  $$w_j = \frac{S\!\left(\tfrac{1}{n}x_j^\top r_{-j},\ \alpha\right)}{\tfrac{1}{n}\lVert x_j \rVert^2}, \qquad S(z, t) = \operatorname{sign}(z)\max(|z| - t, 0),$$

  where $r_{-j}$ is the residual without feature $j$'s contribution. Sweeping over the
  weights until they settle is **coordinate descent**.
- **A check for the answer.** At the solution, every kept weight's feature correlates with
  the residual by exactly $\pm\alpha$, and every dropped one by at most $\alpha$. (This is the
  zero-*sub*gradient condition: at the corner, any slope in $[-\alpha, \alpha]$ counts.)

As with L2, standardise features first and leave the intercept unpenalised.
