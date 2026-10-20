---
title: Sparsity
summary: A sparse model uses only a few of its weights; the rest are exactly zero. L1 penalties produce it.
area: features
prerequisites: [l1-regularization]
---

A weight vector is **sparse** when most of its entries are exactly zero. A sparse linear
model ignores those features entirely: it's shorter to write down, cheaper to compute, and
it names the features that matter.

The L1 penalty produces sparsity, and $\alpha$ controls how much:

- At $\alpha \ge \alpha_{\max} = \frac{1}{n}\max_j |x_j^\top (y - \bar y)|$, every weight is
  zero: no feature explains enough of the target to pay the penalty.
- As $\alpha$ decreases, features **enter** the model one at a time, roughly in order of
  how much they help. Plotted against $\log\alpha$, the weights trace the **Lasso path**.
- At $\alpha \to 0$ you're back to least squares, where every weight is (almost surely)
  non-zero.

Why corners make zeros: Lasso is least squares under the constraint $\sum_j |w_j| \le t$,
a diamond in two dimensions with its corners on the axes. The loss's elliptical contours
usually touch the diamond first at a corner, where one weight is 0. Ridge's constraint is a
circle, with no corners, so its solutions are almost never exactly 0.

Sparsity is a bet that the truth is simple. When only a few features matter, it removes
noise and helps; when most features contribute a little, forcing zeros costs accuracy.
