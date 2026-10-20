---
title: Hyperparameter search
summary: Choose the settings a model can't learn (α, the L1 ratio…) by trying candidates and scoring each with cross-validation.
area: evaluation
prerequisites: [cross-validation, combined-penalties]
---

**Parameters** are learned by fitting (the weights). **Hyperparameters** are chosen before
fitting and change what the fit does: Ridge's $\alpha$, Elastic Net's $\alpha$ and L1 ratio,
a polynomial's degree. No formula gives the best ones; you try candidates and keep the one
that generalises best.

The recipe:

1. Pick a **search space**: which values to try for each hyperparameter. Strengths like
   $\alpha$ act on orders of magnitude, so search them on a **log scale** (0.01, 0.1, 1, 10).
2. Score every candidate by **cross-validation on the training rows only**.
3. Refit the winner on all the training rows, and evaluate it **once** on the test set.

Two common ways to pick candidates:

- **Grid search** tries every combination. With hyperparameters that interact, like $\alpha$
  and the L1 ratio, it searches them together rather than one at a time. But its cost
  multiplies: 6 values × 3 values × 5 folds is already 90 fits.
- **Random search** samples combinations from ranges (log-uniform for $\alpha$). For the
  same budget it tries more distinct values of each hyperparameter, which matters when only
  one or two of them really affect the score. It often gets close to the grid's best at a
  fraction of the cost.

The pitfall is **overfitting the validation data**: try enough candidates and one will
score well by luck. That's why the test set stays out of the search entirely, and why the
best CV score is slightly optimistic. For an honest estimate of the whole procedure, nest
the search inside an outer cross-validation.
