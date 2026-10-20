---
title: Feature selection
summary: Keep only the features that help predict the target. Simpler models, less noise, and cheaper data.
area: features
prerequisites: [sparsity, cross-validation]
---

**Feature selection** chooses a subset of the features to model with. Fewer features mean a
model that is easier to explain, cheaper to run and to feed with data, and less able to fit
noise with features that only correlate with the target by chance.

There are three families of methods:

- **Filter**: score each feature on its own (correlation with the target, mutual
  information…) and keep the best. Fast, but blind to features that only help together, or
  that duplicate each other.
- **Wrapper**: search over subsets, training a model on each (forward selection, recursive
  feature elimination). Accurate, but expensive, and with many subsets tried it can overfit
  the validation data.
- **Embedded**: the model selects while it trains. The **Lasso** is the classic example:
  its L1 penalty sets the weights of unhelpful features exactly to zero, and $\alpha$
  decides how many survive.

Two rules make the selection honest:

1. **Select on training data only.** Choosing features using the test set, or before the
   train/test split, leaks information and makes the test score optimistic. Choose α (and
   so the features) by **cross-validation** on the training rows.
2. **Selection is not causation, and it is not unique.** From a group of correlated
   features, any one may be kept. A dropped feature may be just as related to the target as
   a kept one; it was merely redundant given the others.

A common refinement, the *relaxed lasso*, uses the Lasso only to pick the features, then
refits plain least squares on them, removing the penalty's shrinkage from the kept weights.
