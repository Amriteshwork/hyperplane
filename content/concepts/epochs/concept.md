---
title: Epochs
summary: One epoch is one pass over every training row. Training runs for several, and stops when the loss stops improving.
area: optimisation
prerequisites: [stochastic-gradient-descent, train-test-split]
---

SGD visits the rows one at a time (or in mini-batches). An **epoch** is one full pass over
the training set, every row used once, in a fresh random order each time. With $n$ rows and
batch size $B$, one epoch is $\lceil n/B \rceil$ updates.

One pass is rarely enough to settle: the learning rate is still large, and each row has only
pulled once. So training runs for several epochs, watching the **loss per epoch**:

- the training loss falls quickly at first, then flattens out;
- a **validation** loss (on rows held out from training) shows when extra epochs stop
  helping, or start to **overfit**.

When to stop is itself a choice. scikit-learn's SGD stops once the average loss hasn't
improved by at least `tol` for `n_iter_no_change` epochs in a row (with `early_stopping`,
measured on a validation split), and `max_iter` caps the epochs. Neural networks use the
same idea, keeping the weights from their best validation epoch.

Reshuffling every epoch matters: a fixed order (worse, a sorted one) makes the weights
drift with whatever the rows near the end of each pass have in common.
