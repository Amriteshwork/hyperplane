---
title: Cross-validation
summary: Rotate which part of the training data is held out, and average the scores, to choose settings without touching the test set.
area: evaluation
prerequisites: [train-test-split, mean-squared-error]
---

Choosing a hyperparameter (Ridge's $\alpha$, a polynomial's degree) by its test score
quietly turns the test set into training data: try enough settings and one will look good
by luck. The test set must be used **once**, at the end.

**k-fold cross-validation** finds the setting using the training data alone:

1. Shuffle the training rows and cut them into $k$ equal **folds** (5 or 10 is typical).
2. For each fold: fit on the other $k-1$ folds, score on this one.
3. Average the $k$ scores. Every row is used for validation exactly once.
4. Repeat for every candidate setting; keep the one with the best average.
5. Refit that setting on all the training data, and report its score on the test set.

Averaging over $k$ splits makes the estimate much less dependent on one lucky or unlucky
split, which matters most on small datasets. The cost is $k$ fits per setting.

Variants: **stratified** k-fold keeps class proportions equal in every fold
(classification); **time-series** splits always validate on data later than the training
rows; **leave-one-out** is $k = n$.
