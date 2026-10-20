---
title: Overfitting and underfitting
summary: Too simple misses the pattern; too flexible memorises the noise. Only held-out data tells them apart.
area: evaluation
prerequisites: [train-test-split, mean-squared-error]
---

A model **underfits** when it is too simple for the pattern: a straight line through a
curve. Its error is high on the training data *and* on new data.

A model **overfits** when it is flexible enough to follow the random noise in its training
rows as if it were signal. Training error looks excellent, but the wiggles it learned don't
repeat in new data, so test error is poor.

| | Training error | Test error |
|---|---|---|
| Underfitting | high | high |
| Good fit | low | low (close to training) |
| Overfitting | very low | much higher than training |

Training error alone can never reveal overfitting: a more flexible model always fits its own
training data at least as well. That is why every score in this course is reported on a
held-out test set.

Fixes for overfitting: a simpler model (lower polynomial degree, shallower tree), more data,
or regularisation (Ridge, Lasso). Fixes for underfitting: a more flexible model or better
features.
