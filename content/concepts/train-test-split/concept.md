---
title: Train/test split
summary: Hold out data the model never sees during training, to measure how it does on new data.
---

A model can memorise its training data, so its score on that data is optimistic. Splitting
the rows into a **training set** (to fit) and a **test set** (to evaluate) gives an honest
estimate of performance on unseen data. A common choice is 80/20.

Rules that keep the estimate honest:

- Fit **everything** that learns from data (scalers, imputers, encoders, the model) on the
  training set only. Otherwise information leaks from the test set.
- Fix the random seed so the split, and the result, are reproducible.
- For classification, a **stratified** split keeps class proportions equal in both parts.
- For time series, split by time instead of at random.

When data is scarce, cross-validation repeats the split several times and averages the scores.
