---
title: Residuals
summary: The error left over for each row, true value minus prediction; the best diagnostic you have.
area: evaluation
prerequisites: [numpy-basics]
---

A **residual** is $e_i = y_i - \hat{y}_i$. Plotting residuals against the predictions is the
quickest health check for a regression model:

- **A shapeless band around zero**: the model has captured the pattern; what's left is noise.
- **A curve**: the relationship isn't linear; add transformed or polynomial features, or use a
  non-linear model.
- **A funnel** (spread grows with the prediction): the error variance isn't constant; try
  transforming the target, for example with a log.
- **A few points far away**: outliers that may be pulling the fit.

Residuals on the test set tell you about generalisation; residuals on the training set tell
you about fit.
