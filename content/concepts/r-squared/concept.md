---
title: R² (coefficient of determination)
summary: The share of the target's variance a model explains, compared with predicting the mean.
prerequisites: [mean-squared-error]
---

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

The denominator is the error of the laziest model, always predicting the average $\bar{y}$.
R² says how much of that error the model removes: 1 means perfect predictions, 0 means no
better than the mean, and on a test set it can even be **negative**, which means worse than the
mean.

R² has no units, which makes it easy to compare across problems, but it hides the size of
the errors. Report it together with RMSE or MAE. A high R² on training data and a much
lower one on test data is a classic sign of overfitting.
