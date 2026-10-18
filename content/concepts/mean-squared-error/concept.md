---
title: Mean squared error (MSE)
summary: The average of the squared differences between predictions and true values.
prerequisites: [loss-function]
---

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Squaring makes every error positive and makes large errors count much more than small
ones: one miss of 10 costs as much as a hundred misses of 1. It is also smooth, which makes
it easy to minimise, and minimising it is exactly what ordinary least squares does.

MSE is in squared units (e.g. "dollars²"), so for reporting we usually take its square root,
the **RMSE**, which is back in the target's units. For an error measure that is less
sensitive to outliers, use the mean absolute error (MAE).
