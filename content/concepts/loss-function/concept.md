---
title: Loss function
summary: A single number that says how wrong a model's predictions are; training minimises it.
area: optimisation
prerequisites: [numpy-basics]
---

A **loss function** compares predictions with the true values and returns one number:
small when the model is right, large when it is wrong. Training a model means searching for
the parameters that make the loss as small as possible on the training data.

The choice of loss shapes what "good" means. Squared error punishes big misses heavily, so
models avoid large errors at the cost of many small ones. Absolute error treats every unit of
error equally and is more forgiving of outliers. Classification uses losses such as log-loss.

A loss is what the model optimises; a **metric** is what you report. They are often related
(MSE and RMSE) but needn't be the same.
