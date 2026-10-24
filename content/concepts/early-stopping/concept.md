---
title: Early stopping
summary: Hold out some training rows, stop training when the loss on them stops improving, and keep the weights from the best epoch.
area: evaluation
prerequisites: [adam-optimizer, epochs, overfitting-underfitting]
---

A flexible model trained for long enough fits its training rows ever more closely, noise
included. The **training loss** keeps falling; the loss on rows it never trains on, the
**validation loss**, falls at first, then flattens and climbs. The bottom of that curve is
the best model training will produce, and every epoch after it is overfitting.

Watch it happen: this network trains in your browser, on 30 noisy rows.

::figure[early-stopping]

**Early stopping** watches for that bottom:

1. Split the training rows into a training part and a **validation** part (e.g. 90/10).
   The test set stays untouched.
2. After every epoch, compute the validation loss (or score).
3. If it improved by more than `tol` on the best so far, save a copy of the weights.
4. Once it has failed to improve for `patience` epochs in a row, stop, and **restore the
   saved weights**.

`patience` matters because the validation loss is noisy: one bad epoch is no reason to stop.

Early stopping is a **regulariser**. Weights start small and grow during training, so
stopping early keeps them small, much like an L2 penalty does. For a linear model on
squared error, $t$ steps of gradient descent with step size $\eta$ act roughly like Ridge
with $\alpha \approx 1/(\eta t)$. It costs no extra training (you were going to watch the
loss anyway), and it is the most common regulariser for neural networks.

Its costs: the validation rows are not trained on, which hurts with small data (and a
validation set of a dozen rows is too noisy to judge by; scikit-learn's classifiers stop on
validation accuracy, which saturates fast on small sets). A common fix is to note the best
epoch count, then retrain on all the training rows for that many epochs.
