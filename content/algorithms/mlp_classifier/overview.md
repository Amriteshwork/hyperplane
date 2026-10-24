---
title: MLP Classifier
summary: A neural network for classes. Hidden layers of neurons let the decision boundary curve, and a sigmoid or softmax output gives probabilities.
difficulty: 4
---

## Intuition

Logistic Regression draws one straight boundary between the classes. An MLP
(multi-layer perceptron) first passes the features through a **hidden layer** of neurons.
Each neuron is a small logistic-regression-like unit with its own weights: it asks "which
side of *my* line is this row on?", with a soft answer. The output layer then combines
those answers, so the final boundary can bend, wrap around a class, or separate classes
that no straight line can (XOR).

Try it: train a small network here and watch its boundary bend.

::figure[boundary-playground]

The network learns the hidden neurons' lines and the way to combine them at the same time,
by **backpropagation**: the chain rule sends the error at the output back through every
layer, giving the gradient of every weight. Training then runs mini-batch steps with Adam,
epoch after epoch, as SGD does for a linear model.

::figure[network-shape]

## How it works

1. Preprocess and **standardise** the features.
2. Set up the output layer from the labels: two classes get **one** output neuron with a
   sigmoid, giving $P(\texttt{classes\_[1]})$; $K \ge 3$ classes get $K$ neurons and a
   **softmax**.
3. Start every weight at a small random value (Glorot initialisation).
4. Each epoch, shuffle the training rows and split them into mini-batches of 200. For each
   batch: forward pass, log-loss plus an L2 penalty, backpropagation, an Adam step.
5. Stop once the training loss hasn't improved by `tol` for more than 10 epochs (or, with
   early stopping, the accuracy on a held-out tenth of the training rows).
6. Predict the class with the highest probability.

## Math

With one hidden layer of $h$ neurons and activation $f$:

$$H = f(XW_1 + b_1), \qquad P = \text{softmax}(HW_2 + b_2)\ \text{ (or } \sigma(Hw_2 + b_2) \text{ for two classes)}.$$

Training minimises the log-loss with an L2 penalty on the weights (per mini-batch of size
$n$; $Y$ is the one-hot label matrix):

$$J = -\frac1n\sum_{i}\sum_k Y_{ik}\log P_{ik} + \frac{\alpha}{2n}\sum_\ell \lVert W_\ell\rVert_F^2 .$$

Both output layers pair with log-loss so neatly that the output error is just
$\delta_L = P - Y$, as in Logistic Regression. Backpropagation carries it down:

$$\nabla W_\ell = \tfrac1n\big(A_{\ell-1}^\top\delta_\ell + \alpha W_\ell\big), \qquad
\delta_{\ell-1} = (\delta_\ell W_\ell^\top) \odot f'(Z_{\ell-1}).$$

Adam steps each weight by $\eta\,\hat m/(\sqrt{\hat v} + \epsilon)$. The loss is **not
convex**, so the result depends on the starting weights (`random_state`). With no hidden
layer this would be exactly multinomial Logistic Regression; the hidden layer is what
lets the boundary curve.

## Assumptions

- **Enough rows** for the number of weights, in every class. A few hundred rows can be
  memorised by a 100-neuron network.
- **Standardised features** (done in preprocessing), so neurons start in the steep part of
  their activation and every weight gets useful gradients.
- **Independent rows.**
- No assumption about the shape of the boundary: that is the point of the model.

## Hyperparameters

- `hidden_layers` (default 1) and `hidden_units` (default 100): the network's shape,
  scikit-learn's `hidden_layer_sizes` of `[100]`.
- `activation` (default `relu`): the hidden neurons' bend; `tanh` and `logistic` are
  smooth S-curves, `identity` makes the boundary straight again.
- `alpha` (default 0.0001): the L2 penalty; larger values give smoother boundaries.
- `learning_rate_init` (default 0.001): the optimiser's step size.
- `solver` (default `adam`): `sgd` (with momentum) or `lbfgs`, a full-batch method that
  often does best on small datasets.
- `batch_size` (default: min(200, rows)), `max_iter` (default 1000; scikit-learn's is
  200), `tol` (default 0.001; scikit-learn's is 0.0001), `n_iter_no_change` (default 10):
  mini-batching and when to stop.
- `early_stopping` (default off) and `validation_fraction` (default 0.1): stop on held-out
  accuracy instead of the training loss, and keep the best epoch's weights.
- `random_state` (default 0): the starting weights and the shuffle order.

## When to use / not use

**Use it** when the classes can't be separated by a straight boundary and you have
thousands of rows or more; or as a stepping stone to deep learning, which trains with the
same forward pass, backpropagation and Adam.

**Avoid it** on small datasets, where Logistic Regression or a tree does as well with far
less tuning; when you need to explain how each feature moves the odds (there are no
coefficients to read); or when results must not depend on a random seed. On typical tabular
data, gradient-boosted trees are usually at least as accurate and easier to tune.

## Common pitfalls

- **Unscaled features.** Training is slow or unstable, and sigmoid/tanh neurons saturate.
- **Judging by training accuracy.** 100% on the training rows is easy for a network.
  Compare with test accuracy, and with Logistic Regression.
- **Ignoring the "did not converge" warning.** Raise `max_iter`, or the learning rate.
- **Early stopping on tiny data.** It stops on validation *accuracy*, which on a dozen rows
  hits 100% almost at once and then can't improve: training halts after a few epochs.
- **Trusting probabilities blindly.** An overfitted network is confidently wrong: log-loss
  shows it even when accuracy looks fine.
- **Class imbalance.** There is no `class_weight` option; judge by recall, F1 or ROC-AUC
  rather than accuracy, or resample the training data.

## Interview questions

1. Why can an MLP separate XOR when Logistic Regression can't?
2. Which output activation and loss go with binary, multiclass and multilabel problems?
3. Why does the output error simplify to $P - Y$ for both sigmoid and softmax with
   log-loss?
4. Walk through backpropagation for a one-hidden-layer network.
5. What are vanishing gradients, and why does ReLU help?
6. How does Adam differ from plain SGD?
7. How do `alpha` and early stopping each fight overfitting?
8. Why might two runs with different seeds give different accuracies, and what would you
   report?
9. When would you pick gradient-boosted trees over an MLP on tabular data?
