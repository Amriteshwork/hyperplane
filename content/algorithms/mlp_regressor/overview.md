---
title: MLP Regressor
summary: A neural network for numbers. Layers of neurons learn their own non-linear features, and backpropagation trains all the weights together.
difficulty: 4
---

## Intuition

Linear Regression predicts with one weighted sum of the features. An MLP (multi-layer
perceptron) puts a **hidden layer** of neurons in between: each neuron takes its own
weighted sum of the features and **bends** it (ReLU: negative sums become 0). The output is
a weighted sum of those bent pieces.

Each hidden ReLU neuron adds one bent line, and a hundred of them added together can follow
almost any curve, including interactions between features, without you designing
polynomial or interaction features yourself. The price: thousands of weights to learn,
which needs data, tuning and a loss surface with no single guaranteed minimum.

::figure[relu-sum]

The network is trained the way SGD trains a linear model, in mini-batches over several
epochs. The difference is the gradient: **backpropagation** computes it for every weight in
every layer with the chain rule.

::figure[network-shape]

## How it works

1. Preprocess and **standardise** the features. The target is standardised as well, and
   predictions are converted back to its units.
2. Start every weight at a small random value (Glorot initialisation).
3. Each epoch, shuffle the training rows and split them into mini-batches of 200. For each
   batch:
   - **forward pass**: compute each layer's outputs, up to the prediction;
   - **loss**: half the mean squared error, plus an L2 penalty on the weights;
   - **backward pass**: backpropagate the error to get every weight's gradient;
   - **step**: update the weights with Adam.
4. Stop once the training loss hasn't improved by `tol` for more than 10 epochs (or, with
   early stopping, the R² on a held-out tenth of the training rows).
5. Predict by running the forward pass.

## Math

With one hidden layer of $h$ neurons, activation $f$ and inputs $x \in \mathbb{R}^p$:

$$\hat y = w_2^\top f(W_1^\top x + b_1) + b_2, \qquad W_1 \in \mathbb{R}^{p \times h},\ w_2 \in \mathbb{R}^h .$$

More hidden layers repeat the inner step. Training minimises, over all weights,

$$J = \frac{1}{2n}\sum_{i=1}^n (\hat y_i - y_i)^2 + \frac{\alpha}{2n}\sum_\ell \lVert W_\ell\rVert_F^2$$

(per mini-batch, with $n$ its size; biases are not penalised). Backpropagation gives the
gradient layer by layer, from $\delta_L = \hat y - y$ at the output:

$$\nabla W_\ell = \tfrac1n\big(A_{\ell-1}^\top\delta_\ell + \alpha W_\ell\big), \qquad
\delta_{\ell-1} = (\delta_\ell W_\ell^\top) \odot f'(Z_{\ell-1}).$$

Adam then steps each weight by $\eta\,\hat m/(\sqrt{\hat v} + \epsilon)$, using running
averages of its gradient and squared gradient.

$J$ is **not convex**: different starting weights (`random_state`) can end in different
minima, usually of similar quality. With `activation="identity"` the network can only
represent a linear model, however many layers it has.

## Assumptions

- **Enough rows** for the number of weights. One hidden layer of 100 neurons on 10
  features has 1,201; a few hundred rows can be memorised.
- **Standardised features** (done in preprocessing), so the first weighted sums are small
  and every weight gets useful gradients.
- **Independent rows.**
- No assumption about the shape of the relationship: that is the point of the model.

## Hyperparameters

- `hidden_layers` (default 1) and `hidden_units` (default 100): the network's shape,
  scikit-learn's `hidden_layer_sizes` of `[100]`.
- `activation` (default `relu`): the hidden neurons' bend; `tanh` and `logistic` are
  smooth S-curves, `identity` makes the model linear.
- `alpha` (default 0.0001): the L2 penalty. The first thing to raise when training scores
  beat test scores by a lot.
- `learning_rate_init` (default 0.001): the optimiser's step size.
- `solver` (default `adam`): `sgd` (with momentum) or `lbfgs`, a full-batch method that
  often does best on small datasets.
- `batch_size` (default: min(200, rows)), `max_iter` (default 1000; scikit-learn's is
  200), `tol` (default 0.001; scikit-learn's is 0.0001), `n_iter_no_change` (default 10):
  mini-batching and when to stop.
- `early_stopping` (default off) and `validation_fraction` (default 0.1): stop on a
  held-out R² instead of the training loss, and keep the best epoch's weights.
- `random_state` (default 0): the starting weights and the shuffle order.

## When to use / not use

**Use it** when the target depends on the features in curved or interacting ways and you
have thousands of rows or more: it learns the features a linear model would need you to
build. It is also the gateway to deep learning: the same forward pass, backpropagation and
Adam train every modern network.

**Avoid it** on small datasets, where Ridge or a tree does as well with far less tuning;
when you need to explain each feature's effect (there are no coefficients to read); or when
results must not depend on a random seed. On typical tabular data, gradient-boosted trees
are usually at least as accurate and easier to tune.

## Common pitfalls

- **Unscaled features or target.** Training is slow or unstable; neurons saturate. Both are
  standardised here; don't skip it in your own code.
- **Judging by the training score.** A network can fit its training rows almost perfectly.
  Compare with the test score, and with Linear Regression.
- **Ignoring the "did not converge" warning.** Raise `max_iter`, or the learning rate.
- **A learning rate that's too large.** The loss curve zig-zags or explodes; lower it.
- **Too big a network for the data.** More neurons isn't free: raise `alpha`, shrink the
  layers or use early stopping on larger datasets.
- **Early stopping on tiny data.** A validation set of a few dozen rows is too noisy to stop
  on; it can halt after a handful of epochs.
- **Reading meaning into individual weights.** Hidden neurons mix all the features; no
  single weight is "the effect of" a feature.

## Interview questions

1. Why does an MLP need non-linear activations? What is a network with identity
   activations equivalent to?
2. Walk through one training step: forward pass, loss, backpropagation, update.
3. Why initialise the weights randomly rather than at zero?
4. What are vanishing gradients, and why does ReLU help?
5. How does Adam differ from plain SGD? What does bias correction fix?
6. How do `alpha` and early stopping each fight overfitting?
7. Why standardise the target for an MLP regressor, and what changes if you don't?
8. Why can two training runs with different seeds give different networks? Is that a
   problem?
9. When would you pick gradient-boosted trees over an MLP on tabular data?
