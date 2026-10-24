---
title: Neurons and layers
summary: A neuron is a weighted sum passed through a bend. Layers of them, each feeding the next, make a neural network.
area: neural-networks
prerequisites: [matrix-multiplication, sigmoid, decision-boundary]
---

An artificial **neuron** takes inputs $x_1, \dots, x_p$, multiplies each by a **weight**,
adds a **bias** and passes the result through an **activation function** $f$:

$$a = f(w^\top x + b) = f\Big(\sum_{j} w_j x_j + b\Big).$$

With the sigmoid as $f$ this is exactly logistic regression: one neuron draws one straight
decision boundary, $w^\top x + b = 0$.

A **layer** is many neurons reading the same inputs, each with its own weights. Stack their
weight vectors as the columns of a matrix $W$ and a whole batch of rows $X$ goes through the
layer in one matrix product:

$$A = f(XW + b) \qquad (n \times p)\,(p \times h) \to (n \times h).$$

A **multi-layer perceptron (MLP)** chains layers: the outputs of one are the inputs of the
next. The middle layers are **hidden** (we never see their targets); the last layer gives
the prediction.

$$\hat y = f_{\text{out}}\big(f(f(XW_1 + b_1)W_2 + b_2)\,W_3 + b_3\big)$$

Why bother? Each hidden neuron learns its own feature, a soft "is the row on this side of my
line?", and the next layer combines them. A single neuron can't separate XOR (the points
$(0,0), (1,1)$ against $(0,1), (1,0)$): no straight line does. Two hidden neurons (OR and
AND) and one output (OR but not AND) can. With enough hidden neurons, one hidden layer can
approximate any continuous function (the **universal approximation theorem**), though that
says nothing about how many neurons it takes or whether training will find them.

Try it: here is that XOR network, with its real weights.

::figure[xor-network]

Counting parameters: a layer from $p$ inputs to $h$ neurons has $p \cdot h$ weights and
$h$ biases. They add up quickly, which is why networks need plenty of data.

::figure[network-shape]
