---
title: Backpropagation
summary: The chain rule applied layer by layer, from the loss back to the inputs. One backward pass gives the gradient of every weight.
area: neural-networks
prerequisites: [activation-functions, gradient-descent, mean-squared-error]
---

Gradient descent needs $\partial L / \partial w$ for every weight. A network is a chain of
functions, $L(f_3(f_2(f_1(x))))$, so each derivative is a product of local derivatives (the
**chain rule**). **Backpropagation** organises that product so nothing is computed twice.

**Forward pass.** For each layer $\ell = 1, \dots, L$, keep every output:

$$Z_\ell = A_{\ell-1}W_\ell + b_\ell, \qquad A_\ell = f(Z_\ell), \qquad A_0 = X.$$

**Backward pass.** Define $\delta_\ell = \partial L / \partial Z_\ell$, how much the loss
changes when layer $\ell$'s weighted sums change. At the output, for the MSE and a linear
last layer, $\delta_L = \tfrac{2}{n}(\hat y - y)$. (Sigmoid or softmax with log-loss give the
equally tidy $\tfrac1n(p - y)$.) Then, from the top down:

$$\frac{\partial L}{\partial W_\ell} = A_{\ell-1}^\top \delta_\ell, \qquad
\frac{\partial L}{\partial b_\ell} = \textstyle\sum_i \delta_{\ell, i}, \qquad
\delta_{\ell-1} = \big(\delta_\ell W_\ell^\top\big) \odot f'(Z_{\ell-1}).$$

Read the last equation right to left: the error at layer $\ell$ is sent back through the
same weights it came forward through, and scaled by the slope of the activation, entry by
entry ($\odot$). The weight gradient is "input times error", exactly as in linear
regression, where $\nabla_w = X^\top(\hat y - y)\cdot\tfrac{2}{n}$.

Here is the whole procedure on the smallest network there is, one neuron per layer, with
every number shown:

::figure[backprop-steps]

**Why it matters.** Perturbing each weight to measure its effect costs one forward pass
per weight: millions of passes for a big network. Backpropagation gets all of them for
about the cost of two passes (reverse-mode automatic differentiation). PyTorch's
`loss.backward()` does exactly this on whatever graph your code built.

**What can go wrong.** The product of slopes can shrink (**vanishing gradients**: saturated
sigmoids, many layers) or grow (**exploding gradients**: large weights). Standardised
inputs, ReLU activations, careful initialisation and moderate learning rates keep it in
range. A gradient check against finite differences catches bugs in a hand-written backward
pass.
