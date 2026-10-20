---
title: Derivatives and gradients
summary: The derivative is a function's slope; the gradient collects the slopes in every direction and points uphill.
area: foundations
prerequisites: [numpy-basics]
---

The **derivative** of $f$ at $x$ is its slope there: how fast $f$ changes per unit change
in $x$.

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

A few rules cover most of ML: $(x^n)' = n x^{n-1}$; $(e^x)' = e^x$; and the **chain rule**
$\big(f(g(x))\big)' = f'(g(x))\,g'(x)$, which is what backpropagation applies layer by layer.

A model's loss depends on many parameters at once. A **partial derivative**
$\partial f / \partial w_i$ is the slope along one parameter with the others held fixed, and
the **gradient** stacks them:

$$\nabla f(w) = \left(\frac{\partial f}{\partial w_1}, \dots, \frac{\partial f}{\partial w_n}\right)$$

The gradient points in the direction of steepest increase, so stepping *against* it lowers
the function fastest: that's gradient descent. Where the gradient is zero, $f$ is flat, which
is how the normal equation finds the best weights in one step.

You can always check a hand-derived gradient numerically: nudge each parameter by a tiny $h$
and measure the change.
