---
title: Feature expansion
summary: Add new columns built from the old ones (powers, products) so a linear model can fit curves.
area: features
prerequisites: [normal-equation]
---

A linear model can only add up its features with fixed weights, so on its own it draws
straight lines. But nothing stops us from giving it **new features** computed from the old
ones: $x^2$, $x^3$, or the product $x_1 x_2$. The model is still linear *in its
coefficients*, so it is fitted with exactly the same least squares, yet as a function of the
original $x$ it can now curve.

$$[x_1, x_2] \;\longrightarrow\; [x_1,\ x_2,\ x_1^2,\ x_1 x_2,\ x_2^2]$$

- **Powers** ($x_1^2$) let the effect of one feature bend: rise then fall, or accelerate.
- **Interactions** ($x_1 x_2$) let one feature change the effect of another.

The price is size: $p$ features at degree $d$ give $\binom{p+d}{d} - 1$ columns, and more
columns mean more room to overfit. Scale features first, too: the square of a feature in the
thousands lives in the millions, which makes the least-squares solve less accurate.
