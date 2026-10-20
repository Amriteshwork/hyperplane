---
title: Decision boundary
summary: The surface where a classifier switches its answer. For logistic regression it is flat, where w·x + b = 0.
area: features
prerequisites: [log-loss, gradient-descent, feature-expansion]
---

A classifier divides feature space into regions, one per predicted class. The surface
between them is the **decision boundary**.

Logistic regression predicts class 1 when $p = \sigma(w^\top x + b) \ge 0.5$, which happens
exactly when the score is non-negative. Its boundary is therefore

$$w^\top x + b = 0 ,$$

a straight line with two features, a plane with three, and a **hyperplane** in general.
In two dimensions you can draw it: $x_2 = -(w_1 x_1 + b)/w_2$.

The weights also set how *sharp* the boundary is. Moving a distance $d$ away from it (along
$w$) changes the log-odds by $d\lVert w\rVert$. Large weights mean the probability jumps
from near 0 to near 1 in a narrow band; the L2 penalty keeps them modest.

**Linear boundaries have limits.** If the classes form rings or an XOR pattern, no line
separates them. But "linear" means linear in the **features**: add $x_1^2$, $x_2^2$ or
$x_1x_2$ as features and the boundary, still flat in the expanded space, becomes curved in
the original one.

The threshold need not be 0.5. Moving it shifts the boundary parallel to itself, trading
one kind of mistake for the other.
