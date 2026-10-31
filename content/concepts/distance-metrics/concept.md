---
title: Distance metrics
summary: A distance turns "how similar are these two rows?" into a number, and the choice of formula and feature scales decides which rows count as close.
area: instance-based
prerequisites: [vectors, feature-scaling]
---

Many models need to know which examples are alike. Each example is a vector of features, so
"alike" becomes "close", and a **distance metric** defines close.

The common ones belong to one family, the **Minkowski distance**:

$$d_p(u, v) = \Big(\sum_{i} |u_i - v_i|^p\Big)^{1/p}$$

- $p = 2$, **Euclidean**: the straight-line distance, Pythagoras in any number of dimensions.
- $p = 1$, **Manhattan**: the sum of the per-feature differences, like walking a street grid.
- $p \to \infty$, **Chebyshev**: only the single largest difference counts.

From the origin to $(3, 4)$, these give 5, 7 and 4. Larger $p$ pays more attention to the
biggest difference. So metrics can disagree about which point is nearest: from the origin,
Manhattan prefers $(3, 0)$ (distance 3 over 4) but Euclidean prefers $(2, 2)$ (2.83 over 3).

Each metric has its own idea of a circle, the set of points at distance 1. Move the query and
watch the nearest point change:

::figure[distance-balls]

**Cosine distance** is different. It compares directions and ignores length:

$$d_{\cos}(u, v) = 1 - \frac{u \cdot v}{\lVert u \rVert\, \lVert v \rVert}$$

It is 0 for vectors pointing the same way, however long, 1 at right angles and 2 when opposite.
That suits text and embeddings, where a longer document has larger counts without being about
something else. Strictly it isn't a metric, as it breaks the triangle inequality, but it is
used like one.

## Scale decides everything

Distances add up differences across features, so units matter. Take age in years and income in
dollars: a 40-year age gap adds 40, while a modest income gap adds thousands, so income alone
picks the neighbours. In this page's NumPy code, the 5 nearest rows differ in age by 13 years
on average without scaling, and by 3 with it.

**Standardise your features** before using any of these metrics, fitting the scaler on the
training rows only. Standardising also makes a claim: that one standard deviation of every
feature matters equally. If you know a feature matters more, you can scale it up on purpose.
