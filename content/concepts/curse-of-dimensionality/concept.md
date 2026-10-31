---
title: Curse of dimensionality
summary: With many features, data spreads so thin that every point is far from every other, and the nearest neighbour is barely nearer than the farthest.
area: instance-based
prerequisites: [distance-metrics, lazy-learning, probability]
---

Intuition from two or three dimensions fails badly in fifty. "Find similar rows" assumes some
rows are close by, and as features are added, that stops being true.

## Neighbours stop being near

Spread points evenly in a unit cube and ask for a sub-cube holding 1% of them. Its volume must
be 0.01, so its side is $0.01^{1/d}$:

| Features $d$ | 1 | 2 | 10 | 100 |
|---|---|---|---|---|
| Side holding 1% of the data | 0.01 | 0.10 | 0.63 | 0.95 |

With 100 features, your "local" neighbourhood spans 95% of the range of every feature. It isn't
local any more. To keep a neighbour within 0.1 along each axis you need about $10^d$ rows: 100
for two features, a million for six.

## All distances look the same

Each new feature adds another independent term to the squared distance. Sums of many
independent terms concentrate around their mean, so the distances grow but their relative
spread shrinks. On 1,000 uniform rows, this page's code measures:

| Features $d$ | 2 | 10 | 100 | 1000 |
|---|---|---|---|---|
| Farthest ÷ nearest | 85.5 | 4.0 | 1.44 | 1.12 |

When the farthest row is only 12% farther than the nearest, "nearest" has stopped meaning much.

Slide the dimension up and watch the distances pile together:

::figure[distance-concentration]

## Irrelevant features do the damage

Real data usually has structure: its features are correlated, so its effective dimension can
be far below its column count. That is why k-NN works at all on images or embeddings.

Noise is the true enemy. Every irrelevant feature adds its own random term to every distance,
and those terms drown out the ones that matter. Here two informative features are joined by
noise columns, with 300 rows and $k = 15$:

| Noise features | 0 | 10 | 100 | 1000 |
|---|---|---|---|---|
| Test accuracy | 94% | 89% | 81% | 68% |

The two good features are still there. The distances just can't hear them.

## What to do

- **Remove features** that don't help (see [feature selection](/concepts/feature-selection)).
- **Reduce dimension** with PCA or a learned embedding, then measure distances there.
- **Get more rows**, though the number needed grows exponentially with the true dimension.
- **Choose a model that weighs features.** Lasso, trees and boosting can ignore a useless column.
  Plain k-NN treats every column alike.

The curse hits more than k-NN: density estimates, grid search and anything else that relies on
covering the space all suffer from it.
