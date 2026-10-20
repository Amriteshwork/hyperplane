---
title: ROC curve and AUC
summary: Plot true against false positive rate at every threshold. The area under it is the chance a random positive outscores a random negative.
area: evaluation
prerequisites: [classification-metrics]
---

A classifier that outputs scores (such as probabilities) defines a different confusion
matrix at every threshold. The **ROC curve** shows them all at once: for each threshold, plot

- the **true positive rate** $\text{TPR} = \frac{TP}{TP + FN}$ (recall), against
- the **false positive rate** $\text{FPR} = \frac{FP}{FP + TN}$.

The curve runs from (0, 0), where nothing is called positive, to (1, 1), where everything
is. A perfect model goes straight up to (0, 1); random guessing follows the diagonal.

The **area under the curve (AUC)** summarises it in one number with a neat meaning:

$$\text{AUC} = P(\text{score of a random positive} > \text{score of a random negative}),$$

counting ties as half. So 0.5 is guessing and 1 is perfect separation.

What AUC does and doesn't tell you:

- It measures **ranking only**. Any increasing transformation of the scores gives the same
  AUC, so it says nothing about whether the probabilities are calibrated (log-loss does).
- It is **threshold-free**, which is useful for comparing models, but you still have to
  pick a threshold to make decisions.
- On **heavily imbalanced** data a high AUC can coexist with poor precision; a
  precision–recall curve is more informative there.

With several classes, compute one curve per class against the rest and average the AUCs.
