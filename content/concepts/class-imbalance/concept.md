---
title: Class imbalance
summary: When one class is rare, accuracy flatters and models ignore it. Weight the classes, move the threshold, and measure with the right metric.
area: evaluation
prerequisites: [classification-metrics, decision-boundary]
---

Many real problems have a rare class: fraud, disease, defects, churn. With 5% positives, a
model that always says "no" is 95% accurate and completely useless. This is the
**accuracy paradox**.

The training loss has the same blind spot. Log-loss averages over rows, so 95% of it comes
from negatives, and the cheapest way to reduce it is to lean towards "no". The model learns
to rarely predict the rare class.

Remedies, roughly in order of simplicity:

1. **Measure the right thing.** Use recall, precision, F1, ROC-AUC or a precision–recall
   curve, never accuracy alone.
2. **Move the threshold.** Keep the model, but call a row positive at, say, $p \ge 0.2$
   instead of 0.5. Recall rises, precision falls; choose by the cost of each mistake.
3. **Weight the classes.** Multiply each row's loss by a weight that is larger for rare
   classes. scikit-learn's `class_weight="balanced"` uses
   $w_k = \frac{n}{K \cdot n_k}$, so every class contributes equally. For logistic
   regression this mostly shifts the intercept by $\log$ of the weight ratio, which is
   much like lowering the threshold, but it also inflates the predicted probabilities of
   the rare class.
4. **Resample.** Oversample the rare class or undersample the common one. This has the same
   effect as weighting, at the cost of duplicated or discarded rows.

Also split carefully: with few positives, a random test set may contain hardly any.
Stratified splits keep the class proportions equal in train and test.
