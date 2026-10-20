"""Class imbalance in PyTorch: when one class is rare, accuracy flatters and the model ignores it."""

import math

import torch

# @a:reuse
from classification_metrics_pytorch import precision_recall_f1
from decision_boundary_pytorch import fit_logistic
from sigmoid_pytorch import sigmoid


# @a:balanced
def balanced_weights(y):
    """scikit-learn's class_weight='balanced': n / (n_classes × size of the row's class)."""
    classes, inverse, counts = torch.unique(y, return_inverse=True, return_counts=True)
    per_class = len(y) / (len(classes) * counts.to(torch.float64))
    return per_class[inverse]


# @a:demo
if __name__ == "__main__":
    # @a:data
    torch.manual_seed(0)
    f64 = torch.float64
    n = 4000
    y = (torch.rand(n, dtype=f64) < 0.05).to(f64)  # 5% positives
    X = torch.randn(n, 2, dtype=f64) + 1.2 * y[:, None]  # shifted, but overlapping heavily

    # @a:baseline
    never = torch.zeros(n, dtype=f64)  # always predict the majority class
    accuracy = (never == y).double().mean().item()
    print(f"always 0: accuracy {accuracy:.3f}, recall {precision_recall_f1(y, never)[1]}")
    assert accuracy > 0.93 and precision_recall_f1(y, never)[1] == 0

    # @a:compare
    w, b = fit_logistic(X, y)
    s = balanced_weights(y)
    assert torch.isclose((s * y).sum(), (s * (1 - y)).sum())  # each class now weighs the same
    w_bal, b_bal = fit_logistic(X, y, weights=s)
    for name, (ww, bb) in {"plain": (w, b), "balanced": (w_bal, b_bal)}.items():
        called = X @ ww + bb >= 0
        pr, rc, f1 = precision_recall_f1(y, called)
        print(f"{name:>8}: accuracy {(called == y).double().mean().item():.3f}, "
              f"precision {pr:.3f}, recall {rc:.3f}, F1 {f1:.3f}")
    recall = [precision_recall_f1(y, X @ ww + bb >= 0)[1] for ww, bb in ((w, b), (w_bal, b_bal))]
    assert recall[1] > recall[0] + 0.3  # balanced weights find far more of the rare class

    # @a:threshold
    shift = (b_bal - b).item()  # weighting mostly moves the intercept...
    print(f"intercept shift {shift:.2f}, log of the weight ratio {math.log(s.max() / s.min()):.2f}")
    plain_p = sigmoid(X @ w + b)
    agree = ((plain_p >= 1 / (1 + math.exp(shift))) == (X @ w_bal + b_bal >= 0)).double().mean()
    assert agree > 0.97  # ...so a lower threshold on the plain model does almost the same
