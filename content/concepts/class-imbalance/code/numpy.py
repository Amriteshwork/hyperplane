"""Class imbalance in NumPy: when one class is rare, accuracy flatters and the model ignores it."""

import numpy as np

# @a:reuse
from classification_metrics_numpy import precision_recall_f1
from decision_boundary_numpy import fit_logistic
from sigmoid_numpy import sigmoid


# @a:balanced
def balanced_weights(y):
    """scikit-learn's class_weight='balanced': n / (n_classes × size of the row's class)."""
    classes, counts = np.unique(y, return_counts=True)
    per_class = len(y) / (len(classes) * counts)
    return per_class[np.searchsorted(classes, y)]


# @a:demo
if __name__ == "__main__":
    # @a:data
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.uniform(size=n) < 0.05).astype(float)  # 5% positives
    X = rng.normal(size=(n, 2)) + 1.2 * y[:, None]  # shifted, but overlapping heavily

    # @a:baseline
    never = np.zeros(n)  # always predict the majority class
    print(f"always 0: accuracy {np.mean(never == y):.3f}, recall {precision_recall_f1(y, never)[1]}")
    assert np.mean(never == y) > 0.93 and precision_recall_f1(y, never)[1] == 0

    # @a:compare
    w, b = fit_logistic(X, y)
    s = balanced_weights(y)
    assert np.isclose(np.sum(s * y), np.sum(s * (1 - y)))  # each class now weighs the same
    w_bal, b_bal = fit_logistic(X, y, weights=s)
    for name, (ww, bb) in {"plain": (w, b), "balanced": (w_bal, b_bal)}.items():
        called = X @ ww + bb >= 0
        pr, rc, f1 = precision_recall_f1(y, called)
        print(f"{name:>8}: accuracy {np.mean(called == y):.3f}, precision {pr:.3f}, "
              f"recall {rc:.3f}, F1 {f1:.3f}")
    recall = [precision_recall_f1(y, X @ ww + bb >= 0)[1] for ww, bb in ((w, b), (w_bal, b_bal))]
    assert recall[1] > recall[0] + 0.3  # balanced weights find far more of the rare class

    # @a:threshold
    shift = b_bal - b  # weighting mostly moves the intercept...
    print(f"intercept shift {shift:.2f}, log of the weight ratio {np.log(s.max() / s.min()):.2f}")
    agree = np.mean((sigmoid(X @ w + b) >= sigmoid(-shift)) == (X @ w_bal + b_bal >= 0))
    assert agree > 0.97  # ...so a lower threshold on the plain model does almost the same
