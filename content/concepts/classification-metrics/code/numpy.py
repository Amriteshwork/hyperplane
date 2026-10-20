"""Classification metrics in NumPy: the confusion matrix, and every score computed from it."""

import numpy as np


# @a:confusion
def confusion(y, y_hat):
    """(tp, fp, fn, tn) for 0/1 (or False/True) labels, where 1 is the positive class."""
    y, y_hat = np.asarray(y, dtype=bool), np.asarray(y_hat, dtype=bool)
    return (
        int(np.sum(y & y_hat)),  # true positives: caught
        int(np.sum(~y & y_hat)),  # false positives: false alarms
        int(np.sum(y & ~y_hat)),  # false negatives: missed
        int(np.sum(~y & ~y_hat)),  # true negatives: correctly left alone
    )


# @a:scores
def precision_recall_f1(y, y_hat):
    """Precision tp/(tp+fp), recall tp/(tp+fn) and their harmonic mean; 0 when undefined."""
    tp, fp, fn, _ = confusion(y, y_hat)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


# @a:demo
if __name__ == "__main__":
    # @a:data
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=1000) < 0.3).astype(int)  # 30% positives
    p = np.clip(0.3 + 0.35 * (2 * y - 1) + rng.normal(0, 0.2, 1000), 0, 1)  # a decent model

    # @a:matrix
    y_hat = (p >= 0.5).astype(int)
    tp, fp, fn, tn = confusion(y, y_hat)
    print("          predicted 1  predicted 0")
    print(f"actual 1  {tp:>11}  {fn:>11}")
    print(f"actual 0  {fp:>11}  {tn:>11}")
    assert tp + fp + fn + tn == len(y)

    # @a:metrics
    accuracy = (tp + tn) / len(y)
    precision, recall, f1 = precision_recall_f1(y, y_hat)
    print(f"accuracy {accuracy:.3f}, precision {precision:.3f}, recall {recall:.3f}, F1 {f1:.3f}")
    assert np.isclose(accuracy, np.mean(y == y_hat))
    assert min(precision, recall) <= f1 <= (precision + recall) / 2  # dragged towards the worse

    # @a:threshold
    trade = [precision_recall_f1(y, p >= t)[:2] for t in (0.3, 0.5, 0.7)]
    for t, (pr, rc) in zip((0.3, 0.5, 0.7), trade):
        print(f"threshold {t}: precision {pr:.3f}, recall {rc:.3f}")
    assert trade[0][1] >= trade[1][1] >= trade[2][1]  # stricter threshold: recall can only fall
    assert trade[0][0] < trade[2][0]  # ...while precision rises

    # @a:multiclass
    yk = rng.choice(3, size=600, p=[0.6, 0.3, 0.1])  # unequal class sizes
    yk_hat = np.where(rng.uniform(size=600) < 0.8, yk, rng.integers(0, 3, 600))  # 80% kept
    per_class = [precision_recall_f1(yk == k, yk_hat == k) for k in range(3)]  # one vs rest
    macro_f1 = np.mean([f for _, _, f in per_class])
    micro_precision = sum(confusion(yk == k, yk_hat == k)[0] for k in range(3)) / len(yk)
    print(f"3 classes: macro F1 {macro_f1:.3f}, accuracy {np.mean(yk == yk_hat):.3f}")
    assert np.isclose(micro_precision, np.mean(yk == yk_hat))  # pooled over classes: accuracy
