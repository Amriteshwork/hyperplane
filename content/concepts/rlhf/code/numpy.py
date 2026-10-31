"""RLHF in NumPy: learn a reward from "A beats B" votes, shift a policy towards it, and watch what overdoing it does."""

import numpy as np

# @a:reuse
from sigmoid_numpy import sigmoid
from softmax_numpy import softmax

# @a:replies
REPLIES = ["maybe figs.", "no idea.", "jam.", "jam, thanks.", "jam, thanks for asking.",
           "JAM! thanks thanks for asking, asking, asking!"]
#: What a reward model can see: is it right, is it polite, how many extra words.
FEATURES = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 3]], dtype=float)
#: What people really want. Nobody ever rated the last reply, so the reward model can't know.
TRUE_QUALITY = np.array([0.0, 0.5, 2.0, 3.0, 3.3, -3.0])
#: What the base model already says.
BASE = np.array([0.25, 0.18, 0.40, 0.10, 0.05, 0.02])


# @a:reward
def fit_reward(pairs, features, lr=0.5, steps=500):
    """P(A beats B) = σ(r(A) − r(B)), with r linear in features. This is logistic regression on differences."""
    w = np.zeros(features.shape[1])
    diffs = features[pairs[:, 0]] - features[pairs[:, 1]]
    for _ in range(steps):
        w += lr * ((1 - sigmoid(diffs @ w))[:, None] * diffs).mean(axis=0)
    return w


# @a:policy
def best_policy(base, rewards, beta):
    """The best answer to "maximise reward, minus β × KL from the base model" has this exact form."""
    return softmax((np.log(base) + rewards / beta)[None])[0]


# @a:reinforce
def reinforce(base, rewards, beta, rng, lr=0.05, steps=4000, batch=64):
    """The same goal learnt by trial and error: sample replies, then raise the odds of the ones that scored."""
    logits = np.log(base).copy()
    for _ in range(steps):
        p = softmax(logits[None])[0]
        drawn = rng.choice(len(p), size=batch, p=p)
        score = rewards[drawn] - beta * (np.log(p[drawn]) - np.log(base[drawn]))
        advantage = score - score.mean()  # compare with the average, so gradients are less noisy
        logits += lr * (advantage[:, None] * (np.eye(len(p))[drawn] - p)).mean(axis=0)
    return softmax(logits[None])[0]


# @a:demo
if __name__ == "__main__":
    # @a:votes
    # Raters compared normal replies: right beats wrong, polite beats blunt, a few more words help.
    rng = np.random.default_rng(0)
    seen = [0, 1, 2, 3, 4]
    a, b = rng.choice(seen, 400), rng.choice(seen, 400)
    keep = a != b
    a, b = a[keep], b[keep]
    votes = sigmoid(3 * (TRUE_QUALITY[a] - TRUE_QUALITY[b])) > rng.random(len(a))
    pairs = np.where(votes[:, None], np.c_[a, b], np.c_[b, a])  # winner first
    w = fit_reward(pairs, FEATURES)
    rewards = FEATURES @ w
    print(f"reward weights (right, polite, wordy): {np.round(w, 2)}")
    print(f"rewards {np.round(rewards, 2)}")
    assert w[0] > 0 and w[1] > 0 and w[2] > 0
    assert rewards[5] > rewards[4]  # the score extrapolates: more words, more reward. That's the trap.

    # @a:shift
    for beta in (50.0, 10.0, 5.0, 3.0, 2.0, 1.0, 0.2):
        p = best_policy(BASE, rewards, beta)
        print(f"β={beta:<5} policy {np.round(p, 2)}  real quality {p @ TRUE_QUALITY:+.2f}")

    # @a:agree
    closed_form = best_policy(BASE, rewards, 1.0)
    learned = reinforce(BASE, rewards, 1.0, rng)
    print(f"closed form {np.round(closed_form, 2)}\nREINFORCE   {np.round(learned, 2)}")
    assert np.abs(closed_form - learned).max() < 0.05

    # @a:goodhart
    gentle = best_policy(BASE, rewards, 5.0)
    extreme = best_policy(BASE, rewards, 0.2)
    assert gentle @ TRUE_QUALITY > BASE @ TRUE_QUALITY  # moderate RLHF really does help
    assert extreme @ TRUE_QUALITY < gentle @ TRUE_QUALITY  # chase the score too hard and quality falls
    print(f"real quality: base {BASE @ TRUE_QUALITY:+.2f}, gentle {gentle @ TRUE_QUALITY:+.2f}, "
          f"extreme {extreme @ TRUE_QUALITY:+.2f}")
