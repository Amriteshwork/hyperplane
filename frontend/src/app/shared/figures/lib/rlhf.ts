/** RLHF's figures: the replies and base model from content/concepts/rlhf/code/numpy.py, and its maths. */

export const REPLIES = [
  'maybe figs.',
  'no idea.',
  'jam.',
  'jam, thanks.',
  'jam, thanks for asking.',
  'JAM! thanks thanks for asking, asking, asking!',
] as const;
/** What the reward model sees: is it right, is it polite, how many extra words. */
export const FEATURES: readonly (readonly number[])[] = [
  [0, 0, 0],
  [0, 0, 0],
  [1, 0, 0],
  [1, 1, 0],
  [1, 1, 1],
  [1, 1, 3],
];
/** What people really want. Hidden from the reward model; nobody ever rated the last reply. */
export const TRUE_QUALITY = [0, 0.5, 2, 3, 3.3, -3] as const;
export const BASE = [0.25, 0.18, 0.4, 0.1, 0.05, 0.02] as const;
/** The weights `fit_reward` learns on that page from 400 noisy votes. */
export const LEARNED_WEIGHTS = [4.01, 2.61, 0.95] as const;

const dot = (a: readonly number[], b: readonly number[]) =>
  a.reduce((sum, v, i) => sum + v * b[i], 0);

export const rewards = (weights: readonly number[]): number[] =>
  FEATURES.map((f) => dot(f, weights));

/** The exact answer to "maximise reward − β·KL from base": p ∝ base · exp(reward / β). */
export function bestPolicy(
  base: readonly number[],
  scores: readonly number[],
  beta: number,
): number[] {
  const logs = base.map((p, i) => Math.log(p) + scores[i] / beta);
  const top = Math.max(...logs);
  const raw = logs.map((l) => Math.exp(l - top));
  const total = raw.reduce((a, b) => a + b, 0);
  return raw.map((v) => v / total);
}

export const expected = (p: readonly number[], values: readonly number[]): number => dot(p, values);

/** One vote per pair, winner first. Logistic regression on the difference between their features. */
export function fitReward(
  votes: readonly (readonly [number, number])[],
  learningRate = 0.5,
  steps = 400,
): number[] {
  const w = [0, 0, 0];
  if (!votes.length) return w;
  for (let s = 0; s < steps; s++) {
    const step = [0, 0, 0];
    for (const [a, b] of votes) {
      const diff = FEATURES[a].map((v, i) => v - FEATURES[b][i]);
      const miss = 1 - 1 / (1 + Math.exp(-dot(diff, w)));
      diff.forEach((v, i) => (step[i] += (miss * v) / votes.length));
    }
    step.forEach((v, i) => (w[i] += learningRate * v));
  }
  return w;
}
