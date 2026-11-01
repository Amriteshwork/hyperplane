/** Gaussian Naive Bayes for the figures: the same sums as the concept pages' NumPy code. */
import { Rng, normal } from './random';

export interface GaussianModel {
  priors: number[];
  /** One row per class and one column per feature. */
  means: number[][];
  variances: number[][];
}

export function logNormal(x: number, mean: number, variance: number): number {
  return -0.5 * Math.log(2 * Math.PI * variance) - (x - mean) ** 2 / (2 * variance);
}

/** Normalises log scores into probabilities, subtracting the largest so nothing underflows. */
export function fromLogScores(scores: readonly number[]): number[] {
  const top = Math.max(...scores);
  const raw = scores.map((s) => Math.exp(s - top));
  const total = raw.reduce((sum, v) => sum + v, 0);
  return raw.map((v) => v / total);
}

export function fitGaussians(
  X: readonly (readonly number[])[],
  y: readonly number[],
  classes: number,
): GaussianModel {
  const rows = Array.from({ length: classes }, (_, c) => X.filter((_, i) => y[i] === c));
  const columns = X[0].length;
  const means = rows.map((r) =>
    Array.from({ length: columns }, (_, j) => r.reduce((sum, row) => sum + row[j], 0) / r.length),
  );
  const variances = rows.map((r, c) =>
    means[c].map((m, j) => r.reduce((sum, row) => sum + (row[j] - m) ** 2, 0) / r.length),
  );
  return { priors: rows.map((r) => r.length / X.length), means, variances };
}

/** P(class | row), with features treated as independent given the class. */
export function predictProba(model: GaussianModel, row: readonly number[]): number[] {
  return fromLogScores(
    model.priors.map(
      (prior, c) =>
        Math.log(prior) +
        row.reduce((sum, x, j) => sum + logNormal(x, model.means[c][j], model.variances[c][j]), 0),
    ),
  );
}

/** Two features with this correlation within each class, around each class's centre. */
export function correlatedBlobs(
  rng: Rng,
  rows: number,
  centres: readonly (readonly [number, number])[],
  correlation: number,
): { X: number[][]; y: number[] } {
  const y = Array.from({ length: rows }, (_, i) => i % centres.length);
  const X = y.map((c) => {
    const shared = normal(rng);
    const own = normal(rng);
    return [
      centres[c][0] + shared,
      centres[c][1] + correlation * shared + Math.sqrt(1 - correlation ** 2) * own,
    ];
  });
  return { X, y };
}
