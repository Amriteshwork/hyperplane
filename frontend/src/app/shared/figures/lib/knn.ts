/** k-nearest neighbours for the figures: the same sums as the concept pages' NumPy code. */
import { Rng } from './random';

export type Metric = 'euclidean' | 'manhattan' | 'chebyshev';
export type Weights = 'uniform' | 'distance';

/** The distance between two rows; `stretch` multiplies the first feature, as bad units would. */
export function distance(a: readonly number[], b: readonly number[], metric: Metric, stretch = 1) {
  const gaps = a.map((v, i) => Math.abs(v - b[i]) * (i === 0 ? stretch : 1));
  if (metric === 'manhattan') return gaps.reduce((sum, g) => sum + g, 0);
  if (metric === 'chebyshev') return Math.max(...gaps);
  return Math.sqrt(gaps.reduce((sum, g) => sum + g * g, 0));
}

export interface Neighbour {
  index: number;
  distance: number;
}

/** The query's k closest rows, closest first. */
export function nearest(
  X: readonly (readonly number[])[],
  query: readonly number[],
  k: number,
  metric: Metric = 'euclidean',
  stretch = 1,
): Neighbour[] {
  return X.map((row, index) => ({ index, distance: distance(row, query, metric, stretch) }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, k);
}

/** Each neighbour's say. An exact match decides alone, as in scikit-learn. */
function weigh(neighbours: readonly Neighbour[], weights: Weights): number[] {
  if (weights === 'uniform') return neighbours.map(() => 1);
  if (neighbours.some((n) => n.distance === 0))
    return neighbours.map((n) => (n.distance === 0 ? 1 : 0));
  return neighbours.map((n) => 1 / n.distance);
}

/** The weighted mean of `y` over the neighbours: a regression, or for 0/1 labels P(class 1). */
export function neighbourMean(
  neighbours: readonly Neighbour[],
  y: readonly number[],
  weights: Weights = 'uniform',
): number {
  const w = weigh(neighbours, weights);
  const total = w.reduce((sum, v) => sum + v, 0);
  return neighbours.reduce((sum, n, i) => sum + w[i] * y[n.index], 0) / total;
}

/** The share of rows the model labels correctly; `labels` come from these training rows. */
export function accuracy(
  X: readonly (readonly number[])[],
  y: readonly number[],
  query: readonly (readonly number[])[],
  labels: readonly number[],
  k: number,
  weights: Weights = 'uniform',
): number {
  const right = query.filter((row, i) => {
    const p = neighbourMean(nearest(X, row, k), y, weights);
    return (p > 0.5 ? 1 : 0) === labels[i];
  });
  return right.length / query.length;
}

/** Flip each label with this probability, so there is noise for a small k to memorise. */
export function flipLabels(y: readonly number[], rate: number, rng: Rng): number[] {
  return y.map((label) => (rng() < rate ? 1 - label : label));
}

/** The side of a sub-cube of the unit cube holding this fraction of its volume. */
export function edgeForFraction(fraction: number, dimensions: number): number {
  return fraction ** (1 / dimensions);
}

/** The distance from one random point in the unit cube to `rows` others. */
export function cubeDistances(dimensions: number, rows: number, rng: Rng): number[] {
  const query = Array.from({ length: dimensions }, () => rng());
  return Array.from({ length: rows }, () => {
    let squared = 0;
    for (let d = 0; d < dimensions; d++) squared += (rng() - query[d]) ** 2;
    return Math.sqrt(squared);
  });
}

/** How many values fall in each of `bins` equal cells of [low, high]; outliers join the ends. */
export function histogram(values: readonly number[], low: number, high: number, bins: number) {
  const counts = new Array<number>(bins).fill(0);
  for (const value of values) {
    const cell = Math.floor(((value - low) / (high - low)) * bins);
    counts[Math.min(bins - 1, Math.max(0, cell))]++;
  }
  return counts;
}
