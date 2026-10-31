import {
  accuracy,
  cubeDistances,
  distance,
  edgeForFraction,
  flipLabels,
  histogram,
  nearest,
  neighbourMean,
} from './knn';
import { seeded } from './random';
import { toyData } from './toy-data';

describe('distance', () => {
  it('gives the 3-4-5 triangle its three lengths', () => {
    expect(distance([0, 0], [3, 4], 'euclidean')).toBe(5);
    expect(distance([0, 0], [3, 4], 'manhattan')).toBe(7);
    expect(distance([0, 0], [3, 4], 'chebyshev')).toBe(4);
  });

  it('stretching a feature changes who is nearest', () => {
    const X = [
      [0.5, 0],
      [0, 1],
    ];
    expect(nearest(X, [0, 0], 1)[0].index).toBe(0);
    expect(nearest(X, [0, 0], 1, 'euclidean', 10)[0].index).toBe(1);
  });
});

describe('nearest and neighbourMean', () => {
  const X = [[0], [1], [2], [10]];

  it('returns the closest rows, closest first', () => {
    expect(nearest(X, [1.2], 2).map((n) => n.index)).toEqual([1, 2]);
  });

  it('averages uniformly or by inverse distance', () => {
    const y = [0, 10, 20, 100];
    const pair = nearest(X, [1.25], 2);
    expect(neighbourMean(pair, y)).toBe(15);
    // Distances 0.25 and 0.75, so weights 4 and 4/3.
    expect(neighbourMean(pair, y, 'distance')).toBeCloseTo(12.5, 10);
  });

  it('lets an exact match decide alone', () => {
    expect(neighbourMean(nearest(X, [1], 3), [0, 10, 20, 100], 'distance')).toBe(10);
  });
});

describe('k as a bias-variance dial', () => {
  const train = toyData('moons', seeded(7), 150);
  const noisy = flipLabels(train.y, 0.15, seeded(5));
  const test = toyData('moons', seeded(11), 150);

  it('k = 1 memorises the noise, a moderate k generalises, k = n knows nothing', () => {
    const score = (k: number) => accuracy(train.X, noisy, test.X, test.y, k);
    expect(accuracy(train.X, noisy, train.X, noisy, 1)).toBe(1);
    expect(score(15)).toBeGreaterThan(score(1));
    expect(score(150)).toBeLessThan(0.6);
  });
});

describe('the curse of dimensionality', () => {
  it('neighbourhoods of 1% stop being local', () => {
    expect(edgeForFraction(0.01, 2)).toBeCloseTo(0.1, 10);
    expect(edgeForFraction(0.01, 100)).toBeCloseTo(0.955, 3);
  });

  it('nearest and farthest come together as dimensions grow', () => {
    const contrast = (d: number) => {
      const all = cubeDistances(d, 500, seeded(d));
      return Math.max(...all) / Math.min(...all);
    };
    expect(contrast(2)).toBeGreaterThan(10);
    expect(contrast(1000)).toBeLessThan(1.3);
  });

  it('histogram counts every value, including those beyond the ends', () => {
    expect(histogram([-1, 0.1, 0.6, 5], 0, 1, 2)).toEqual([2, 2]);
  });
});
