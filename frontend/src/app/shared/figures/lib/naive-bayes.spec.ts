import {
  correlatedBlobs,
  fitGaussians,
  fromLogScores,
  logNormal,
  predictProba,
} from './naive-bayes';
import { seeded } from './random';

describe('naive bayes maths', () => {
  it('computes the log of the normal density', () => {
    expect(logNormal(0, 0, 1)).toBeCloseTo(Math.log(1 / Math.sqrt(2 * Math.PI)));
    expect(logNormal(1, 0, 1)).toBeCloseTo(logNormal(0, 0, 1) - 0.5);
  });

  it('turns very negative log scores into probabilities without underflowing', () => {
    const [a, b] = fromLogScores([-5000, -5000 - Math.log(3)]);
    expect(a).toBeCloseTo(0.75);
    expect(b).toBeCloseTo(0.25);
  });

  it('fits priors, means and variances per class', () => {
    const model = fitGaussians([[0], [2], [10], [10], [10], [10]], [0, 0, 1, 1, 1, 1], 2);
    expect(model.priors[0]).toBeCloseTo(1 / 3);
    expect(model.means).toEqual([[1], [10]]);
    expect(model.variances).toEqual([[1], [0]]);
  });

  it('applies Bayes with independent features', () => {
    const model = {
      priors: [0.5, 0.5],
      means: [
        [-1, -1],
        [1, 1],
      ],
      variances: [
        [1, 1],
        [1, 1],
      ],
    };
    // Log-likelihood gap per feature is 2x, so at (0.5, 0) the odds of class 1 are e^1.
    expect(predictProba(model, [0.5, 0])[1]).toBeCloseTo(Math.E / (1 + Math.E));
  });

  it('generates data with the requested within-class correlation', () => {
    const { X, y } = correlatedBlobs(
      seeded(1),
      8000,
      [
        [0, 0],
        [3, 3],
      ],
      0.8,
    );
    const rows = X.filter((_, i) => y[i] === 0);
    const mean = (j: number) => rows.reduce((s, r) => s + r[j], 0) / rows.length;
    const [ma, mb] = [mean(0), mean(1)];
    const cov = rows.reduce((s, r) => s + (r[0] - ma) * (r[1] - mb), 0) / rows.length;
    expect(cov).toBeCloseTo(0.8, 1);
  });
});
