import { leastSquares } from './least-squares';

describe('leastSquares', () => {
  it('recovers exact coefficients', () => {
    const A = [0, 1, 2, 3, 4].map((x) => [1, x, x * x]);
    const coef = leastSquares(
      A,
      A.map(([, x, x2]) => 2 - 3 * x + 0.5 * x2),
    );
    [2, -3, 0.5].forEach((want, i) => expect(coef[i]).toBeCloseTo(want, 6));
  });
});
