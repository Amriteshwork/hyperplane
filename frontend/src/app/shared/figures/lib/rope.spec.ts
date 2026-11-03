import { alignedScore, decayBound, dot, rotate, thetas } from './rope';

describe('rope', () => {
  it('rotates without changing length', () => {
    const [x, y] = rotate([3, 4], 1.234);
    expect(Math.hypot(x, y)).toBeCloseTo(5, 10);
    const [qx, qy] = rotate([1, 0], Math.PI / 2);
    expect(qx).toBeCloseTo(0, 10);
    expect(qy).toBeCloseTo(1, 10);
  });

  it('makes the score depend only on the distance', () => {
    const q = [0.8, 0.3] as const;
    const k = [0.2, -0.9] as const;
    const score = (m: number, n: number) => dot(rotate(q, m * 0.4), rotate(k, n * 0.4));
    expect(score(2, 5)).toBeCloseTo(score(12, 15), 10);
    expect(score(2, 5)).not.toBeCloseTo(score(2, 6), 3);
  });

  it('spaces the frequencies geometrically from 1 down', () => {
    const all = thetas(128, 10000);
    expect(all).toHaveLength(64);
    expect(all[0]).toBe(1);
    expect(all[1] / all[0]).toBeCloseTo(all[63] / all[62], 10);
  });

  it('starts every curve at 1 and lets it fall with distance', () => {
    expect(alignedScore(0, 128, 10000)).toBe(1);
    expect(decayBound(0, 128, 10000)).toBeCloseTo(1, 10);
    expect(alignedScore(1000, 128, 10000)).toBeLessThan(alignedScore(10, 128, 10000));
    expect(decayBound(1000, 128, 10000)).toBeLessThan(decayBound(10, 128, 10000));
    // A bigger base slows every clock, so the fall comes later.
    expect(alignedScore(100, 128, 500000)).toBeGreaterThan(alignedScore(100, 128, 10000));
  });
});
