import { encoding, frequency, similarity } from './positional';

describe('positional encoding', () => {
  it('starts every clock at angle 0', () => {
    encoding(0, 8).forEach((v, j) => expect(v).toBe(j % 2 ? 1 : 0));
  });

  it('runs the first clock at 1 radian per token, the last ones slowly', () => {
    expect(frequency(0, 32)).toBe(1);
    expect(frequency(15, 32)).toBeLessThan(0.001);
  });

  it('makes similarity depend only on the distance', () => {
    expect(similarity(5, 8, 32)).toBeCloseTo(similarity(40, 43, 32), 10);
    expect(similarity(7, 7, 32)).toBeCloseTo(16, 10); // d/2 clocks, each sin² + cos² = 1
    expect(similarity(7, 7, 32)).toBeGreaterThan(similarity(7, 9, 32));
  });
});
