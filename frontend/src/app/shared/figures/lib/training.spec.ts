import { clipByNorm, dropoutMask, learningRate } from './training';
import { seeded } from './random';

describe('training maths', () => {
  it('the schedule matches the NumPy page', () => {
    const rates = Array.from({ length: 1000 }, (_, s) => learningRate(s, 1e-3, 100, 1000));
    expect(Math.max(...rates)).toBeCloseTo(1e-3, 8);
    expect(rates.indexOf(Math.max(...rates))).toBe(99);
    expect(rates[0]).toBeLessThan(2e-5);
    expect(rates[999]).toBeLessThan(1e-7);
    expect(learningRate(5, 0.1, 0, 100)).toBe(0.1);
  });

  it('clipping keeps the direction and leaves short gradients alone', () => {
    const { clipped, norm } = clipByNorm([30, 40], 1);
    expect(norm).toBe(50);
    expect(clipped[0]).toBeCloseTo(0.6);
    expect(clipped[1]).toBeCloseTo(0.8);
    expect(clipByNorm([0.3, 0.4], 1).clipped).toEqual([0.3, 0.4]);
  });

  it('dropout silences about the requested share', () => {
    const mask = dropoutMask(10_000, 0.2, seeded(1));
    expect(Math.abs(mask.filter((on) => !on).length / 10_000 - 0.2)).toBeLessThan(0.02);
  });
});
