import { thetas } from './rope';
import { LLAMA2, extrapolated, ramp, scaledThetas, temperature, turns } from './yarn';

describe('yarn', () => {
  const original = thetas(LLAMA2.d, LLAMA2.base);

  it('position interpolation slows every clock by s', () => {
    scaledThetas('pi', 16).forEach((t, i) => expect(t).toBeCloseTo(original[i] / 16, 12));
  });

  it('NTK-aware keeps the fastest clock and slows the slowest by exactly s', () => {
    const scaled = scaledThetas('ntk-aware', 16);
    expect(scaled[0]).toBe(1);
    expect(original[63] / scaled[63]).toBeCloseTo(16, 8);
  });

  it('NTK-by-parts leaves fast clocks alone and fully interpolates slow ones', () => {
    const scaled = scaledThetas('by-parts', 16);
    const r = original.map((t) => turns(t, LLAMA2.L));
    expect(r.filter((x) => x > 32)).toHaveLength(21); // pairs 0–20 make over 32 turns
    expect(r.filter((x) => x < 1)).toHaveLength(18); // pairs 46–63 never complete one
    expect(scaled[20]).toBe(original[20]);
    expect(scaled[46]).toBeCloseTo(original[46] / 16, 12);
    expect(ramp(16.5)).toBeCloseTo(0.5, 12);
  });

  it('only NTK-aware pushes clocks past angles they saw in training', () => {
    expect(extrapolated('pi', 16)).toEqual([]);
    expect(extrapolated('by-parts', 16)).toEqual([]);
    expect(extrapolated('ntk-aware', 16)).toEqual(Array.from({ length: 17 }, (_, i) => 46 + i));
  });

  it('sharpens attention more for bigger stretches', () => {
    expect(temperature(1)).toBe(1);
    expect(1 / Math.sqrt(temperature(16))).toBeCloseTo(1.277, 3);
    expect(temperature(32)).toBeLessThan(temperature(16));
  });
});
