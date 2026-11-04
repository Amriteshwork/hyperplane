/** Context extension for RoPE (YaRN paper): how each method changes every clock's speed θ_d. */
import { thetas } from './rope';

export type Method = 'pi' | 'ntk-aware' | 'by-parts';

/** Llama 2: 4,096 trained positions, 128 numbers per head, base 10,000. */
export const LLAMA2 = { L: 4096, d: 128, base: 10_000 } as const;
/** NTK-by-parts ramp edges for Llama: below α turns, interpolate fully; above β, leave alone. */
export const ALPHA = 1;
export const BETA = 32;

/** r(d) = L / λ_d: how many full turns clock d makes across the training context. */
export function turns(theta: number, L: number): number {
  return (L * theta) / (2 * Math.PI);
}

/** γ(r): 0 → fully interpolate (θ/s), 1 → leave the clock alone, linear in between. */
export function ramp(r: number): number {
  if (r < ALPHA) return 0;
  if (r > BETA) return 1;
  return (r - ALPHA) / (BETA - ALPHA);
}

/** Every clock's new speed after extending the context s times. */
export function scaledThetas(method: Method, s: number, { L, d, base } = LLAMA2): number[] {
  const original = thetas(d, base);
  switch (method) {
    case 'pi':
      return original.map((t) => t / s);
    case 'ntk-aware': {
      // b' = b · s^(d/(d−2)) slows the last clock by exactly s and leaves the first alone.
      return thetas(d, base * s ** (d / (d - 2)));
    }
    case 'by-parts':
      return original.map((t) => {
        const g = ramp(turns(t, L));
        return (1 - g) * (t / s) + g * t;
      });
  }
}

/** Clocks that, at the new length, reach angles they never reached in training: extrapolation.
 * A clock that completed a full turn in training has seen every angle, so it's always safe. */
export function extrapolated(method: Method, s: number, setup = LLAMA2): number[] {
  const original = thetas(setup.d, setup.base);
  const scaled = scaledThetas(method, s, setup);
  return original.flatMap((t, i) => {
    const seenAll = t * setup.L >= 2 * Math.PI;
    const reached = scaled[i] * setup.L * s;
    return !seenAll && reached > t * setup.L * (1 + 1e-9) ? [i] : [];
  });
}

/** YaRN's attention temperature: logits are divided by t, with √(1/t) = 0.1·ln(s) + 1. */
export function temperature(s: number): number {
  return 1 / (0.1 * Math.log(s) + 1) ** 2;
}
