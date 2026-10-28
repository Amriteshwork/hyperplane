/** Seeded randomness, so every figure draws the same data and trains the same way each visit. */
export type Rng = () => number;

/** mulberry32: a tiny, fast generator of uniform numbers in [0, 1). */
export function seeded(seed: number): Rng {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function uniform(rng: Rng, low: number, high: number): number {
  return low + (high - low) * rng();
}

/** Box–Muller: one standard normal draw. */
export function normal(rng: Rng): number {
  const u = 1 - rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}
