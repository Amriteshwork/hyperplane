/** Sinusoidal positional encoding: one (sin, cos) clock per pair of dimensions. */

/** ωᵢ = 1 / 10000^(2i/d): the speed of clock i, in radians per token. */
export function frequency(i: number, d: number): number {
  return 1 / 10000 ** ((2 * i) / d);
}

/** p_t: [sin(tω₀), cos(tω₀), sin(tω₁), cos(tω₁), …], length d. */
export function encoding(t: number, d: number): number[] {
  return Array.from({ length: d }, (_, j) => {
    const angle = t * frequency(Math.floor(j / 2), d);
    return j % 2 ? Math.cos(angle) : Math.sin(angle);
  });
}

/** p_t · p_s, which depends only on t − s: Σᵢ cos((t − s)ωᵢ). */
export function similarity(t: number, s: number, d: number): number {
  const p = encoding(t, d);
  return encoding(s, d).reduce((sum, v, j) => sum + v * p[j], 0);
}
