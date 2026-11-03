/** Rotary position embedding (RoPE): rotate each (x, y) pair of q and k by position × θᵢ. */

export type Vec2 = readonly [number, number];

/** Rotates a 2D vector anticlockwise by `angle` radians. */
export function rotate([x, y]: Vec2, angle: number): Vec2 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [x * c - y * s, x * s + y * c];
}

export function dot(a: Vec2, b: Vec2): number {
  return a[0] * b[0] + a[1] * b[1];
}

/** θᵢ = base^(−2i/d) for the d/2 pairs: the first turns 1 radian per token, the last barely. */
export function thetas(d: number, base: number): number[] {
  return Array.from({ length: d / 2 }, (_, i) => base ** ((-2 * i) / d));
}

/** Score of a query and key that match perfectly, `distance` tokens apart, as a share of the
 * score at distance 0: the mean of cos(distance · θᵢ). */
export function alignedScore(distance: number, d: number, base: number): number {
  const all = thetas(d, base);
  return all.reduce((sum, t) => sum + Math.cos(distance * t), 0) / all.length;
}

/** The paper's decay bound (section 3.4.3): the mean of |Sⱼ|, Sⱼ = Σ_{i<j} e^{i·distance·θᵢ},
 * as a share of its value at distance 0. */
export function decayBound(distance: number, d: number, base: number): number {
  let re = 0;
  let im = 0;
  let total = 0;
  const all = thetas(d, base);
  for (const t of all) {
    re += Math.cos(distance * t);
    im += Math.sin(distance * t);
    total += Math.hypot(re, im);
  }
  const atZero = (all.length * (all.length + 1)) / 2; // |Sⱼ| = j when every arrow lines up
  return total / atZero;
}
