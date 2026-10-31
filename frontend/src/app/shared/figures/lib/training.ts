/** LLM training's figures, as plain functions: the learning-rate schedule, clipping and dropout. */
import { Rng } from './random';

/** A linear climb to `peak`, then a half cosine down to 0. `warmup = 0` means constant. */
export function learningRate(step: number, peak: number, warmup: number, total: number): number {
  if (warmup === 0) return peak;
  if (step < warmup) return (peak * (step + 1)) / warmup;
  return peak * 0.5 * (1 + Math.cos((Math.PI * (step - warmup)) / (total - warmup)));
}

/** A gradient shrunk to length `maxNorm` if it's longer. Its direction never changes. */
export function clipByNorm(
  gradient: readonly [number, number],
  maxNorm: number,
): { clipped: [number, number]; norm: number } {
  const norm = Math.hypot(...gradient);
  const scale = norm > maxNorm ? maxNorm / norm : 1;
  return { clipped: [gradient[0] * scale, gradient[1] * scale], norm };
}

/** One dropout mask: true where a unit stays on. */
export function dropoutMask(size: number, rate: number, rng: Rng): boolean[] {
  return Array.from({ length: size }, () => rng() >= rate);
}
