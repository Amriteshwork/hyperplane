/** Helpers for hand-drawn SVG charts: linear scales and polyline paths. */
export type Scale = (value: number) => number;

/** Maps [d0, d1] onto [r0, r1]. */
export function linear(d0: number, d1: number, r0: number, r1: number): Scale {
  return (value) => r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);
}

/** An SVG path through the points, `x` and `y` already in pixels. */
export function path(points: readonly (readonly [number, number])[]): string {
  return points.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`).join('');
}

/** `count` evenly spaced values from `low` to `high`, both included. */
export function range(low: number, high: number, count: number): number[] {
  return Array.from({ length: count }, (_, i) => low + ((high - low) * i) / (count - 1));
}
