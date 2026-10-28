/** Optimisers on a two-weight ravine, f(w) = ½(A·w₀² + w₁²), for the optimiser-race figure. */
/** Steep across the ravine (w₀), nearly flat along it (w₁). */
export const A = 100;
const START: [number, number] = [-0.9, 1.0];

export type Point = [number, number];

function gradient([a, b]: Point): Point {
  return [A * a, b];
}

export function lossAt([a, b]: Point): number {
  return 0.5 * (A * a * a + b * b);
}

/** Plain gradient descent, just under the step size that would diverge (2/A). */
export function gradientDescent(steps: number, lr = 0.019): Point[] {
  const out: Point[] = [START];
  for (let t = 0; t < steps; t++) {
    const [p, g] = [out[t], gradient(out[t])];
    out.push([p[0] - lr * g[0], p[1] - lr * g[1]]);
  }
  return out;
}

export function momentum(steps: number, lr = 0.01, beta = 0.9): Point[] {
  const out: Point[] = [START];
  let v: Point = [0, 0];
  for (let t = 0; t < steps; t++) {
    const g = gradient(out[t]);
    v = [beta * v[0] + g[0], beta * v[1] + g[1]];
    out.push([out[t][0] - lr * v[0], out[t][1] - lr * v[1]]);
  }
  return out;
}

export function adam(steps: number, lr = 0.05, b1 = 0.9, b2 = 0.999): Point[] {
  const out: Point[] = [START];
  let m: Point = [0, 0];
  let v: Point = [0, 0];
  for (let t = 1; t <= steps; t++) {
    const g = gradient(out[t - 1]);
    m = [b1 * m[0] + (1 - b1) * g[0], b1 * m[1] + (1 - b1) * g[1]];
    v = [b2 * v[0] + (1 - b2) * g[0] ** 2, b2 * v[1] + (1 - b2) * g[1] ** 2];
    const step = (i: 0 | 1) =>
      (lr * (m[i] / (1 - b1 ** t))) / (Math.sqrt(v[i] / (1 - b2 ** t)) + 1e-8);
    out.push([out[t - 1][0] - step(0), out[t - 1][1] - step(1)]);
  }
  return out;
}
