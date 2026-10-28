import { Rng, normal, uniform } from './random';

/** Two-class datasets in [-1, 1]², each needing a boundary that a straight line can't draw. */
export type ToyDataset = 'circle' | 'xor' | 'moons' | 'spiral';

export interface Points {
  X: number[][];
  /** 0 or 1. */
  y: number[];
}

const MAKERS: Record<ToyDataset, (rng: Rng, n: number) => Points> = {
  circle: (rng, n) =>
    collect(n, () => {
      const [a, b] = [uniform(rng, -1, 1), uniform(rng, -1, 1)];
      return [[a, b], a * a + b * b < 0.4 ? 1 : 0];
    }),
  xor: (rng, n) =>
    collect(n, () => {
      const [a, b] = [uniform(rng, -1, 1), uniform(rng, -1, 1)];
      return [[a, b], a * b > 0 ? 1 : 0];
    }),
  moons: (rng, n) =>
    collect(n, (i) => {
      const label = i % 2;
      const t = Math.PI * rng();
      const [a, b] = label ? [1 - Math.cos(t), 0.5 - Math.sin(t)] : [Math.cos(t), Math.sin(t)];
      return [[(a - 0.5) / 1.6 + 0.08 * normal(rng), (b - 0.25) / 1.1 + 0.08 * normal(rng)], label];
    }),
  spiral: (rng, n) =>
    collect(n, (i) => {
      const label = i % 2;
      const r = 0.1 + 0.85 * rng();
      const angle = 3.2 * r * Math.PI + label * Math.PI;
      return [
        [r * Math.cos(angle) + 0.03 * normal(rng), r * Math.sin(angle) + 0.03 * normal(rng)],
        label,
      ];
    }),
};

export function toyData(name: ToyDataset, rng: Rng, n = 200): Points {
  return MAKERS[name](rng, n);
}

function collect(n: number, draw: (i: number) => [number[], number]): Points {
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const [x, label] = draw(i);
    X.push(x);
    y.push(label);
  }
  return { X, y };
}
