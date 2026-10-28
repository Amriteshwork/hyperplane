import { Rng, uniform } from './random';

/**
 * A small fully connected network in plain TypeScript, for figures that train live in the
 * page. Same maths as the NumPy pages: forward pass, backpropagation, Adam, full batches.
 */
export type Activation = 'relu' | 'tanh' | 'logistic' | 'identity';
/** `identity` for regression (MSE); `sigmoid` for two classes (log-loss). */
export type Head = 'identity' | 'sigmoid';
type Matrix = number[][];

export interface Layer {
  /** `w[i][j]`: from input i to neuron j. */
  w: Matrix;
  b: number[];
}

export interface Network {
  layers: Layer[];
  activation: Activation;
  head: Head;
}

export const ACTIVATIONS: Record<
  Activation,
  { f: (z: number) => number; slope: (a: number) => number }
> = {
  relu: { f: (z) => Math.max(0, z), slope: (a) => (a > 0 ? 1 : 0) },
  tanh: { f: Math.tanh, slope: (a) => 1 - a * a },
  logistic: { f: sigmoid, slope: (a) => a * (1 - a) },
  identity: { f: (z) => z, slope: () => 1 },
};

export function sigmoid(z: number): number {
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

/** Glorot-uniform weights, zero biases. `sizes` = [inputs, ...hidden, outputs]. */
export function createNetwork(
  sizes: number[],
  activation: Activation,
  head: Head,
  rng: Rng,
): Network {
  const layers = sizes.slice(1).map((outputs, i) => {
    const inputs = sizes[i];
    const bound = Math.sqrt(6 / (inputs + outputs));
    return {
      w: Array.from({ length: inputs }, () =>
        Array.from({ length: outputs }, () => uniform(rng, -bound, bound)),
      ),
      b: new Array<number>(outputs).fill(0),
    };
  });
  return { layers, activation, head };
}

/** Every layer's outputs for every row, inputs first. */
export function forward(net: Network, X: Matrix): Matrix[] {
  const bend = ACTIVATIONS[net.activation].f;
  const outputs = [X];
  net.layers.forEach((layer, l) => {
    const last = l === net.layers.length - 1;
    const below = outputs[outputs.length - 1];
    outputs.push(
      below.map((row) =>
        layer.b.map((bias, j) => {
          let z = bias;
          for (let i = 0; i < row.length; i++) z += row[i] * layer.w[i][j];
          if (!last) return bend(z);
          return net.head === 'sigmoid' ? sigmoid(z) : z;
        }),
      ),
    );
  });
  return outputs;
}

/** The network's single output for each row. */
export function predict(net: Network, X: Matrix): number[] {
  const outputs = forward(net, X);
  return outputs[outputs.length - 1].map((row) => row[0]);
}

/** Mean loss (MSE, or log-loss for a sigmoid head) of the network's output against `y`. */
export function loss(net: Network, X: Matrix, y: number[]): number {
  const p = predict(net, X);
  let total = 0;
  for (let i = 0; i < y.length; i++) {
    if (net.head === 'identity') total += (p[i] - y[i]) ** 2;
    else {
      const q = Math.min(Math.max(p[i], 1e-12), 1 - 1e-12);
      total -= y[i] * Math.log(q) + (1 - y[i]) * Math.log(1 - q);
    }
  }
  return total / y.length;
}

/** Backpropagation: the gradient of `loss` for every weight and bias, layer by layer. */
export function gradients(net: Network, X: Matrix, y: number[]): Layer[] {
  const outputs = forward(net, X);
  const n = y.length;
  const scale = net.head === 'identity' ? 2 / n : 1 / n; // MSE's 2(ŷ − y)/n, log-loss's (p − y)/n
  let delta = outputs[outputs.length - 1].map((row, i) => [scale * (row[0] - y[i])]);
  const slope = ACTIVATIONS[net.activation].slope;
  const grads: Layer[] = new Array<Layer>(net.layers.length);
  for (let l = net.layers.length - 1; l >= 0; l--) {
    const { w, b } = net.layers[l];
    const below = outputs[l];
    grads[l] = {
      w: w.map((row, i) => row.map((_, j) => sum(below, (r, k) => r[i] * delta[k][j]))),
      b: b.map((_, j) => sum(delta, (r) => r[j])),
    };
    if (l > 0) {
      delta = delta.map((d, k) =>
        w.map((row, i) => slope(below[k][i]) * row.reduce((s, wij, j) => s + wij * d[j], 0)),
      );
    }
  }
  return grads;
}

function sum(rows: Matrix, term: (row: number[], k: number) => number): number {
  let total = 0;
  for (let k = 0; k < rows.length; k++) total += term(rows[k], k);
  return total;
}

/** Adam over every weight of a network, updated in place. */
export class Adam {
  private t = 0;
  private readonly m: Layer[];
  private readonly v: Layer[];

  constructor(
    net: Network,
    private readonly learningRate = 0.01,
    private readonly beta1 = 0.9,
    private readonly beta2 = 0.999,
  ) {
    this.m = zerosLike(net);
    this.v = zerosLike(net);
  }

  step(net: Network, grads: Layer[]): void {
    this.t++;
    const lr =
      (this.learningRate * Math.sqrt(1 - this.beta2 ** this.t)) / (1 - this.beta1 ** this.t);
    const update = (p: number[], g: number[], m: number[], v: number[]) => {
      for (let i = 0; i < p.length; i++) {
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * g[i];
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * g[i] ** 2;
        p[i] -= (lr * m[i]) / (Math.sqrt(v[i]) + 1e-8);
      }
    };
    net.layers.forEach((layer, l) => {
      layer.w.forEach((row, i) => update(row, grads[l].w[i], this.m[l].w[i], this.v[l].w[i]));
      update(layer.b, grads[l].b, this.m[l].b, this.v[l].b);
    });
  }
}

function zerosLike(net: Network): Layer[] {
  return net.layers.map(({ w, b }) => ({
    w: w.map((row) => row.map(() => 0)),
    b: b.map(() => 0),
  }));
}
