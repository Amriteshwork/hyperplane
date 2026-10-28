/** A one-unit RNN, h_t = act(w·h_{t−1} + x_t), with a loss on its last state only. */
export type Activation = 'tanh' | 'linear';

export interface Unrolled {
  /** h_0 … h_T, starting from h_0 = 0. */
  h: number[];
  /** δ_t = ∂L/∂h_t for t = 1 … T (index 0 unused). */
  delta: number[];
  /** ∂h_t/∂h_{t−1} = w · act′ at step t: what each step back multiplies δ by. */
  factor: number[];
  /** Each step's share of ∂L/∂w; the shared weight's gradient is their sum. */
  dW: number[];
  loss: number;
}

/** Forward through every step, then backpropagation through time from the last step to the first. */
export function unroll(
  xs: readonly number[],
  w: number,
  activation: Activation,
  target: number,
): Unrolled {
  const act = activation === 'tanh' ? Math.tanh : (z: number) => z;
  const T = xs.length;
  const h = [0];
  for (const x of xs) h.push(act(w * h[h.length - 1] + x));
  const slope = (t: number) => (activation === 'tanh' ? 1 - h[t] ** 2 : 1);

  const delta = new Array<number>(T + 1).fill(0);
  const factor = new Array<number>(T + 1).fill(0);
  const dW = new Array<number>(T + 1).fill(0);
  delta[T] = h[T] - target; // L = ½(h_T − y)²
  for (let t = T; t >= 1; t--) {
    factor[t] = w * slope(t);
    dW[t] = delta[t] * slope(t) * h[t - 1]; // this step's use of w
    if (t > 1) delta[t - 1] = delta[t] * factor[t]; // hand δ one step back
  }
  return { h, delta, factor, dW, loss: 0.5 * (h[T] - target) ** 2 };
}
