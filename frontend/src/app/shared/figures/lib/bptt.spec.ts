import { unroll } from './bptt';

const XS = [1, 0.2, -0.3, 0.1, 0.25, -0.15, 0.05, 0.3, -0.2, 0.1];

describe('backpropagation through time', () => {
  it('sums to the true gradient of the shared weight', () => {
    for (const activation of ['tanh', 'linear'] as const) {
      const w = 0.9;
      const eps = 1e-6;
      const numeric =
        (unroll(XS, w + eps, activation, 1).loss - unroll(XS, w - eps, activation, 1).loss) /
        (2 * eps);
      const total = unroll(XS, w, activation, 1).dW.reduce((a, b) => a + b, 0);
      expect(total).toBeCloseTo(numeric, 6);
    }
  });

  it('vanishes below w = 1 and explodes above it, without tanh', () => {
    const reach = (w: number) => {
      const { delta } = unroll(XS, w, 'linear', 5);
      return Math.abs(delta[1] / delta[XS.length]);
    };
    expect(reach(0.5)).toBeCloseTo(0.5 ** 9, 10);
    expect(reach(1.5)).toBeCloseTo(1.5 ** 9, 6);
  });

  it('with tanh, every step back can only shrink the gradient more than w alone', () => {
    const { factor } = unroll(XS, 1.5, 'tanh', 1);
    for (const f of factor.slice(1)) expect(Math.abs(f)).toBeLessThanOrEqual(1.5);
  });
});
