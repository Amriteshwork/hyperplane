import { seeded } from './random';
import { Adam, Head, createNetwork, gradients, loss } from './tiny-mlp';

const X = [
  [0.5, -1],
  [-0.3, 0.8],
  [1.2, 0.1],
  [-1, -0.4],
];

describe('tiny-mlp', () => {
  it.each<[Head, number[]]>([
    ['identity', [0.3, -0.7, 1.1, 0.2]],
    ['sigmoid', [1, 0, 1, 0]],
  ])('backpropagation matches finite differences (%s head)', (head, y) => {
    const net = createNetwork([2, 3, 2, 1], 'tanh', head, seeded(1));
    const grads = gradients(net, X, y);
    const h = 1e-6;
    net.layers.forEach((layer, l) => {
      layer.w.forEach((row, i) =>
        row.forEach((_, j) => {
          const w = row[j];
          row[j] = w + h;
          const up = loss(net, X, y);
          row[j] = w - h;
          const down = loss(net, X, y);
          row[j] = w;
          expect(grads[l].w[i][j]).toBeCloseTo((up - down) / (2 * h), 6);
        }),
      );
      layer.b.forEach((b, j) => {
        layer.b[j] = b + h;
        const up = loss(net, X, y);
        layer.b[j] = b - h;
        const down = loss(net, X, y);
        layer.b[j] = b;
        expect(grads[l].b[j]).toBeCloseTo((up - down) / (2 * h), 6);
      });
    });
  });

  it('Adam drives the training loss down', () => {
    const y = [0.3, -0.7, 1.1, 0.2];
    const net = createNetwork([2, 8, 1], 'relu', 'identity', seeded(2));
    const adam = new Adam(net, 0.02);
    const before = loss(net, X, y);
    for (let t = 0; t < 300; t++) adam.step(net, gradients(net, X, y));
    expect(loss(net, X, y)).toBeLessThan(before / 20);
  });
});
