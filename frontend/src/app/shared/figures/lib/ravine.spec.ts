import { adam, gradientDescent, lossAt, momentum } from './ravine';

describe('ravine optimisers', () => {
  it('tell the story the caption tells: plain GD is still crawling after 100 steps', () => {
    const final = (points: [number, number][]) => lossAt(points[points.length - 1]);
    const gd = final(gradientDescent(100));
    expect(final(momentum(100))).toBeLessThan(gd / 10);
    expect(final(adam(100))).toBeLessThan(gd / 10);
  });

  it('keeps plain GD stable on the steep axis', () => {
    expect(Math.max(...gradientDescent(100).map(([a]) => Math.abs(a)))).toBeLessThanOrEqual(0.9);
  });
});
