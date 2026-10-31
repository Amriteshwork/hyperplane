import {
  BASE,
  LEARNED_WEIGHTS,
  TRUE_QUALITY,
  bestPolicy,
  expected,
  fitReward,
  rewards,
} from './rlhf';

const scores = rewards(LEARNED_WEIGHTS);
const quality = (beta: number) => expected(bestPolicy(BASE, scores, beta), TRUE_QUALITY);

describe('RLHF maths', () => {
  it('matches the numbers the NumPy page prints', () => {
    expect(expected(BASE, TRUE_QUALITY)).toBeCloseTo(1.29, 2);
    expect(quality(5)).toBeCloseTo(1.63, 1);
    expect(quality(3)).toBeCloseTo(1.64, 1);
    expect(quality(0.2)).toBeCloseTo(-3, 1);
  });

  it('a huge β stays at the base model and sums to 1', () => {
    const p = bestPolicy(BASE, scores, 1e6);
    p.forEach((v, i) => expect(v).toBeCloseTo(BASE[i], 3));
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
  });

  it('real quality rises, peaks, then collapses while reward keeps rising', () => {
    expect(quality(3)).toBeGreaterThan(expected(BASE, TRUE_QUALITY));
    expect(quality(0.2)).toBeLessThan(quality(3));
    const reward = (beta: number) => expected(bestPolicy(BASE, scores, beta), scores);
    expect(reward(0.2)).toBeGreaterThan(reward(3));
  });

  it('learns from votes: preferring polite raises the polite weight', () => {
    expect(fitReward([])).toEqual([0, 0, 0]);
    const w = fitReward([
      [3, 2],
      [2, 0],
    ]);
    expect(w[1]).toBeGreaterThan(0);
    expect(w[0]).toBeGreaterThan(0);
  });
});
