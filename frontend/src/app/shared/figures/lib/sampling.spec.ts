import { attentionWork, beamSteps, drawToken, nextTokenProbs } from './sampling';

const logits = [0.5, 0.3, 0.15, 0.04, 0.01].map(Math.log);
const plain = { temperature: 1, topK: null, topP: null };

describe('text generation maths', () => {
  it('matches the NumPy page: τ = 1 changes nothing, τ → 0 is greedy', () => {
    nextTokenProbs(logits, plain).forEach((p, i) =>
      expect(p).toBeCloseTo([0.5, 0.3, 0.15, 0.04, 0.01][i]),
    );
    expect(nextTokenProbs(logits, { ...plain, temperature: 0.05 })[0]).toBeGreaterThan(0.99);
    expect(Math.max(...nextTokenProbs(logits, { ...plain, temperature: 1000 }))).toBeLessThan(0.21);
  });

  it('top-k keeps k tokens, top-p keeps as many as it takes', () => {
    const count = (p: number[]) => p.filter((v) => v > 0).length;
    expect(count(nextTokenProbs(logits, { ...plain, topK: 2 }))).toBe(2);
    expect(count(nextTokenProbs(logits, { ...plain, topP: 0.9 }))).toBe(3);
    const sure = [0.97, 0.01, 0.01, 0.005, 0.005].map(Math.log);
    expect(count(nextTokenProbs(sure, { ...plain, topP: 0.9 }))).toBe(1);
  });

  it('what survives a cut still sums to 1', () => {
    const p = nextTokenProbs(logits, { temperature: 0.7, topK: 3, topP: 0.8 });
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });

  it('draws land on a token with probability above 0', () => {
    const p = [0.5, 0, 0.5];
    expect(drawToken(p, 0.1)).toBe(0);
    expect(drawToken(p, 0.9)).toBe(2);
    expect(drawToken(p, 0.999999999)).toBe(2);
  });

  it('a beam of 2 finds the path greedy misses, as in the NumPy page', () => {
    const best = (width: number) => beamSteps(width).at(-1)![0];
    expect(best(1).words).toEqual(['the', 'cat']);
    expect(best(1).probability).toBeCloseTo(0.2);
    expect(best(2).words).toEqual(['a', 'cat']);
    expect(best(2).probability).toBeCloseTo(0.36);
  });

  it('counts the attention work the PyTorch page prints', () => {
    expect(attentionWork(256)).toEqual({ rerun: 5_625_216, cached: 32_896 });
  });
});
