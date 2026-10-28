import { SENTENCE, attentionWeights, softmax } from './attention';

const queries = SENTENCE.map((t) => t.query);
const keys = SENTENCE.map((t) => t.key);
const word = (i: number) => SENTENCE[i].word;
const argmax = (row: number[]) => row.indexOf(Math.max(...row));

describe('attention', () => {
  it('softmax sums to 1 and gives −Infinity nothing', () => {
    const p = softmax([1000, 1001, -Infinity]);
    expect(p[0] + p[1]).toBeCloseTo(1, 12);
    expect(p[2]).toBe(0);
  });

  it('every row of weights sums to 1', () => {
    for (const row of attentionWeights(queries, keys, 3, false))
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  });

  it('the causal mask zeroes every later token', () => {
    const weights = attentionWeights(queries, keys, 3, true);
    weights.forEach((row, t) => row.slice(t + 1).forEach((w) => expect(w).toBe(0)));
    expect(weights[0][0]).toBe(1);
  });

  it('"it" finds "animal", and sharpness 0 is a plain average', () => {
    const it = SENTENCE.findIndex((t) => t.word === 'it');
    expect(word(argmax(attentionWeights(queries, keys, 3, true)[it]))).toBe('animal');
    const uniform = attentionWeights(queries, keys, 0, true)[it];
    uniform.slice(0, it + 1).forEach((w) => expect(w).toBeCloseTo(1 / (it + 1), 12));
    expect(Math.max(...attentionWeights(queries, keys, 20, true)[it])).toBeGreaterThan(0.99);
  });
});
