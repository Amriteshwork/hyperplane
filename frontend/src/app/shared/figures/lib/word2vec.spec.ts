import { Tiktoken } from 'js-tiktoken/lite';
import gpt2 from 'js-tiktoken/ranks/gpt2';

import { seeded } from './random';
import {
  ANALOGIES,
  TOKEN_ROWS,
  TOKEN_SENTENCES,
  WORD_LIST,
  createSkipGram,
  groupSeparation,
  nearestWords,
  pca2d,
  skipGramStep,
  solveAnalogy,
} from './word2vec';

describe('word embeddings maths', () => {
  it('live skip-gram separates the groups it was never told about', () => {
    const rng = seeded(3);
    const model = createSkipGram(rng);
    const before = groupSeparation(model);
    for (let i = 0; i < 60_000; i++) skipGramStep(model, rng);
    expect(groupSeparation(model)).toBeGreaterThan(before + 0.3);
    expect(groupSeparation(model)).toBeGreaterThan(0.5);
  });

  it('every analogy the figure offers is solved by real GloVe vectors', () => {
    const answers = ANALOGIES.map(([a, b, c]) => solveAnalogy(a, b, c)[0].word);
    expect(answers).toContain('queen');
    expect(answers).toContain('tokyo');
    expect(answers).toContain('actress');
    expect(answers).toContain('worse');
    expect(answers[0]).toBe('queen');
  });

  it('nearest words make sense on real vectors', () => {
    const near = nearestWords('monday').map((n) => n.word);
    expect(near).toContain('tuesday');
    expect(nearestWords('cat', 5).map((n) => n.word)).toContain('dog');
    expect(nearestWords('piano', 3).map((n) => n.word)).toContain('violin');
  });

  it('PCA gives each word two finite coordinates', () => {
    const xy = pca2d(WORD_LIST);
    for (const w of WORD_LIST) expect(xy[w].every(Number.isFinite)).toBe(true);
    const spread = (i: 0 | 1) => {
      const vs = WORD_LIST.map((w) => xy[w][i]);
      return Math.max(...vs) - Math.min(...vs);
    };
    expect(spread(0)).toBeGreaterThan(1);
    expect(spread(1)).toBeGreaterThan(1);
  });

  it('the browser tokenizer agrees with the ids Python tiktoken wrote', () => {
    const encoding = new Tiktoken(gpt2);
    for (const { text, ids } of TOKEN_SENTENCES) expect(encoding.encode(text)).toEqual([...ids]);
    const known = new Set(TOKEN_ROWS.map((t) => t.id));
    for (const { ids } of TOKEN_SENTENCES) for (const id of ids) expect(known.has(id)).toBe(true);
  });
});
