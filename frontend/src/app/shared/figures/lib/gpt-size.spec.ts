import { PRESETS, parameterCounts } from './gpt-size';

describe('parameterCounts', () => {
  it('matches GPT-2 small exactly', () => {
    expect(parameterCounts(PRESETS[0].shape).total).toBe(124_439_808);
  });

  it('gives GPT-2 XL about 1.5B and GPT-3 about 175B', () => {
    expect(parameterCounts(PRESETS[1].shape).total / 1e9).toBeCloseTo(1.56, 2);
    expect(parameterCounts(PRESETS[2].shape).total / 1e9).toBeCloseTo(175, -1);
  });

  it('spends about two thirds of a big model on the MLPs', () => {
    const big = parameterCounts(PRESETS[2].shape);
    expect(big.mlp / big.total).toBeGreaterThan(0.6);
  });
});
