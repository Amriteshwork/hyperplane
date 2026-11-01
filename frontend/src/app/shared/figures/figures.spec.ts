import { Type } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AttentionLookbackFigure } from './attention-lookback-figure';
import { BaseVsChatFigure } from './base-vs-chat-figure';
import { BeamSearchTreeFigure } from './beam-search-tree-figure';
import { BottleneckCurveFigure } from './bottleneck-curve-figure';
import { ChatLabelsFigure } from './chat-labels-figure';
import { CrossAttentionFigure } from './cross-attention-figure';
import { DropoutMaskFigure } from './dropout-mask-figure';
import { GradientClipFigure } from './gradient-clip-figure';
import { KvCacheFigure } from './kv-cache-figure';
import { LrWarmupFigure } from './lr-warmup-figure';
import { RewardVotesFigure } from './reward-votes-figure';
import { RlhfLeashFigure } from './rlhf-leash-figure';
import { SamplingExplorerFigure } from './sampling-explorer-figure';
import { Seq2seqStepsFigure } from './seq2seq-steps-figure';
import { SkipgramTrainerFigure } from './skipgram-trainer-figure';
import { TemperatureTradeoffFigure } from './temperature-tradeoff-figure';
import { TokenToVectorFigure } from './token-to-vector-figure';
import { TokenizerPlaygroundFigure } from './tokenizer-playground-figure';
import { WordMapFigure } from './word-map-figure';
import { AttentionHeatmapFigure } from './attention-heatmap-figure';
import { BackpropStepsFigure } from './backprop-steps-figure';
import { BertHeadsFigure } from './bert-heads-figure';
import { BertInputFigure } from './bert-input-figure';
import { BoundaryPlaygroundFigure } from './boundary-playground-figure';
import { BpttUnrolledFigure } from './bptt-unrolled-figure';
import { EarlyStoppingFigure } from './early-stopping-figure';
import { BayesBaseRateFigure } from './bayes-base-rate-figure';
import { DoubleCountingFigure } from './double-counting-figure';
import { LikelihoodExplorerFigure } from './likelihood-explorer-figure';
import { NaiveBayesPlaygroundFigure } from './naive-bayes-playground-figure';
import { DistanceBallsFigure } from './distance-balls-figure';
import { DistanceConcentrationFigure } from './distance-concentration-figure';
import { DEFAULT_FIGURES } from './figure-registry';
import { FillInTheBlankFigure } from './fill-in-the-blank-figure';
import { KnnPlaygroundFigure } from './knn-playground-figure';
import { KnnRegressionFigure } from './knn-regression-figure';
import { NetworkShapeFigure } from './network-shape-figure';
import { PositionalEncodingExplorerFigure } from './positional-encoding-explorer-figure';
import { ReluSumFigure } from './relu-sum-figure';
import { TransformerBlockFigure } from './transformer-block-figure';
import { XorNetworkFigure } from './xor-network-figure';

async function mount<T>(type: Type<T>): Promise<[ComponentFixture<T>, HTMLElement]> {
  const fixture = TestBed.createComponent(type);
  await fixture.whenStable();
  return [fixture, fixture.nativeElement];
}

function button(el: HTMLElement, label: string): HTMLButtonElement {
  const found = [...el.querySelectorAll('button')].find((b) => b.textContent?.includes(label));
  if (!found) throw new Error(`no button "${label}"`);
  return found;
}

function slide(el: HTMLElement, label: string, value: number): void {
  const input = [...el.querySelectorAll('label')]
    .find((l) => l.textContent?.includes(label))
    ?.querySelector('input');
  if (!input) throw new Error(`no slider "${label}"`);
  input.value = String(value);
  input.dispatchEvent(new Event('input'));
}

/** Animation frames run only when the test says so. */
function manualFrames(): () => void {
  // Angular's zoneless scheduler also uses frames, so cancel by id, not wholesale.
  let queue = new Map<number, FrameRequestCallback>();
  let id = 0;
  vi.spyOn(globalThis, 'requestAnimationFrame').mockImplementation((cb) => {
    queue.set(++id, cb);
    return id;
  });
  vi.spyOn(globalThis, 'cancelAnimationFrame').mockImplementation((handle) => queue.delete(handle));
  return () => {
    const now = queue;
    queue = new Map();
    now.forEach((cb) => cb(0));
  };
}

describe('figures', () => {
  afterEach(() => vi.restoreAllMocks());

  it('registers a loader for every figure that resolves to a component', async () => {
    for (const load of Object.values(DEFAULT_FIGURES))
      expect(typeof (await load())).toBe('function');
  });

  const reading = (el: HTMLElement, id: string) =>
    parseFloat(el.querySelector(`[data-testid="${id}"]`)?.textContent ?? '');

  it('bayes base rate shows false positives swamping a rare condition', async () => {
    const [fixture, el] = await mount(BayesBaseRateFigure);
    expect(reading(el, 'posterior')).toBeCloseTo(15.4, 0); // 1% sick, 90% hit, 5% false alarm
    slide(el, 'How common', 50);
    await fixture.whenStable();
    expect(reading(el, 'posterior')).toBeCloseTo(94.7, 0);
    expect(el.querySelector('svg')?.getAttribute('aria-label')).toContain('94.7');
  });

  it('likelihood explorer rewards the sample mean and spread', async () => {
    const [fixture, el] = await mount(LikelihoodExplorerFigure);
    const start = reading(el, 'log-likelihood');
    button(el, 'Jump to the best fit').click();
    await fixture.whenStable();
    const best = reading(el, 'log-likelihood');
    expect(best).toBeGreaterThan(start);
    expect(best).toBeCloseTo(reading(el, 'best-log-likelihood'), 1);
    slide(el, 'Spread', 0.3);
    await fixture.whenStable();
    expect(reading(el, 'log-likelihood')).toBeLessThan(best);
  });

  it("double counting raises the model's certainty but not the truth", async () => {
    const [fixture, el] = await mount(DoubleCountingFigure);
    expect(reading(el, 'naive')).toBeCloseTo(reading(el, 'truth'), 5);
    const truth = reading(el, 'truth');
    slide(el, 'Copies', 10);
    await fixture.whenStable();
    expect(reading(el, 'truth')).toBe(truth);
    expect(reading(el, 'naive')).toBeGreaterThan(truth + 10);
  });

  it('naive bayes playground grows overconfident as features correlate', async () => {
    const [fixture, el] = await mount(NaiveBayesPlaygroundFigure);
    const gap = () => reading(el, 'confidence') - reading(el, 'accuracy');
    expect(Math.abs(gap())).toBeLessThan(5);
    slide(el, 'Correlation', 0.95);
    await fixture.whenStable();
    expect(gap()).toBeGreaterThan(5);
    const plot = el.querySelector('svg');
    const before = reading(el, 'query-p');
    plot?.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight' }));
    await fixture.whenStable();
    expect(reading(el, 'query-p')).not.toBe(before);
  });

  it('knn playground memorises at k = 1, generalises at a larger k', async () => {
    const [fixture, el] = await mount(KnnPlaygroundFigure);
    const score = (id: string) =>
      parseFloat(el.querySelector(`[data-testid="${id}"]`)?.textContent ?? '');
    expect(score('train-score')).toBe(100);
    const memorising = score('test-score');
    slide(el, 'Neighbours', 21);
    await fixture.whenStable();
    expect(score('train-score')).toBeLessThan(100);
    expect(score('test-score')).toBeGreaterThan(memorising);
    expect(el.querySelectorAll('line')).toHaveLength(21); // one per neighbour
    button(el, 'Circle').click();
    await fixture.whenStable();
    const plot = el.querySelector('svg');
    expect(plot?.getAttribute('aria-label')).toContain('circle');
    const dot = () => [...el.querySelectorAll('circle')].at(-1)?.getAttribute('cx');
    const before = dot();
    plot?.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft' }));
    await fixture.whenStable();
    expect(dot()).not.toBe(before);
  });

  it('knn regression is closest to the curve at a moderate k', async () => {
    const [fixture, el] = await mount(KnnRegressionFigure);
    const mse = () => parseFloat(el.querySelector('[data-testid="mse"]')?.textContent ?? '');
    const atOne = mse();
    slide(el, 'Neighbours', 8);
    await fixture.whenStable();
    const atEight = mse();
    slide(el, 'Neighbours', 60);
    await fixture.whenStable();
    expect(atEight).toBeLessThan(atOne);
    expect(atEight).toBeLessThan(mse());
  });

  it('distance balls change shape with the metric and collapse under a stretch', async () => {
    const [fixture, el] = await mount(DistanceBallsFigure);
    const shape = () => el.querySelector('path')?.getAttribute('d') ?? '';
    const circle = shape();
    expect(circle).toContain('a');
    button(el, 'Manhattan').click();
    await fixture.whenStable();
    expect(shape()).not.toContain('a');
    expect(shape()).not.toBe(circle);
    const before = el.querySelector('[data-testid="nearest-distance"]')?.textContent;
    slide(el, 'Stretch', 20);
    await fixture.whenStable();
    expect(el.querySelector('[data-testid="nearest-distance"]')?.textContent).not.toBe(before);
    expect(el.querySelector('[data-testid="agreement"]')?.textContent).toMatch(/of 3/);
  });

  it('distance concentration pulls nearest and farthest together', async () => {
    const [fixture, el] = await mount(DistanceConcentrationFigure);
    const read = (id: string) =>
      parseFloat(el.querySelector(`[data-testid="${id}"]`)?.textContent?.replace(',', '') ?? '');
    const contrast = read('contrast');
    const edge = read('edge');
    slide(el, 'Features', 10);
    await fixture.whenStable();
    expect(el.querySelector('svg')?.getAttribute('aria-label')).toContain('1000 features');
    expect(read('contrast')).toBeLessThan(1.3);
    expect(contrast).toBeGreaterThan(read('contrast'));
    expect(read('edge')).toBeGreaterThan(edge);
  });

  it('tokenizer playground shows real tokens and refuses nothing', async () => {
    const [fixture, el] = await mount(TokenizerPlaygroundFigure);
    await vi.waitFor(async () => {
      await fixture.whenStable();
      expect(el.querySelector('[data-testid="tokens"]')).not.toBeNull();
    });
    button(el, 'Numbers').click();
    await fixture.whenStable();
    expect(el.textContent).toContain('tokens');
    button(el, 'Show ids').click();
    await fixture.whenStable();
    const chips = [...el.querySelectorAll('[data-testid="tokens"] span')].map((c) => c.textContent);
    expect(chips.every((c) => /^\d+$/.test(c?.trim() ?? ''))).toBe(true);
  });

  it('token to vector shows one row for "bank" in both sentences', async () => {
    const [fixture, el] = await mount(TokenToVectorFigure);
    const row = () => el.querySelector('p.mt-3')?.textContent ?? '';
    const river = row();
    button(el, 'money').click();
    await fixture.whenStable();
    expect(row()).toBe(river);
  });

  it('sampling explorer: low temperature only draws the favourite', async () => {
    const [fixture, el] = await mount(SamplingExplorerFigure);
    slide(el, 'Temperature', 0.05);
    await fixture.whenStable();
    button(el, 'Draw 20').click();
    await fixture.whenStable();
    expect(el.querySelector('p.font-mono')?.textContent?.trim()).toBe(
      Array(20).fill('mat').join(' '),
    );
  });

  it('temperature trade-off reports the measured runs', async () => {
    const [fixture, el] = await mount(TemperatureTradeoffFigure);
    button(el, 'τ = 2.5').click();
    await fixture.whenStable();
    expect(el.querySelector('svg')?.getAttribute('aria-label')).toContain('0% of sentences');
  });

  it('beam search finds the better sentence only at width 2', async () => {
    const [fixture, el] = await mount(BeamSearchTreeFigure);
    const finish = async () => {
      button(el, 'Next step').click();
      await fixture.whenStable();
      button(el, 'Next step').click();
      await fixture.whenStable();
      return el.querySelector('svg')?.getAttribute('aria-label') ?? '';
    };
    expect(await finish()).toContain('Best: "the cat" at 0.20');
    button(el, 'Width 2').click();
    await fixture.whenStable();
    expect(await finish()).toContain('Best: "a cat" at 0.36');
  });

  it('the KV cache turns a whole grid into one row', async () => {
    const [fixture, el] = await mount(KvCacheFigure);
    expect(el.querySelector('svg')?.getAttribute('aria-label')).toContain('computes 64');
    button(el, 'With the KV cache').click();
    await fixture.whenStable();
    expect(el.querySelector('svg')?.getAttribute('aria-label')).toContain('computes 8');
  });

  it('bottleneck curve shows the gap at 12 words', async () => {
    const [fixture, el] = await mount(BottleneckCurveFigure);
    slide(el, 'List length', 8);
    await fixture.whenStable();
    expect(el.querySelector('svg')?.getAttribute('aria-label')).toBe(
      'At 12 words: one vector 0%, with attention 70%.',
    );
  });

  it('look-back follows the diagonal', async () => {
    const [fixture, el] = await mount(AttentionLookbackFigure);
    button(el, 'reis').click();
    await fixture.whenStable();
    expect(el.querySelector('[role="img"]')?.getAttribute('aria-label')).toContain(
      '"reis" reads "rice" most, at 63%',
    );
  });

  it('cross-attention shows which English a German word reads', async () => {
    const [fixture, el] = await mount(CrossAttentionFigure);
    button(el, 'pflaumen').click();
    await fixture.whenStable();
    expect(el.querySelector('table')?.getAttribute('aria-label')).toContain('reads "plums"');
  });

  it('translation steps: encoder first, then three steps for each word', async () => {
    const [fixture, el] = await mount(Seq2seqStepsFigure);
    expect(el.textContent).toContain('Step 1 of 28');
    for (let i = 0; i < 3; i++) {
      button(el, 'Next step').click();
      await fixture.whenStable();
    }
    expect(el.textContent).toContain('1 of 9 words');
  });

  it('skip-gram trains live and separates the groups', async () => {
    const run = manualFrames();
    const [fixture, el] = await mount(SkipgramTrainerFigure);
    const label = () => el.querySelector('svg')?.getAttribute('aria-label') ?? '';
    const separation = () => Number(label().split('separation ')[1]);
    const before = separation();
    button(el, 'Train').click();
    for (let i = 0; i < 200; i++) run();
    await fixture.whenStable();
    expect(separation()).toBeGreaterThan(before + 0.2);
    fixture.destroy();
  });

  it('word map solves king − man + woman on real vectors', async () => {
    const [, el] = await mount(WordMapFigure);
    expect(el.querySelector('svg')?.getAttribute('aria-label')).toContain('is closest to queen');
  });

  it('warmup slider draws a schedule and lists the measured runs', async () => {
    const [fixture, el] = await mount(LrWarmupFigure);
    slide(el, 'Warmup steps', 0);
    await fixture.whenStable();
    expect(el.textContent).toContain('off (constant)');
    expect(el.textContent).toContain('1.58');
    expect(el.textContent).toContain('0.31');
  });

  it('clipping only touches gradients over the limit', async () => {
    const [fixture, el] = await mount(GradientClipFigure);
    expect(el.querySelector('svg')?.getAttribute('aria-label')).toContain('becomes length 3.0');
    slide(el, 'Gradient length', 2);
    await fixture.whenStable();
    expect(el.querySelector('svg')?.getAttribute('aria-label')).toContain('becomes length 2.0');
    expect(el.textContent).toContain("it's not touched");
  });

  it('dropout: rate 0 silences nothing, and each batch differs', async () => {
    const [fixture, el] = await mount(DropoutMaskFigure);
    const label = () => el.querySelector('svg')?.getAttribute('aria-label') ?? '';
    slide(el, 'Dropout rate', 0);
    await fixture.whenStable();
    expect(label()).toContain('0 of 16');
  });

  it('chat labels: only the assistant counts', async () => {
    const [fixture, el] = await mount(ChatLabelsFigure);
    expect(el.textContent).toContain('This counts');
    button(el, 'what').click();
    await fixture.whenStable();
    expect(el.textContent).toContain('costs nothing');
  });

  it('base vs chat shows the measured rates', async () => {
    const [fixture, el] = await mount(BaseVsChatFigure);
    expect(el.textContent).toContain('92%');
    button(el, 'Base model').click();
    await fixture.whenStable();
    expect(el.textContent).toContain('never stops');
  });

  it('votes teach the reward model, and it scores a reply no one rated', async () => {
    const [fixture, el] = await mount(RewardVotesFigure);
    const unrated = 'JAM! thanks thanks for asking, asking, asking!';
    const score = () =>
      [...el.querySelectorAll('li')].find((l) => l.textContent?.includes(unrated))?.textContent ??
      '';
    expect(score()).toContain('+0.0');
    for (let i = 0; i < 6; i++) {
      el.querySelector<HTMLButtonElement>('[role="group"] button')?.click();
      await fixture.whenStable();
    }
    expect(el.textContent).toContain('6 votes');
    expect(score()).not.toContain('+0.0');
  });

  it('a loose leash scores higher and is worse', async () => {
    const [fixture, el] = await mount(RlhfLeashFigure);
    const reward = () => Number(/reward score: ([\d.]+)/.exec(el.textContent ?? '')?.[1]);
    const tight = (slide(el, 'Leash', 50), await fixture.whenStable(), reward());
    slide(el, 'Leash', 100);
    await fixture.whenStable();
    expect(reward()).toBeGreaterThan(tight);
    expect(el.textContent).toContain('Worse than the base model');
  });

  it('XOR network outputs 1 only for mixed inputs', async () => {
    const [fixture, el] = await mount(XorNetworkFigure);
    const output = () => el.querySelector('svg')?.getAttribute('aria-label') ?? '';
    for (const [corner, want] of [
      ['(0, 1)', '1.00'],
      ['(1, 1)', '0.00'],
      ['(0, 0)', '0.00'],
    ]) {
      button(el, corner).click();
      await fixture.whenStable();
      expect(output()).toContain(`XOR = ${want}`);
    }
  });

  it('network shape counts weights and biases per layer', async () => {
    const [fixture, el] = await mount(NetworkShapeFigure);
    slide(el, 'Input features', 10);
    slide(el, 'Neurons per hidden layer', 100);
    await fixture.whenStable();
    expect(el.textContent).toMatch(/Total\s*1,201 weights and biases/);
  });

  it('more ReLU neurons fit the curve better', async () => {
    const [fixture, el] = await mount(ReluSumFigure);
    const error = () =>
      Number(
        /error (\d+\.\d+)/.exec(el.querySelector('svg')?.getAttribute('aria-label') ?? '')?.[1],
      );
    slide(el, 'Hidden ReLU neurons', 1);
    await fixture.whenStable();
    const one = error();
    slide(el, 'Hidden ReLU neurons', 12);
    await fixture.whenStable();
    expect(error()).toBeLessThan(one / 10);
  });

  it('backprop stepper walks to an update that lowers the loss', async () => {
    const [fixture, el] = await mount(BackpropStepsFigure);
    const next = button(el, 'Next step');
    while (!next.disabled) {
      next.click();
      await fixture.whenStable();
    }
    const text = el.querySelector('[aria-live]')?.textContent ?? '';
    const [before, after] =
      /drops from (\d+\.\d+) to (\d+\.\d+)/.exec(text)?.slice(1).map(Number) ?? [];
    expect(after).toBeLessThan(before);
    button(el, 'Start over').click();
    await fixture.whenStable();
    expect(el.textContent).toContain('Step 1 of 11');
  });

  it('boundary playground trains when asked and resets on a new dataset', async () => {
    const frame = manualFrames();
    const [fixture, el] = await mount(BoundaryPlaygroundFigure);
    const epoch = () => el.querySelector('dl dd')?.textContent?.trim();
    button(el, 'Train').click();
    for (let i = 0; i < 4; i++) frame();
    await fixture.whenStable();
    expect(epoch()).toBe('20');
    expect(button(el, 'Pause')).toBeTruthy();
    button(el, 'XOR').click();
    await fixture.whenStable();
    expect(epoch()).toBe('0');
  });

  it('early stopping trains live and can jump to the best epoch', async () => {
    const frame = manualFrames();
    const [fixture, el] = await mount(EarlyStoppingFigure);
    for (let i = 0; i < 100; i++) frame();
    await fixture.whenStable();
    expect(el.textContent).not.toContain('Training…');
    const best = /lowest \(([\d.]+)\) at epoch (\d+)/.exec(
      el.querySelectorAll('svg')[1].getAttribute('aria-label') ?? '',
    );
    expect(Number(best?.[2])).toBeLessThan(1000); // validation loss turned back up: it overfits
    button(el, 'Jump to the best epoch').click();
    await fixture.whenStable();
    expect(el.textContent).toMatch(
      new RegExp(`fit at epoch ${Math.round(Number(best?.[2]) / 5) * 5}\\b`),
    );
  });

  it('attention heatmap: "it" finds "animal", and the mask hides the future', async () => {
    const [fixture, el] = await mount(AttentionHeatmapFigure);
    const label = () => el.querySelector('svg')?.getAttribute('aria-label') ?? '';
    expect(label()).toContain('"it" puts');
    expect(label()).toContain('on "animal"');
    expect(el.textContent).not.toContain('✕');
    button(el, 'Causal mask off').click();
    await fixture.whenStable();
    expect(el.textContent).toContain('✕');
    button(el, 'the').click();
    slide(el, 'Sharpness', 0);
    await fixture.whenStable();
    expect(label()).toContain('"the" puts 100% of its attention on "the"'); // masked: only itself
  });

  it('positional encoding: similarity peaks at the chosen position', async () => {
    const [fixture, el] = await mount(PositionalEncodingExplorerFigure);
    slide(el, 'Position t', 30);
    await fixture.whenStable();
    const curve = el.querySelectorAll('svg')[2].getAttribute('aria-label') ?? '';
    expect(curve).toContain('Similarity of position 30');
    expect(curve).toContain('30 vs 33');
  });

  it('transformer block: walks the stages and counts GPT-2 exactly', async () => {
    const [fixture, el] = await mount(TransformerBlockFigure);
    expect(el.textContent).toContain('124,439,808');
    button(el, 'MLP').click();
    await fixture.whenStable();
    expect(el.querySelector('[aria-live]')?.textContent).toContain('768 → 3,072');
    button(el, 'GPT-3').click();
    await fixture.whenStable();
    expect(el.querySelector('[aria-live]')?.textContent).toContain('12,288 → 49,152');
  });

  it('BPTT unrolled: the gradient walks back and vanishes or explodes with w', async () => {
    const [fixture, el] = await mount(BpttUnrolledFigure);
    const live = () => el.querySelector('[aria-live]')?.textContent ?? '';
    const bars = () => el.querySelectorAll('svg rect').length;
    expect(bars()).toBe(1);
    button(el, 'Step back').click();
    await fixture.whenStable();
    expect(bars()).toBe(2);
    expect(live()).toContain('δ₉ is');

    button(el, 'No activation').click();
    slide(el, 'Recurrent weight w', 0.5);
    button(el, 'Show all').click();
    await fixture.whenStable();
    expect(bars()).toBe(10);
    expect(live()).toContain('δ₁ is 2.0e−3 × δ₁₀'); // 0.5⁹
    expect(live()).toContain('barely shapes');

    slide(el, 'Recurrent weight w', 1.5);
    await fixture.whenStable();
    expect(live()).toContain('38.44 × δ₁₀'); // 1.5⁹
    expect(live()).toContain('exploding');
  });

  it('fill in the blank: BERT copies the first name from the right; GPT can only guess', async () => {
    const [fixture, el] = await mount(FillInTheBlankFigure);
    const live = () => el.querySelector('[aria-live]')?.textContent ?? '';
    expect(live()).toContain('copies “zoe” from position 5');
    button(el, 'Left to right').click();
    await fixture.whenStable();
    expect(live()).toContain('out of sight');
    expect(live()).toContain('1 in 8');
    button(el, 'Both ways').click();
    button(el, 'zoe').click(); // hide the copy too: now nobody can know
    await fixture.whenStable();
    expect(live()).toContain('twin is hidden');
  });

  it('BERT input: packs both sentences and explains the picked token', async () => {
    const [fixture, el] = await mount(BertInputFigure);
    const tokens = [...el.querySelectorAll('tbody tr:first-child button')].map((b) =>
      b.textContent?.trim(),
    );
    expect(tokens[0]).toBe('[CLS]');
    expect(tokens).toContain('##ing');
    expect(tokens.filter((t) => t === '[SEP]')).toHaveLength(2);
    button(el, '##ing').click();
    await fixture.whenStable();
    expect(el.querySelector('[aria-live]')?.textContent).toContain('word piece');
  });

  it('BERT heads: classification reads [CLS], tagging reads every word', async () => {
    const [fixture, el] = await mount(BertHeadsFigure);
    const live = () => el.querySelector('[aria-live]')?.textContent ?? '';
    expect(live()).toContain('1,536');
    button(el, 'Names & places').click();
    await fixture.whenStable();
    expect(el.textContent).toContain('PLACE');
    expect(live()).toContain('6,912');
    button(el, 'Question answering').click();
    await fixture.whenStable();
    expect(el.querySelector('[role="img"]')?.getAttribute('aria-label')).toContain('on: start');
  });
});
