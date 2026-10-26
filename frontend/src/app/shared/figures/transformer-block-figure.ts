import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { GptCounts, GptShape, PRESETS, parameterCounts } from './lib/gpt-size';

interface Stage {
  id: string;
  label: string;
  inBlock: boolean;
  /** What happens here, in plain words. */
  text: (s: GptShape, heads: number) => string;
  /** The shape of what comes out, e.g. "T × 768". */
  shape: (s: GptShape) => string;
  /** Parameters this stage adds, over the whole model. */
  params: (s: GptShape, c: GptCounts) => number;
}

const fmt = (n: number) => n.toLocaleString('en-US');

const STAGES: readonly Stage[] = [
  {
    id: 'ids',
    label: 'Token ids',
    inBlock: false,
    text: (s) => `The text, tokenized: T integers, each one of ${fmt(s.vocab)} possible tokens.`,
    shape: () => 'T',
    params: () => 0,
  },
  {
    id: 'embed',
    label: 'Embed + position',
    inBlock: false,
    text: (s) =>
      `Each id picks its row of the ${fmt(s.vocab)} × ${s.d} embedding table, and the row for its ` +
      `position (one of ${fmt(s.context)}) is added. From here on, every token is ${s.d} numbers.`,
    shape: (s) => `T × ${s.d}`,
    params: (_, c) => c.tokenEmbedding + c.positions,
  },
  {
    id: 'ln1',
    label: 'LayerNorm',
    inBlock: true,
    text: () =>
      'Standardise each token’s vector to mean 0 and variance 1, then scale by γ and shift by β. ' +
      'The residual stream itself is left untouched.',
    shape: (s) => `T × ${s.d}`,
    params: (s) => s.layers * 2 * s.d,
  },
  {
    id: 'attn',
    label: 'Multi-head attention',
    inBlock: true,
    text: (s, heads) =>
      `Tokens talk. ${heads} heads of ${s.d / heads} dimensions each compute their own T × T ` +
      'attention pattern, masked so nobody sees the future, then W_O mixes their findings.',
    shape: (s) => `T × ${s.d}`,
    params: (_, c) => c.attention,
  },
  {
    id: 'add1',
    label: 'Add (residual)',
    inBlock: true,
    text: () =>
      'Add attention’s output to the stream: an update, not a replacement. The gradient’s express lane.',
    shape: (s) => `T × ${s.d}`,
    params: () => 0,
  },
  {
    id: 'ln2',
    label: 'LayerNorm',
    inBlock: true,
    text: () => 'Standardise again, so the MLP also reads a well-scaled input.',
    shape: (s) => `T × ${s.d}`,
    params: (s) => s.layers * 2 * s.d,
  },
  {
    id: 'mlp',
    label: 'MLP',
    inBlock: true,
    text: (s) =>
      `Each token thinks, alone: ${fmt(s.d)} → ${fmt(4 * s.d)} numbers, a GELU, then back to ${fmt(s.d)}. ` +
      'The same small network at every position.',
    shape: (s) => `T × ${s.d} (T × ${fmt(4 * s.d)} inside)`,
    params: (_, c) => c.mlp,
  },
  {
    id: 'add2',
    label: 'Add (residual)',
    inBlock: true,
    text: (s) =>
      `Add the MLP’s output to the stream. Then the next block starts: ${s.layers} blocks in all.`,
    shape: (s) => `T × ${s.d}`,
    params: () => 0,
  },
  {
    id: 'lnf',
    label: 'Final LayerNorm',
    inBlock: false,
    text: () => 'One last standardisation of the stream before the prediction.',
    shape: (s) => `T × ${s.d}`,
    params: (s) => 2 * s.d,
  },
  {
    id: 'unembed',
    label: 'Unembed',
    inBlock: false,
    text: (s) =>
      `Multiply by the embedding table, transposed: one score for each of the ${fmt(s.vocab)} ` +
      'tokens, at every position. GPT-2 reuses the embedding matrix here, so it adds no parameters.',
    shape: (s) => `T × ${fmt(s.vocab)}`,
    params: () => 0,
  },
  {
    id: 'softmax',
    label: 'Softmax → next token',
    inBlock: false,
    text: () =>
      'Softmax turns the last position’s scores into probabilities. Sample one, append it to the ' +
      'ids, and run the whole thing again: that is generation.',
    shape: (s) => `${fmt(s.vocab)} probabilities`,
    params: () => 0,
  },
];

const SHARES = [
  { key: 'embeddings', label: 'embeddings', color: 'bg-fig-3' },
  { key: 'attention', label: 'attention', color: 'bg-fig-1' },
  { key: 'mlp', label: 'MLPs', color: 'bg-fig-2' },
  { key: 'norms', label: 'layer norms', color: 'bg-fig-4' },
] as const;

/** A GPT, stage by stage: what each does, the shape it outputs, and what it costs at real sizes. */
@Component({
  selector: 'app-transformer-block-figure',
  imports: [DecimalPipe, FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Inside a GPT, stage by stage"
      hint="Click a stage to see what it does, what shape comes out, and how many parameters it costs."
    >
      <div class="flex flex-wrap gap-2" role="group" aria-label="Model size">
        @for (preset of presets; track preset.label; let i = $index) {
          <button
            type="button"
            class="rounded-md border px-2.5 py-1 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
            [class.bg-muted]="model() === i"
            [attr.aria-pressed]="model() === i"
            (click)="model.set(i)"
          >
            {{ preset.label }}
          </button>
        }
      </div>

      <div class="mt-3 grid gap-3 sm:grid-cols-[minmax(0,11rem)_1fr]">
        <div
          class="flex flex-col items-stretch gap-1 text-xs"
          role="group"
          aria-label="Stages, from input to output"
        >
          @for (stage of stages; track stage.id; let i = $index) {
            @if (stage.id === 'ln1') {
              <div class="text-center text-[10px] text-muted-foreground" aria-hidden="true">
                ┌ repeated {{ shape().layers }}× ┐
              </div>
            }
            <div [class]="stage.inBlock ? 'border-x-2 border-dashed border-border px-1.5' : ''">
              <button
                type="button"
                class="w-full rounded-md border px-2 py-1 text-left hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
                [class.bg-muted]="index() === i"
                [class.font-semibold]="index() === i"
                [class.border-fig-1]="stage.id === 'attn'"
                [class.border-fig-2]="stage.id === 'mlp'"
                [attr.aria-pressed]="index() === i"
                (click)="index.set(i)"
              >
                {{ stage.label }}
              </button>
            </div>
            @if (stage.id === 'add2') {
              <div class="text-center text-[10px] text-muted-foreground" aria-hidden="true">
                └──────────┘
              </div>
            }
          }
        </div>

        <div class="rounded-md bg-muted/50 p-3 text-sm" aria-live="polite">
          <p class="text-xs text-muted-foreground">
            Stage {{ index() + 1 }} of {{ stages.length }}
          </p>
          <p class="mt-1 font-semibold">{{ stage().label }}</p>
          <p class="mt-1">{{ stage().text(shape(), heads()) }}</p>
          <dl class="mt-3 grid grid-cols-2 gap-2 text-xs">
            <div class="rounded-md bg-card p-2">
              <dt class="text-muted-foreground">output shape</dt>
              <dd class="font-mono tabular-nums">{{ stage().shape(shape()) }}</dd>
            </div>
            <div class="rounded-md bg-card p-2">
              <dt class="text-muted-foreground">
                parameters here{{ stage().inBlock ? ', all blocks' : '' }}
              </dt>
              <dd class="font-mono tabular-nums">
                {{ stage().params(shape(), counts()) | number }}
              </dd>
            </div>
          </dl>
          <div class="mt-3 flex gap-2">
            <button
              type="button"
              class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted disabled:opacity-50"
              [disabled]="index() === 0"
              (click)="index.set(index() - 1)"
            >
              ← Back
            </button>
            <button
              type="button"
              class="rounded-md border bg-primary px-3 py-1.5 text-xs text-primary-foreground hover:opacity-90 disabled:opacity-50"
              [disabled]="index() === stages.length - 1"
              (click)="index.set(index() + 1)"
            >
              Next stage →
            </button>
          </div>
        </div>
      </div>

      <div class="mt-4">
        <p class="text-xs">
          <span class="font-semibold">{{ presets[model()].label }}:</span>
          <span class="font-mono tabular-nums"> {{ counts().total | number }}</span> parameters
        </p>
        <div
          class="mt-1.5 flex h-3 w-full overflow-hidden rounded-full"
          role="img"
          [attr.aria-label]="shareText()"
        >
          @for (share of shares(); track share.key) {
            <div [class]="share.color" [style.width.%]="share.percent"></div>
          }
        </div>
        <div class="mt-1.5 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
          @for (share of shares(); track share.key) {
            <span class="flex items-center gap-1">
              <span class="inline-block size-2 rounded-full" [class]="share.color"></span>
              {{ share.label }} {{ share.percent < 1 ? '<1' : (share.percent | number: '1.0-0') }}%
            </span>
          }
        </div>
      </div>
      <p caption>
        Every arrow between stages carries the same thing: T vectors of d numbers, the residual
        stream. Only the start (ids in) and the end (scores out) look different. The counts are
        exact for GPT-2: 124,439,808 parameters for the small model. Switch to GPT-3 and the recipe
        is identical, only wider and deeper. At that size two thirds of the parameters sit in the
        MLPs, while the embedding table, a third of GPT-2 small, shrinks to under 1%.
      </p>
    </app-figure-frame>
  `,
})
export class TransformerBlockFigure {
  protected readonly stages = STAGES;
  protected readonly presets = PRESETS;
  protected readonly model = signal(0);
  protected readonly index = signal(3);

  protected readonly shape = computed(() => PRESETS[this.model()].shape);
  protected readonly heads = computed(() => PRESETS[this.model()].heads);
  protected readonly counts = computed(() => parameterCounts(this.shape()));
  protected readonly stage = computed(() => STAGES[this.index()]);

  protected readonly shares = computed(() => {
    const c = this.counts();
    const parts = {
      embeddings: c.tokenEmbedding + c.positions,
      attention: c.attention,
      mlp: c.mlp,
      norms: c.norms,
    };
    return SHARES.map((s) => ({ ...s, percent: (100 * parts[s.key]) / c.total }));
  });

  protected readonly shareText = computed(
    () =>
      'Where the parameters are: ' +
      this.shares()
        .map((s) => `${s.label} ${s.percent.toFixed(1)}%`)
        .join(', ') +
      '.',
  );
}
