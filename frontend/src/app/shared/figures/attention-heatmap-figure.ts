import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { SENTENCE, attentionWeights } from './lib/attention';

const CELL = 38;
const LEFT = 64;
const TOP = 34;
const N = SENTENCE.length;

/** The attention pattern of a hand-built sentence: pick a word, sharpen the softmax, mask the future. */
@Component({
  selector: 'app-attention-heatmap-figure',
  imports: [DecimalPipe, FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Who looks at whom"
      hint="Pick a word to see its question and where its attention goes. Then change the sharpness, or mask the future."
    >
      <div class="flex flex-wrap gap-1.5" role="group" aria-label="Word asking the question">
        @for (token of sentence; track $index; let i = $index) {
          <button
            type="button"
            class="rounded-md border px-2 py-1 font-mono text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
            [class.bg-muted]="selected() === i"
            [attr.aria-pressed]="selected() === i"
            (click)="selected.set(i)"
          >
            {{ token.word }}
          </button>
        }
      </div>
      <svg
        [attr.viewBox]="'0 0 ' + width + ' ' + height"
        class="mt-3 w-full max-w-md"
        role="img"
        [attr.aria-label]="description()"
      >
        @for (token of sentence; track $index; let s = $index) {
          <text
            [attr.x]="LEFT + s * CELL + CELL / 2"
            [attr.y]="TOP - 8"
            text-anchor="middle"
            class="fill-foreground font-mono text-[9px]"
          >
            {{ token.word }}
          </text>
        }
        @for (row of weights(); track $index; let t = $index) {
          <text
            [attr.x]="LEFT - 6"
            [attr.y]="TOP + t * CELL + CELL / 2 + 3"
            text-anchor="end"
            class="font-mono text-[9px]"
            [class]="t === selected() ? 'fill-fig-2 font-semibold' : 'fill-foreground'"
          >
            {{ sentence[t].word }}
          </text>
          @for (w of row; track $index; let s = $index) {
            <rect
              [attr.x]="LEFT + s * CELL + 1"
              [attr.y]="TOP + t * CELL + 1"
              [attr.width]="CELL - 2"
              [attr.height]="CELL - 2"
              rx="3"
              [class]="hidden(t, s) ? 'fill-muted' : 'fill-fig-1'"
              [attr.fill-opacity]="hidden(t, s) ? 1 : 0.06 + 0.94 * w"
            />
            @if (hidden(t, s)) {
              <text
                [attr.x]="LEFT + s * CELL + CELL / 2"
                [attr.y]="TOP + t * CELL + CELL / 2 + 3"
                text-anchor="middle"
                class="fill-muted-foreground text-[9px]"
              >
                ✕
              </text>
            } @else if (w >= 0.1) {
              <text
                [attr.x]="LEFT + s * CELL + CELL / 2"
                [attr.y]="TOP + t * CELL + CELL / 2 + 3"
                text-anchor="middle"
                class="text-[9px] tabular-nums"
                [class]="w > 0.5 ? 'fill-card' : 'fill-foreground'"
              >
                {{ w | number: '1.2-2' }}
              </text>
            }
          }
        }
        <rect
          [attr.x]="LEFT"
          [attr.y]="TOP + selected() * CELL"
          [attr.width]="N * CELL"
          [attr.height]="CELL"
          rx="4"
          class="fill-none stroke-fig-2"
          stroke-width="2"
        />
      </svg>

      <p class="mt-1 text-xs text-muted-foreground">
        Rows: the word asking. Columns: the words it looks at.
      </p>
      <div class="mt-2 rounded-md bg-muted/50 p-3" aria-live="polite">
        <p class="text-sm">
          <span class="font-mono font-semibold">{{ token().word }}</span> asks: “{{ token().asks }}”
        </p>
        <p class="mt-1 text-xs text-muted-foreground">
          Its answer is mostly <span class="font-mono text-foreground">{{ top().word }}</span> ({{
            top().weight | number: '1.0-0'
          }}% of its attention).
        </p>
      </div>

      <div class="mt-3 grid gap-3 sm:grid-cols-2">
        <app-figure-slider
          label="Sharpness (how big the scores are)"
          [(value)]="sharpness"
          [min]="0"
          [max]="12"
          [step]="0.5"
          [readout]="'×' + sharpness()"
        />
        <button
          type="button"
          class="self-end rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
          [class.bg-muted]="causal()"
          [attr.aria-pressed]="causal()"
          (click)="causal.set(!causal())"
        >
          {{
            causal() ? 'Causal mask on: no peeking ahead' : 'Causal mask off: every word sees all'
          }}
        </button>
      </div>
      <p caption>
        Each row is one word's softmax over the scores of its query against every key, so each row
        adds up to 1. These queries and keys were picked by hand to be readable; a trained model
        learns its own, and they are rarely this tidy. At sharpness 0 every row is a plain average:
        a vague question. Turn it up and each word snaps onto one answer. That is what large,
        unscaled dot products do, and why attention divides by √d. With the mask on, everything
        above the diagonal is gone: "the" can no longer see the noun it introduces, and "animal"
        sees only "the" and itself.
      </p>
    </app-figure-frame>
  `,
})
export class AttentionHeatmapFigure {
  protected readonly sentence = SENTENCE;
  protected readonly CELL = CELL;
  protected readonly LEFT = LEFT;
  protected readonly TOP = TOP;
  protected readonly N = N;
  protected readonly width = LEFT + N * CELL + 4;
  protected readonly height = TOP + N * CELL + 4;

  protected readonly selected = signal(SENTENCE.findIndex((t) => t.word === 'it'));
  protected readonly sharpness = signal(3);
  protected readonly causal = signal(false);

  protected readonly weights = computed(() =>
    attentionWeights(
      SENTENCE.map((t) => t.query),
      SENTENCE.map((t) => t.key),
      this.sharpness(),
      this.causal(),
    ),
  );
  protected readonly token = computed(() => SENTENCE[this.selected()]);
  protected readonly top = computed(() => {
    const row = this.weights()[this.selected()];
    const best = row.indexOf(Math.max(...row));
    return { word: SENTENCE[best].word, weight: 100 * row[best] };
  });

  protected hidden(t: number, s: number): boolean {
    return this.causal() && s > t;
  }

  protected readonly description = computed(
    () =>
      `Attention weights for "${SENTENCE.map((t) => t.word).join(' ')}", sharpness ` +
      `${this.sharpness()}, causal mask ${this.causal() ? 'on' : 'off'}. "${this.token().word}" ` +
      `puts ${this.top().weight.toFixed(0)}% of its attention on "${this.top().word}".`,
  );
}
