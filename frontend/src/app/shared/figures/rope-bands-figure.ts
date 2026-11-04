import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { linear, path } from './lib/plot';
import { thetas } from './lib/rope';

/** Head size of Llama-style models: 64 clocks per head. */
const D = 128;
const BASES = [10_000, 500_000];
const LOG_MAX = 17; // distances up to 2¹⁷ = 131,072 tokens
// Integer distances, spread evenly on a log scale.
const DISTANCES = [
  ...new Set(Array.from({ length: 700 }, (_, i) => Math.round(2 ** ((i / 699) * LOG_MAX)))),
];
const sx = linear(0, LOG_MAX, 34, 340);
const sy = linear(-1, 1, 130, 10);
const TICKS = [1, 16, 256, 4096, 131_072];
const CONTEXT = 128_000; // Llama 3's context window

/** One RoPE clock's contribution to a perfect match, as the distance grows. */
@Component({
  selector: 'app-rope-bands-figure',
  imports: [FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Fast clocks for position, slow clocks for meaning"
      hint="Slide from the fastest clock to the slowest, then raise the base."
    >
      <div class="flex flex-wrap items-center gap-2" role="group" aria-label="RoPE base">
        <span class="text-xs text-muted-foreground">base</span>
        @for (b of bases; track b) {
          <button
            type="button"
            class="rounded-md border px-3 py-1.5 font-mono text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
            [class.bg-muted]="base() === b"
            [attr.aria-pressed]="base() === b"
            (click)="base.set(b)"
          >
            {{ b.toLocaleString('en-US') }}
          </button>
        }
      </div>
      <app-figure-slider
        class="mt-3"
        label="Clock (pair)"
        [(value)]="pair"
        [min]="0"
        [max]="last"
        [readout]="pairLabel()"
      />

      <svg viewBox="0 0 360 156" class="mt-3 w-full" role="img" [attr.aria-label]="summary()">
        <line x1="34" x2="340" [attr.y1]="sy(0)" [attr.y2]="sy(0)" class="stroke-border" />
        <line
          [attr.x1]="sx(Math.log2(context))"
          [attr.x2]="sx(Math.log2(context))"
          y1="10"
          y2="130"
          class="stroke-muted-foreground"
          stroke-dasharray="2 3"
        />
        <text
          [attr.x]="sx(Math.log2(context)) - 3"
          y="18"
          text-anchor="end"
          class="fill-muted-foreground text-[7px]"
        >
          128k context
        </text>
        @for (t of ticks; track t) {
          <text
            [attr.x]="sx(Math.log2(t))"
            y="142"
            text-anchor="middle"
            class="fill-muted-foreground text-[8px]"
          >
            {{ t.toLocaleString('en-US') }}
          </text>
        }
        <text x="187" y="154" text-anchor="middle" class="fill-muted-foreground text-[8px]">
          distance between query and key (tokens, log scale)
        </text>
        @for (v of [1, 0, -1]; track v) {
          <text
            x="30"
            [attr.y]="sy(v) + 3"
            text-anchor="end"
            class="fill-muted-foreground text-[8px]"
          >
            {{ v }}
          </text>
        }
        <line
          x1="34"
          x2="340"
          [attr.y1]="sy(1)"
          [attr.y2]="sy(1)"
          class="stroke-fig-2"
          stroke-width="1.5"
          stroke-dasharray="4 3"
        />
        <path [attr.d]="curve()" class="fill-none stroke-fig-1" stroke-width="1.5" />
      </svg>
      <p class="mt-1 flex flex-wrap gap-x-4 text-xs">
        <span class="text-fig-1">━ this clock, rotated (RoPE)</span>
        <span class="text-fig-2">┅ the same clock left unrotated (p-RoPE)</span>
      </p>
      <p class="mt-2 rounded-md bg-muted/50 p-2 text-sm" aria-live="polite">
        One full turn every
        <span class="font-semibold tabular-nums">{{ period().toLocaleString('en-US') }}</span>
        tokens, so over a 128,000-token context it turns
        <span class="font-semibold tabular-nums">{{ turns() }}</span> times. {{ verdict() }}
      </p>
      <p caption>
        The query and key match perfectly in this one pair, so without rotation they'd score 1 at
        every distance (the dashed line). Rotation multiplies that by cos(distance × θ). The fastest
        clocks swing between +1 and −1 from one token to the next: useless for meaning, but perfect
        for telling "the word right before me" apart from everything else, which is what positional
        heads use them for. The slowest clocks stay near 1 for thousands of tokens, so a match there
        means "same meaning" almost wherever the key is: a semantic channel. With base 10,000 even
        the slowest one drifts below 0 before 128k tokens; with base 500,000 it holds. p-RoPE goes
        one step further and simply doesn't rotate the slowest quarter of the clocks.
      </p>
    </app-figure-frame>
  `,
})
export class RopeBandsFigure {
  protected readonly Math = Math;
  protected readonly bases = BASES;
  protected readonly ticks = TICKS;
  protected readonly context = CONTEXT;
  protected readonly sx = sx;
  protected readonly sy = sy;
  protected readonly last = D / 2 - 1;
  protected readonly base = signal(10_000);
  protected readonly pair = signal(D / 2 - 1);

  private readonly theta = computed(() => thetas(D, this.base())[this.pair()]);
  protected readonly period = computed(() => Math.round((2 * Math.PI) / this.theta()));
  protected readonly turns = computed(() => {
    const turns = (CONTEXT * this.theta()) / (2 * Math.PI);
    return turns >= 10 ? Math.round(turns).toLocaleString('en-US') : turns.toFixed(2);
  });

  protected readonly pairLabel = computed(() => {
    const p = this.pair();
    const name = p === 0 ? 'fastest' : p === this.last ? 'slowest' : `${p + 1} of ${D / 2}`;
    return `${name}: ${this.theta().toPrecision(2)} rad per token`;
  });

  protected readonly verdict = computed(() => {
    const turns = (CONTEXT * this.theta()) / (2 * Math.PI);
    if (this.period() < 64) return 'A positional clock: it scrambles a match within a few tokens.';
    if (turns < 0.25) return 'A semantic channel: a match counts almost wherever the key is.';
    return 'In between: a match survives for a while, then drifts out of line.';
  });

  protected readonly curve = computed(() => {
    const theta = this.theta();
    return path(DISTANCES.map((t) => [sx(Math.log2(t)), sy(Math.cos(t * theta))] as const));
  });

  protected readonly summary = computed(
    () =>
      `Clock ${this.pair() + 1} of ${D / 2} with base ${this.base().toLocaleString('en-US')}: ` +
      `score of a perfect match is ${Math.cos(this.theta()).toFixed(2)} at distance 1, ` +
      `${Math.cos(1000 * this.theta()).toFixed(2)} at 1,000 and ` +
      `${Math.cos(CONTEXT * this.theta()).toFixed(2)} at 128,000.`,
  );
}
