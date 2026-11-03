import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { linear, path } from './lib/plot';
import { alignedScore, decayBound, thetas } from './lib/rope';

/** Head size of Llama-style models: 64 clocks per head. */
const D = 128;
const BASES = [100, 10_000, 500_000];
const LOG_MAX = 12; // distances up to 2¹² = 4,096 tokens
const DISTANCES = Array.from({ length: 241 }, (_, i) => Math.round(2 ** ((i / 240) * LOG_MAX)));
const sx = linear(0, LOG_MAX, 34, 340);
const sy = linear(-0.3, 1, 150, 10);
const TICKS = [1, 4, 16, 64, 256, 1024, 4096];

/** How loudly a perfectly matching key is heard as it moves away, for three RoPE bases. */
@Component({
  selector: 'app-rope-decay-figure',
  imports: [FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="How fast a match fades with distance"
      hint="Change the base: a bigger base slows every clock down, so the fade starts later."
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

      <svg viewBox="0 0 360 176" class="mt-3 w-full" role="img" [attr.aria-label]="summary()">
        <line x1="34" x2="340" [attr.y1]="sy(0)" [attr.y2]="sy(0)" class="stroke-border" />
        <line x1="34" x2="34" y1="10" y2="150" class="stroke-border" />
        @for (t of ticks; track t) {
          <text
            [attr.x]="sx(Math.log2(t))"
            y="162"
            text-anchor="middle"
            class="fill-muted-foreground text-[8px]"
          >
            {{ t.toLocaleString('en-US') }}
          </text>
        }
        <text x="192" y="174" text-anchor="middle" class="fill-muted-foreground text-[8px]">
          distance between query and key (tokens, log scale)
        </text>
        @for (v of [1, 0.5, 0]; track v) {
          <text
            x="30"
            [attr.y]="sy(v) + 3"
            text-anchor="end"
            class="fill-muted-foreground text-[8px]"
          >
            {{ v }}
          </text>
        }
        <path
          [attr.d]="bound()"
          class="fill-none stroke-fig-2"
          stroke-width="1.5"
          stroke-dasharray="4 3"
        />
        <path [attr.d]="aligned()" class="fill-none stroke-fig-1" stroke-width="2" />
      </svg>
      <p class="mt-1 flex flex-wrap gap-x-4 text-xs">
        <span class="text-fig-1">━ score of a perfect match</span>
        <span class="text-fig-2">┅ the paper's ceiling on any score</span>
      </p>
      <p class="mt-2 rounded-md bg-muted/50 p-2 text-sm" aria-live="polite">
        Fastest clock: one turn every 6 tokens. Slowest: one turn every
        <span class="font-semibold tabular-nums">{{ slowest().toLocaleString('en-US') }}</span>
        tokens. A perfect match 1,000 tokens away keeps
        <span class="font-semibold tabular-nums">{{ at1000() }}%</span> of its score.
      </p>
      <p caption>
        Each head has 64 clocks (d = 128, as in Llama). The solid line is a query and key that are
        identical, so at distance 0 every clock agrees and the score is as large as it can be (1 on
        this scale). Move them apart and each clock turns by its own angle, the 64 terms stop lining
        up and start cancelling. The dashed line is the bound the paper proves in section 3.4.3: it
        caps the score of any query and key, and it shrinks with distance too. With base 100 the
        slowest clock turns once every ~580 tokens and the curve falls fast; base 10,000 is the
        paper's choice; Llama 3 uses 500,000 to keep long-range scores alive across its much longer
        context.
      </p>
    </app-figure-frame>
  `,
})
export class RopeDecayFigure {
  protected readonly Math = Math;
  protected readonly bases = BASES;
  protected readonly ticks = TICKS;
  protected readonly sx = sx;
  protected readonly sy = sy;
  protected readonly base = signal(10_000);

  protected readonly aligned = computed(() => this.curve(alignedScore));
  protected readonly bound = computed(() => this.curve(decayBound));
  protected readonly slowest = computed(() => {
    const all = thetas(D, this.base());
    return Math.round((2 * Math.PI) / all[all.length - 1]);
  });
  protected readonly at1000 = computed(() => Math.round(100 * alignedScore(1000, D, this.base())));

  protected readonly summary = computed(
    () =>
      `With base ${this.base().toLocaleString('en-US')}, a perfectly matching key keeps ` +
      `${Math.round(100 * alignedScore(16, D, this.base()))}% of its score at 16 tokens, ` +
      `${Math.round(100 * alignedScore(256, D, this.base()))}% at 256 and ` +
      `${this.at1000()}% at 1,000.`,
  );

  private curve(f: (distance: number, d: number, base: number) => number): string {
    const base = this.base();
    return path(DISTANCES.map((t) => [sx(Math.log2(t)), sy(f(t, D, base))] as const));
  }
}
