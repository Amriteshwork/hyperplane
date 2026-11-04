import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { linear } from './lib/plot';
import { thetas } from './lib/rope';
import { LLAMA2, Method, extrapolated, scaledThetas, turns } from './lib/yarn';

const PAIRS = LLAMA2.d / 2;
const SCALES = [2, 4, 8, 16, 32];
const ORIGINAL = thetas(LLAMA2.d, LLAMA2.base);
const R = ORIGINAL.map((t) => turns(t, LLAMA2.L));
// Zone edges: the last pair with over 32 turns, and the first that never completes one.
const FAST_END = R.filter((r) => r > 32).length;
const SLOW_START = R.findIndex((r) => r < 1);
const BAR = 5;
const LEFT = 34;
const bx = (i: number) => LEFT + i * BAR;
const sy = linear(0, 5, 130, 20); // log₂ of the slowdown: 1× … 32×

/** How PI, NTK-aware and NTK-by-parts each slow down Llama 2's 64 clocks to stretch its context. */
@Component({
  selector: 'app-yarn-frequencies-figure',
  imports: [FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Three ways to slow the clocks"
      hint="Pick a method and a stretch. Red bars are clocks pushed to angles they never saw."
    >
      <div class="flex flex-wrap gap-2" role="group" aria-label="Method">
        @for (m of methods; track m.id) {
          <button
            type="button"
            class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
            [class.bg-muted]="method() === m.id"
            [attr.aria-pressed]="method() === m.id"
            (click)="method.set(m.id)"
          >
            {{ m.label }}
          </button>
        }
      </div>
      <div class="mt-2 flex flex-wrap items-center gap-2" role="group" aria-label="Stretch s">
        <span class="text-xs text-muted-foreground">stretch s</span>
        @for (v of scales; track v) {
          <button
            type="button"
            class="rounded-md border px-2.5 py-1 font-mono text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
            [class.bg-muted]="s() === v"
            [attr.aria-pressed]="s() === v"
            (click)="s.set(v)"
          >
            {{ v }}× → {{ (4 * v).toLocaleString('en-US') }}k
          </button>
        }
      </div>

      <svg viewBox="0 0 360 168" class="mt-3 w-full" role="img" [attr.aria-label]="summary()">
        <rect
          [attr.x]="bx(0)"
          y="14"
          [attr.width]="fastEnd * bar"
          height="116"
          class="fill-fig-1/10"
        />
        <rect
          [attr.x]="bx(slowStart)"
          y="14"
          [attr.width]="(pairs - slowStart) * bar"
          height="116"
          class="fill-fig-2/10"
        />
        <text [attr.x]="bx(fastEnd / 2)" y="11" text-anchor="middle" class="fill-fig-1 text-[7px]">
          fast: 32+ turns in training
        </text>
        <text
          [attr.x]="bx((slowStart + pairs) / 2)"
          y="11"
          text-anchor="middle"
          class="fill-fig-2 text-[7px]"
        >
          slow: under 1 turn
        </text>
        @for (v of [1, 2, 4, 8, 16, 32]; track v) {
          <line
            [attr.x1]="left"
            [attr.x2]="bx(pairs)"
            [attr.y1]="sy(Math.log2(v))"
            [attr.y2]="sy(Math.log2(v))"
            class="stroke-border"
            stroke-width="0.5"
          />
          <text
            [attr.x]="left - 4"
            [attr.y]="sy(Math.log2(v)) + 3"
            text-anchor="end"
            class="fill-muted-foreground text-[8px]"
          >
            {{ v }}×
          </text>
        }
        @for (b of bars(); track $index) {
          <rect
            [attr.x]="bx($index) + 0.5"
            [attr.y]="sy(b.log)"
            [attr.width]="bar - 1"
            [attr.height]="Math.max(0.8, sy(0) - sy(b.log))"
            [class]="b.out ? 'fill-fig-negative' : 'fill-fig-1'"
          />
        }
        <text [attr.x]="bx(0)" y="142" class="fill-muted-foreground text-[8px]">pair 1</text>
        <text
          [attr.x]="bx(pairs)"
          y="142"
          text-anchor="end"
          class="fill-muted-foreground text-[8px]"
        >
          pair 64
        </text>
        <text
          [attr.x]="bx(pairs / 2)"
          y="156"
          text-anchor="middle"
          class="fill-muted-foreground text-[8px]"
        >
          each bar: how many times slower that clock now turns
        </text>
      </svg>

      <p class="mt-2 rounded-md bg-muted/50 p-2 text-sm" aria-live="polite">
        Neighbouring words now differ by
        <span class="font-semibold tabular-nums">{{ step() }} rad</span> on the fastest clock (was 1
        rad). Untouched clocks: <span class="font-semibold tabular-nums">{{ untouched() }}</span> of
        64. Pushed past angles never seen in training:
        <span class="font-semibold tabular-nums" [class.text-fig-negative]="outCount() > 0">{{
          outCount()
        }}</span
        >.
      </p>
      <p caption>
        Llama 2: 64 clocks per head, trained on 4,096 tokens. To read s times more text, every
        method slows some clocks. <b>Position interpolation</b> slows all of them by s, so even the
        fastest clock barely moves between neighbouring words: the model loses its sense of "right
        next to me". <b>NTK-aware</b> raises the base instead, leaving the fastest clock alone and
        slowing the slowest by exactly s, but the clocks just before the slowest are slowed less
        than s, so their angles run past anything seen in training (red). <b>NTK-by-parts</b>, the
        heart of YaRN, decides per clock: clocks that turned 32+ times in training have seen every
        angle and stay untouched, clocks that never completed a turn are slowed by the full s, and
        the ones in between get a mix.
      </p>
    </app-figure-frame>
  `,
})
export class YarnFrequenciesFigure {
  protected readonly Math = Math;
  protected readonly scales = SCALES;
  protected readonly pairs = PAIRS;
  protected readonly bar = BAR;
  protected readonly left = LEFT;
  protected readonly fastEnd = FAST_END;
  protected readonly slowStart = SLOW_START;
  protected readonly bx = bx;
  protected readonly sy = sy;
  protected readonly methods: readonly { id: Method; label: string }[] = [
    { id: 'pi', label: 'Position interpolation' },
    { id: 'ntk-aware', label: 'NTK-aware' },
    { id: 'by-parts', label: 'NTK-by-parts (YaRN)' },
  ];

  protected readonly method = signal<Method>('pi');
  protected readonly s = signal(16);

  private readonly scaled = computed(() => scaledThetas(this.method(), this.s()));
  private readonly out = computed(() => new Set(extrapolated(this.method(), this.s())));

  protected readonly bars = computed(() =>
    this.scaled().map((t, i) => ({
      log: Math.log2(ORIGINAL[i] / t),
      out: this.out().has(i),
    })),
  );
  protected readonly step = computed(() => this.scaled()[0].toFixed(3));
  protected readonly untouched = computed(
    () => this.scaled().filter((t, i) => Math.abs(t - ORIGINAL[i]) < 1e-12).length,
  );
  protected readonly outCount = computed(() => this.out().size);

  protected readonly summary = computed(
    () =>
      `${this.methods.find((m) => m.id === this.method())?.label}, stretch ${this.s()}×: ` +
      `${this.untouched()} of 64 clocks untouched, ${this.outCount()} pushed past trained angles, ` +
      `slowest clock ${(ORIGINAL[PAIRS - 1] / this.scaled()[PAIRS - 1]).toFixed(1)}× slower.`,
  );
}
