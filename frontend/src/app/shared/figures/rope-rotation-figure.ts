import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { linear } from './lib/plot';
import { Vec2, dot, rotate } from './lib/rope';

const SIZE = 240;
const C = SIZE / 2;
const toPx = linear(0, 1.6, 0, C - 8);
const Q: Vec2 = [0.95, 0.3];
const K: Vec2 = [0.45, -0.8];
const MAX = 24;
/** How many shifted copies of the same (m, n) gap the bar strip compares. */
const SHIFTS = 9;
const bar = linear(-1.5, 1.5, 52, 2);

type Mode = 'rotate' | 'add';

/** q and k as two arrows: RoPE turns each by its position, so only the gap survives the dot product. */
@Component({
  selector: 'app-rope-rotation-figure',
  imports: [FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Rotate, don't add"
      hint="Move m and n, then shift both together. Switch to 'Add a stamp' and shift again."
    >
      <div class="flex flex-wrap gap-2" role="group" aria-label="How position is mixed in">
        @for (option of modes; track option.id) {
          <button
            type="button"
            class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
            [class.bg-muted]="mode() === option.id"
            [attr.aria-pressed]="mode() === option.id"
            (click)="mode.set(option.id)"
          >
            {{ option.label }}
          </button>
        }
      </div>

      <svg
        [attr.viewBox]="'0 0 ' + size + ' ' + size"
        class="mx-auto mt-3 w-full max-w-xs"
        role="img"
        [attr.aria-label]="picture()"
      >
        <line x1="8" [attr.x2]="size - 8" [attr.y1]="c" [attr.y2]="c" class="stroke-border" />
        <line y1="8" [attr.y2]="size - 8" [attr.x1]="c" [attr.x2]="c" class="stroke-border" />
        <circle
          [attr.cx]="c"
          [attr.cy]="c"
          [attr.r]="px(1)"
          class="fill-none stroke-border"
          stroke-dasharray="2 3"
        />
        <!-- Where q and k start, at position 0. -->
        @for (v of [q0, k0]; track $index) {
          <line
            [attr.x1]="c"
            [attr.y1]="c"
            [attr.x2]="x(v)"
            [attr.y2]="y(v)"
            [class]="$index ? 'stroke-fig-2' : 'stroke-fig-1'"
            stroke-opacity="0.35"
            stroke-width="2"
            stroke-dasharray="3 3"
          />
        }
        <path [attr.d]="arc()" class="fill-none stroke-foreground" stroke-width="1.2" />
        @for (v of [qm(), kn()]; track $index) {
          <line
            [attr.x1]="c"
            [attr.y1]="c"
            [attr.x2]="x(v)"
            [attr.y2]="y(v)"
            [class]="$index ? 'stroke-fig-2' : 'stroke-fig-1'"
            stroke-width="3"
            stroke-linecap="round"
          />
          <circle
            [attr.cx]="x(v)"
            [attr.cy]="y(v)"
            r="4"
            [class]="$index ? 'fill-fig-2' : 'fill-fig-1'"
          />
          <text
            [attr.x]="x(v) + (v[0] >= 0 ? 7 : -7)"
            [attr.y]="y(v) + 4"
            [attr.text-anchor]="v[0] >= 0 ? 'start' : 'end'"
            [class]="($index ? 'fill-fig-2' : 'fill-fig-1') + ' text-[10px] font-semibold'"
          >
            {{ $index ? 'k at n=' + n() : 'q at m=' + m() }}
          </text>
        }
      </svg>

      <div class="mt-2 grid gap-3 sm:grid-cols-3">
        <app-figure-slider label="Query position m" [(value)]="m" [min]="0" [max]="max" />
        <app-figure-slider label="Key position n" [(value)]="n" [min]="0" [max]="max" />
        <app-figure-slider
          label="Turn per position θ"
          [(value)]="degrees"
          [min]="5"
          [max]="60"
          [step]="5"
          [readout]="degrees() + '°'"
        />
      </div>
      <button
        type="button"
        class="mt-3 rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring disabled:opacity-50"
        [disabled]="m() === max || n() === max"
        (click)="shift()"
      >
        Shift both by +1
      </button>

      <p class="mt-2 rounded-md bg-muted/50 p-2 text-sm" aria-live="polite">
        Gap n − m = {{ n() - m() }}.
        @if (mode() === 'rotate') {
          Since position 0, the angle between the arrows has changed by {{ n() - m() }} ×
          {{ degrees() }}° = {{ (n() - m()) * degrees() }}°.
        }
        Score q · k = <span class="font-semibold tabular-nums">{{ score().toFixed(3) }}</span>
      </p>

      <p class="mt-3 text-xs">
        <span class="font-semibold">Same gap, different places:</span>
        <span class="text-muted-foreground">
          the score at (m + s, n + s) for s = 0 … {{ shifts - 1 }}
        </span>
      </p>
      <svg viewBox="0 0 300 64" class="w-full" role="img" [attr.aria-label]="stripText()">
        <line x1="0" x2="300" [attr.y1]="bar(0)" [attr.y2]="bar(0)" class="stroke-border" />
        @for (s of strip(); track $index) {
          <rect
            [attr.x]="6 + $index * 32"
            [attr.y]="Math.min(bar(s), bar(0))"
            width="24"
            [attr.height]="Math.max(1, Math.abs(bar(s) - bar(0)))"
            class="fill-fig-1"
          />
          <text
            [attr.x]="18 + $index * 32"
            y="62"
            text-anchor="middle"
            class="fill-muted-foreground text-[8px]"
          >
            s={{ $index }}
          </text>
        }
      </svg>

      <p caption>
        The faded dashed arrows are a query q and a key k before any position is mixed in. With
        <b>rotate</b>, the query is turned by m·θ and the key by n·θ. Shift both by +1 and both
        arrows turn by the same θ, like two people standing on a turntable: the angle between them,
        and so the dot product, can't change. Every bar in the strip is the same height. With
        <b>add a stamp</b>, each arrow instead gets a small position vector added to its tip (here
        the 2D sinusoidal stamp, half size). Now shifting both changes the score, so the same gap
        reads differently at different places in the sentence, and the bars wobble.
      </p>
    </app-figure-frame>
  `,
})
export class RopeRotationFigure {
  protected readonly Math = Math;
  protected readonly size = SIZE;
  protected readonly c = C;
  protected readonly max = MAX;
  protected readonly shifts = SHIFTS;
  protected readonly bar = bar;
  protected readonly px = toPx;
  protected readonly q0 = Q;
  protected readonly k0 = K;
  protected readonly modes: readonly { id: Mode; label: string }[] = [
    { id: 'rotate', label: 'Rotate (RoPE)' },
    { id: 'add', label: 'Add a stamp' },
  ];

  protected readonly mode = signal<Mode>('rotate');
  protected readonly m = signal(2);
  protected readonly n = signal(5);
  protected readonly degrees = signal(20);

  private readonly theta = computed(() => (this.degrees() * Math.PI) / 180);
  protected readonly qm = computed(() => this.place(Q, this.m()));
  protected readonly kn = computed(() => this.place(K, this.n()));
  protected readonly score = computed(() => dot(this.qm(), this.kn()));
  protected readonly strip = computed(() =>
    Array.from({ length: SHIFTS }, (_, s) =>
      dot(this.place(Q, this.m() + s), this.place(K, this.n() + s)),
    ),
  );

  /** A small arc from the query's direction to the key's, to show the angle between them. */
  protected readonly arc = computed(() => {
    const a = Math.atan2(this.qm()[1], this.qm()[0]);
    let b = Math.atan2(this.kn()[1], this.kn()[0]);
    while (b - a > Math.PI) b -= 2 * Math.PI;
    while (a - b > Math.PI) b += 2 * Math.PI;
    const r = 26;
    const end = (t: number) =>
      `${(C + r * Math.cos(t)).toFixed(1)},${(C - r * Math.sin(t)).toFixed(1)}`;
    return `M${end(a)}A${r},${r} 0 0 ${b > a ? 0 : 1} ${end(b)}`;
  });

  protected readonly picture = computed(
    () =>
      `Query at position ${this.m()} and key at position ${this.n()}, ` +
      `${this.mode() === 'rotate' ? 'rotated' : 'with a position stamp added'}; ` +
      `their dot product is ${this.score().toFixed(3)}.`,
  );

  protected readonly stripText = computed(() => {
    const values = this.strip();
    const spread = Math.max(...values) - Math.min(...values);
    return (
      `Scores for the same gap at ${SHIFTS} places: ` +
      values.map((v) => v.toFixed(2)).join(', ') +
      (spread < 1e-9 ? '. All equal.' : `. They differ by up to ${spread.toFixed(2)}.`)
    );
  });

  protected x(v: Vec2): number {
    return C + toPx(v[0]);
  }

  protected y(v: Vec2): number {
    return C - toPx(v[1]);
  }

  protected shift(): void {
    this.m.update((m) => Math.min(MAX, m + 1));
    this.n.update((n) => Math.min(MAX, n + 1));
  }

  private place(v: Vec2, position: number): Vec2 {
    const angle = position * this.theta();
    if (this.mode() === 'rotate') return rotate(v, angle);
    return [v[0] + 0.5 * Math.sin(angle), v[1] + 0.5 * Math.cos(angle)];
  }
}
