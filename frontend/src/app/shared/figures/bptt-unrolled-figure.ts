import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { Activation, unroll } from './lib/bptt';

// The first input is the thing worth remembering; the rest is small noise.
const XS = [1, 0.2, -0.3, 0.1, 0.25, -0.15, 0.05, 0.3, -0.2, 0.1];
const T = XS.length;
const TARGET = 1;
const LEFT = 50;
const GAP = 30;
const BASE = 182; // bars' baseline; heights are log-scaled
const LOG_MIN = -6;
const LOG_MAX = 2;
const BAR_SPAN = 72;

/** Formats a ratio readably at any size: 0.42, 3.1e−5, 57. */
function times(v: number): string {
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.01 || a >= 1000)) return v.toExponential(1).replace('-', '−');
  return v.toFixed(2);
}

/** An RNN unrolled over ten steps: walk the gradient back from the loss and watch it shrink or grow. */
@Component({
  selector: 'app-bptt-unrolled-figure',
  imports: [FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Backpropagation through time, unrolled"
      hint="Step the gradient back from the loss, one time step at a time. Then change w."
    >
      <svg viewBox="0 0 360 190" class="w-full" role="img" [attr.aria-label]="summary()">
        @for (tick of ticks; track tick.label) {
          <line
            [attr.x1]="LEFT - 14"
            x2="340"
            [attr.y1]="tick.y"
            [attr.y2]="tick.y"
            class="stroke-border"
            stroke-dasharray="2 3"
          />
          <text
            [attr.x]="LEFT - 17"
            [attr.y]="tick.y + 3"
            text-anchor="end"
            class="fill-muted-foreground text-[7px]"
          >
            {{ tick.label }}
          </text>
        }
        @for (t of steps; track t) {
          <text
            [attr.x]="x(t)"
            y="14"
            text-anchor="middle"
            class="fill-muted-foreground text-[8px]"
          >
            x{{ sub(t) }}
          </text>
          <line [attr.x1]="x(t)" [attr.x2]="x(t)" y1="18" y2="44" class="stroke-border" />
          @if (t > 1) {
            <line
              [attr.x1]="x(t - 1) + 10"
              [attr.x2]="x(t) - 10"
              y1="56"
              y2="56"
              [class]="revealed(t - 1) ? 'stroke-fig-2' : 'stroke-border'"
              [attr.stroke-width]="revealed(t - 1) ? 2 : 1.2"
            />
          }
          <circle
            [attr.cx]="x(t)"
            cy="56"
            r="10"
            class="fill-fig-1 stroke-fig-1"
            fill-opacity="0.2"
            stroke-width="1.2"
          />
          <text
            [attr.x]="x(t)"
            y="59"
            text-anchor="middle"
            class="fill-foreground text-[8px] font-semibold"
          >
            h{{ sub(t) }}
          </text>
          <text
            [attr.x]="x(t)"
            y="80"
            text-anchor="middle"
            class="fill-fig-1 text-[7px] tabular-nums"
          >
            {{ run().h[t].toFixed(2) }}
          </text>
          @if (revealed(t)) {
            <rect
              [attr.x]="x(t) - 8"
              [attr.y]="BASE - bar(t)"
              width="16"
              [attr.height]="bar(t)"
              class="fill-fig-2"
              fill-opacity="0.8"
            />
          }
        }
        <line
          [attr.x1]="x(T) + 10"
          [attr.x2]="340"
          y1="56"
          y2="56"
          class="stroke-fig-2"
          stroke-width="2"
        />
        <text x="348" y="59" text-anchor="middle" class="fill-fig-2 text-[9px] font-semibold">
          L
        </text>

        <text x="4" y="100" class="fill-fig-2 text-[7px]">|δₜ| vs |δ₁₀|</text>
      </svg>

      <div class="mt-2 rounded-md bg-muted/50 p-3 text-sm" aria-live="polite">
        <p>{{ readout() }}</p>
        <p class="mt-1 font-mono text-xs leading-5">{{ maths() }}</p>
      </div>

      <div class="mt-3 grid gap-3 sm:grid-cols-2">
        <app-figure-slider
          label="Recurrent weight w"
          [(value)]="w"
          [min]="0.1"
          [max]="2"
          [step]="0.05"
          [readout]="w().toFixed(2)"
        />
        <div class="flex flex-wrap items-end gap-2" role="group" aria-label="Activation">
          @for (option of activations; track option.id) {
            <button
              type="button"
              class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted"
              [class.bg-muted]="activation() === option.id"
              [attr.aria-pressed]="activation() === option.id"
              (click)="activation.set(option.id)"
            >
              {{ option.label }}
            </button>
          }
        </div>
      </div>
      <div class="mt-3 flex flex-wrap gap-2">
        <button
          type="button"
          class="rounded-md border bg-primary px-3 py-1.5 text-xs text-primary-foreground hover:opacity-90 disabled:opacity-50"
          [disabled]="earliest() === 1"
          (click)="earliest.set(earliest() - 1)"
        >
          ← Step back
        </button>
        <button
          type="button"
          class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted disabled:opacity-50"
          [disabled]="earliest() === 1"
          (click)="earliest.set(1)"
        >
          Show all
        </button>
        <button
          type="button"
          class="rounded-md px-3 py-1.5 text-xs text-muted-foreground hover:bg-muted"
          (click)="earliest.set(T)"
        >
          Start over
        </button>
      </div>
      <p caption>
        A one-unit RNN, hₜ = tanh(w·hₜ₋₁ + xₜ), reads ten inputs and is scored only on its last
        state. Backpropagation through time starts at the loss and walks left. Every step back
        multiplies the gradient by the same kind of factor, w·(1 − hₜ²), so ten steps multiply ten
        of them. Below 1 the far past fades to nothing, above 1 it blows up, and tanh only ever
        shrinks the factor. That's why a plain RNN struggles to learn from anything more than a few
        steps back.
      </p>
    </app-figure-frame>
  `,
})
export class BpttUnrolledFigure {
  protected readonly T = T;
  protected readonly LEFT = LEFT;
  protected readonly BASE = BASE;
  protected readonly steps = Array.from({ length: T }, (_, i) => i + 1);
  protected readonly activations: { id: Activation; label: string }[] = [
    { id: 'tanh', label: 'tanh' },
    { id: 'linear', label: 'No activation' },
  ];
  protected readonly ticks = [2, 0, -2, -4, -6].map((p) => ({
    y: BASE - ((p - LOG_MIN) / (LOG_MAX - LOG_MIN)) * BAR_SPAN,
    label: p === 0 ? '1' : `10${p < 0 ? '⁻' : ''}${'⁰¹²³⁴⁵⁶'[Math.abs(p)]}`,
  }));

  protected readonly w = signal(0.9);
  protected readonly activation = signal<Activation>('tanh');
  /** The earliest step the gradient has been walked back to. */
  protected readonly earliest = signal(T);

  protected readonly run = computed(() => unroll(XS, this.w(), this.activation(), TARGET));
  private readonly ratio = (t: number) => this.run().delta[t] / this.run().delta[T];

  protected readonly readout = computed(() => {
    const t = this.earliest();
    const r = this.run();
    if (t === T) {
      return `At the last step the gradient is just the error: δ₁₀ = h₁₀ − y = ${r.delta[T].toFixed(3)}. Step back to send it left.`;
    }
    const reach = `δ${this.sub(t)} is ${times(this.ratio(t))} × δ₁₀ after ${T - t} step${T - t === 1 ? '' : 's'} back.`;
    if (t > 1) return reach;
    // Contributions can have either sign, so compare their sizes.
    const size = (from: number, to: number) =>
      r.dW.slice(from, to).reduce((a, b) => a + Math.abs(b), 0);
    const share = (size(1, 4) / (size(1, T + 1) || 1)) * 100;
    const steps = `Steps 1–3 make up ${share.toFixed(share < 1 ? 2 : 0)}% of ∂L/∂w`;
    const end = Math.abs(this.ratio(1));
    if (end < 0.1) return `${reach} ${steps}: the first input barely shapes what w learns.`;
    if (end > 10)
      return `${reach} ${steps}, and every update is magnified ${times(end)}×: exploding.`;
    return `${reach} ${steps}.`;
  });

  protected readonly maths = computed(() => {
    const t = this.earliest();
    const r = this.run();
    if (t === T) return `L = ½(h₁₀ − y)² = ${r.loss.toFixed(3)}`;
    const slope = this.activation() === 'tanh' ? ` × (1 − ${r.h[t + 1].toFixed(2)}²)` : '';
    return `δ${this.sub(t)} = δ${this.sub(t + 1)} × w${slope} = ${times(r.delta[t + 1])} × ${r.factor[t + 1].toFixed(3)} = ${times(r.delta[t])}`;
  });

  protected readonly summary = computed(() => {
    return `Ten steps unrolled with w = ${this.w().toFixed(2)} (${this.activation()}). The gradient reaching step 1 is ${times(this.ratio(1))} times the one at the loss.`;
  });

  protected x(t: number): number {
    return LEFT + (t - 1) * GAP;
  }

  protected sub(t: number): string {
    return String(t).replace(/\d/g, (d) => '₀₁₂₃₄₅₆₇₈₉'[Number(d)]);
  }

  protected revealed(t: number): boolean {
    return t >= this.earliest();
  }

  protected bar(t: number): number {
    const v = Math.log10(Math.max(Math.abs(this.ratio(t)), 1e-12));
    const clamped = Math.min(Math.max(v, LOG_MIN), LOG_MAX);
    return Math.max(((clamped - LOG_MIN) / (LOG_MAX - LOG_MIN)) * BAR_SPAN, 1);
  }
}
