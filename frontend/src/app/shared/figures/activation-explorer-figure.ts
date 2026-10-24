import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { linear, path, range } from './lib/plot';
import { ACTIVATIONS, Activation } from './lib/tiny-mlp';

const W = 340;
const H = 190;
const Z = 6;
const x = linear(-Z, Z, 30, W - 10);
const y = linear(-1.6, 3.2, H - 20, 10);
const CHOICES: { id: Activation; label: string }[] = [
  { id: 'relu', label: 'ReLU' },
  { id: 'tanh', label: 'tanh' },
  { id: 'logistic', label: 'sigmoid' },
  { id: 'identity', label: 'identity' },
];

/** Each activation's curve, a point on it and its slope, and how slopes compound over layers. */
@Component({
  selector: 'app-activation-explorer-figure',
  imports: [DecimalPipe, FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Activation functions and their slopes"
      hint="Pick an activation, then slide z: the orange line is the slope backpropagation uses."
    >
      <div class="flex flex-wrap gap-2" role="group" aria-label="Activation">
        @for (choice of choices; track choice.id) {
          <button
            type="button"
            class="rounded-md border px-2.5 py-1 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
            [class.bg-muted]="activation() === choice.id"
            [attr.aria-pressed]="activation() === choice.id"
            (click)="activation.set(choice.id)"
          >
            {{ choice.label }}
          </button>
        }
      </div>
      <svg
        [attr.viewBox]="'0 0 ' + w + ' ' + h"
        class="mt-3 w-full"
        role="img"
        [attr.aria-label]="description()"
      >
        <line
          [attr.x1]="x(-z)"
          [attr.x2]="x(z)"
          [attr.y1]="y(0)"
          [attr.y2]="y(0)"
          class="stroke-border"
        />
        <line
          [attr.x1]="x(0)"
          [attr.x2]="x(0)"
          [attr.y1]="y(-1.6)"
          [attr.y2]="y(3.2)"
          class="stroke-border"
        />
        <text
          [attr.x]="w - 12"
          [attr.y]="y(0) - 4"
          text-anchor="end"
          class="fill-muted-foreground text-[9px]"
        >
          z
        </text>
        <path [attr.d]="curve()" class="fill-none stroke-fig-1" stroke-width="2.5" />
        <path [attr.d]="tangent()" class="fill-none stroke-fig-2" stroke-width="2" />
        <circle [attr.cx]="x(point())" [attr.cy]="y(output())" r="5" class="fill-fig-2" />
        <text x="34" y="20" class="fill-fig-1 text-[10px] font-semibold">f(z)</text>
      </svg>
      <app-figure-slider
        class="mt-2"
        label="Weighted sum z"
        [(value)]="point"
        [min]="-6"
        [max]="6"
        [step]="0.1"
        [readout]="(point() | number: '1.1-1') ?? ''"
      />
      <dl class="mt-3 grid grid-cols-2 gap-2 text-xs sm:grid-cols-3">
        <div class="rounded-md bg-muted/50 p-2">
          <dt class="text-muted-foreground">output f(z)</dt>
          <dd class="font-mono text-sm tabular-nums">{{ output() | number: '1.3-3' }}</dd>
        </div>
        <div class="rounded-md bg-muted/50 p-2">
          <dt class="text-muted-foreground">slope f′(z)</dt>
          <dd class="font-mono text-sm tabular-nums">{{ slope() | number: '1.3-3' }}</dd>
        </div>
        <div class="col-span-2 rounded-md bg-muted/50 p-2 sm:col-span-1">
          <dt class="text-muted-foreground">gradient left after {{ depth() }} such layers</dt>
          <dd class="font-mono text-sm tabular-nums">{{ compounded() | number: '1.0-6' }}×</dd>
        </div>
      </dl>
      <app-figure-slider
        class="mt-2"
        label="Layers to backpropagate through"
        [(value)]="depth"
        [min]="1"
        [max]="10"
      />
      <p caption>
        Backpropagation multiplies by one slope per layer. Where the curve is flat the slope is near
        0, and the gradient fades layer after layer: sigmoid's slope is never above 0.25, so ten
        sigmoid layers keep less than a millionth of it. ReLU's slope is exactly 1 wherever it is
        on, which is why it became the default. But slide z below 0: an "off" ReLU passes no
        gradient at all.
      </p>
    </app-figure-frame>
  `,
})
export class ActivationExplorerFigure {
  protected readonly w = W;
  protected readonly h = H;
  protected readonly z = Z;
  protected readonly x = x;
  protected readonly y = y;
  protected readonly choices = CHOICES;
  protected readonly activation = signal<Activation>('relu');
  protected readonly point = signal(1);
  protected readonly depth = signal(5);

  private readonly fn = computed(() => ACTIVATIONS[this.activation()]);
  protected readonly output = computed(() => this.fn().f(this.point()));
  protected readonly slope = computed(() => this.fn().slope(this.output()));
  protected readonly compounded = computed(() => this.slope() ** this.depth());

  protected readonly curve = computed(() => {
    const f = this.fn().f;
    return path(range(-Z, Z, 121).map((z) => [x(z), y(clamp(f(z)))] as const));
  });

  /** A short segment through the point with the curve's slope. */
  protected readonly tangent = computed(() => {
    const [z0, a, s] = [this.point(), this.output(), this.slope()];
    return path([-1.2, 1.2].map((d) => [x(z0 + d), y(clamp(a + s * d))] as const));
  });

  protected readonly description = computed(
    () =>
      `Plot of ${this.activation()}: at z = ${this.point().toFixed(1)} the output is ` +
      `${this.output().toFixed(3)} and the slope ${this.slope().toFixed(3)}.`,
  );
}

function clamp(value: number): number {
  return Math.min(Math.max(value, -1.6), 3.2);
}
