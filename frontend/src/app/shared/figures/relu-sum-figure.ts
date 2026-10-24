import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { leastSquares } from './lib/least-squares';
import { linear, path, range } from './lib/plot';

const W = 340;
const H = 200;
const X0 = -3;
const X1 = 3;
const x = linear(X0, X1, 10, W - 10);
const y = linear(-1.6, 1.6, H - 10, 10);
const GRID = range(X0, X1, 121);
const target = (v: number) => Math.sin(2 * v);
const PIECE_CLASSES = ['stroke-fig-2', 'stroke-fig-3', 'stroke-fig-4', 'stroke-fig-negative'];

/** One hidden layer of ReLUs: each neuron adds one bent line, and together they trace a curve. */
@Component({
  selector: 'app-relu-sum-figure',
  imports: [DecimalPipe, FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Bent lines add up to a curve"
      hint="Add hidden neurons and watch the network's output (blue) close in on the target."
    >
      <svg
        [attr.viewBox]="'0 0 ' + w + ' ' + h"
        class="w-full"
        role="img"
        [attr.aria-label]="description()"
      >
        <line
          [attr.x1]="x(x0)"
          [attr.x2]="x(x1)"
          [attr.y1]="y(0)"
          [attr.y2]="y(0)"
          class="stroke-border"
        />
        @if (showPieces()) {
          @for (piece of pieces(); track $index; let i = $index) {
            <path
              [attr.d]="piece"
              [class]="'fill-none ' + pieceClass(i)"
              stroke-width="1"
              stroke-opacity="0.7"
            />
          }
        }
        <path
          [attr.d]="targetPath"
          class="fill-none stroke-muted-foreground"
          stroke-width="2"
          stroke-dasharray="5 4"
        />
        <path [attr.d]="fitPath()" class="fill-none stroke-fig-1" stroke-width="2.5" />
        @for (kink of kinks(); track $index) {
          <circle [attr.cx]="x(kink)" [attr.cy]="y(fitAt(kink))" r="2.5" class="fill-fig-1" />
        }
      </svg>
      <div class="mt-2 grid gap-3 sm:grid-cols-[1fr_auto] sm:items-end">
        <app-figure-slider label="Hidden ReLU neurons" [(value)]="neurons" [min]="1" [max]="16" />
        <label class="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            class="accent-fig-1"
            [checked]="showPieces()"
            (change)="showPieces.set(!showPieces())"
          />
          show each neuron's piece
        </label>
      </div>
      <p class="mt-2 text-xs text-muted-foreground">
        Mean squared error to the target:
        <span class="font-mono tabular-nums text-foreground">{{ error() | number: '1.4-4' }}</span>
      </p>
      <p caption>
        Each hidden neuron computes max(0, x − kink): flat, then a straight rise from its kink (the
        dots). The output layer scales each piece and adds them up, so the blue line can change
        direction at every kink. One neuron gives one bend; a dozen trace the dashed sine wave
        closely. That is why more hidden neurons can fit more complicated shapes. Here the kinks are
        spread evenly and only the output weights are fitted; training a real network moves the
        kinks too.
      </p>
    </app-figure-frame>
  `,
})
export class ReluSumFigure {
  protected readonly w = W;
  protected readonly h = H;
  protected readonly x0 = X0;
  protected readonly x1 = X1;
  protected readonly x = x;
  protected readonly y = y;
  protected readonly neurons = signal(3);
  protected readonly showPieces = signal(true);
  protected readonly targetPath = path(GRID.map((v) => [x(v), y(target(v))] as const));

  /** Kinks spread evenly over the range, the first at its left edge. */
  protected readonly kinks = computed(() => {
    const k = this.neurons();
    return Array.from({ length: k }, (_, i) => X0 + ((X1 - X0) * i) / k);
  });

  /** Output weights (and bias) that best fit the target, by least squares. */
  private readonly weights = computed(() => {
    const kinks = this.kinks();
    const features = GRID.map((v) => [1, ...kinks.map((k) => Math.max(0, v - k))]);
    return leastSquares(features, GRID.map(target));
  });

  protected fitAt(v: number): number {
    const [bias, ...scales] = this.weights();
    return this.kinks().reduce((s, k, i) => s + scales[i] * Math.max(0, v - k), bias);
  }

  protected readonly fitPath = computed(() =>
    path(GRID.map((v) => [x(v), y(clamp(this.fitAt(v)))] as const)),
  );

  protected readonly pieces = computed(() => {
    const scales = this.weights().slice(1);
    return this.kinks().map((k, i) =>
      path(GRID.map((v) => [x(v), y(clamp(scales[i] * Math.max(0, v - k)))] as const)),
    );
  });

  protected readonly error = computed(
    () => GRID.reduce((s, v) => s + (this.fitAt(v) - target(v)) ** 2, 0) / GRID.length,
  );

  protected readonly description = computed(
    () =>
      `A sum of ${this.neurons()} ReLU pieces approximating a sine wave; ` +
      `mean squared error ${this.error().toFixed(4)}.`,
  );

  protected pieceClass(i: number): string {
    return PIECE_CLASSES[i % PIECE_CLASSES.length];
  }
}

function clamp(value: number): number {
  return Math.min(Math.max(value, -1.6), 1.6);
}
