import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { Weights, nearest, neighbourMean } from './lib/knn';
import { linear, path, range } from './lib/plot';
import { normal, seeded, uniform } from './lib/random';

const WIDTH = 460;
const HEIGHT = 240;
const ROWS = 60;
const sx = linear(-3.2, 3.2, 0, WIDTH);
const sy = linear(-2.1, 2.1, HEIGHT, 0);

const rng = seeded(4);
const X = Array.from({ length: ROWS }, () => [uniform(rng, -3, 3)]);
const Y = X.map(([x]) => Math.sin(x) + 0.35 * normal(rng));
const GRID = range(-3.2, 3.2, 321);
const TRUE_CURVE = path(GRID.map((x) => [sx(x), sy(Math.sin(x))]));
const WEIGHTS: { id: Weights; label: string }[] = [
  { id: 'uniform', label: 'Equal say' },
  { id: 'distance', label: 'Closer counts more' },
];

/** Regression as a local average: k sets how wide each average reaches. */
@Component({
  selector: 'app-knn-regression-figure',
  imports: [DecimalPipe, FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame heading="A fit made of averages" hint="Slide k from 1 up to every row.">
      <svg
        [attr.viewBox]="'0 0 ' + width + ' ' + height"
        class="w-full rounded-md border"
        role="img"
        [attr.aria-label]="description()"
      >
        <path [attr.d]="trueCurve" fill="none" class="stroke-fig-3" stroke-width="1.5" />
        @for (p of points; track $index) {
          <circle [attr.cx]="sx(p[0])" [attr.cy]="sy(ys[$index])" r="2.8" class="fill-fig-2" />
        }
        <path [attr.d]="fit().line" fill="none" class="stroke-fig-1" stroke-width="2" />
      </svg>
      <div class="mt-3 grid gap-3 sm:grid-cols-2">
        <app-figure-slider label="Neighbours (k)" [(value)]="k" [min]="1" [max]="rows" />
        <div>
          <p class="text-xs text-muted-foreground">Neighbours' say</p>
          <div class="mt-1 flex flex-wrap gap-1.5" role="group" aria-label="Neighbours' say">
            @for (w of weightings; track w.id) {
              <button
                type="button"
                [class]="chip"
                [class.bg-muted]="weights() === w.id"
                [attr.aria-pressed]="weights() === w.id"
                (click)="weights.set(w.id)"
              >
                {{ w.label }}
              </button>
            }
          </div>
        </div>
      </div>
      <p class="mt-2 text-xs" aria-live="polite">
        Error against the true curve (MSE):
        <span class="font-mono tabular-nums" data-testid="mse">{{
          fit().mse | number: '1.3-3'
        }}</span>
      </p>
      <p caption>
        Orange points are noisy samples of the green curve; blue is the prediction. At
        <strong>k = 1</strong> the fit jumps through every point, noise included. Around k = 10 it
        tracks the curve closely. At k = {{ rows }} it is one flat line, the mean of all the
        targets. Notice the edges too: the fit flattens beyond the last point, because an average of
        training targets can't go past them.
      </p>
    </app-figure-frame>
  `,
})
export class KnnRegressionFigure {
  protected readonly width = WIDTH;
  protected readonly height = HEIGHT;
  protected readonly rows = ROWS;
  protected readonly sx = sx;
  protected readonly sy = sy;
  protected readonly points = X;
  protected readonly ys = Y;
  protected readonly trueCurve = TRUE_CURVE;
  protected readonly weightings = WEIGHTS;
  protected readonly chip =
    'rounded-md border px-2.5 py-1 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring';

  protected readonly k = signal(1);
  protected readonly weights = signal<Weights>('uniform');

  protected readonly fit = computed(() => {
    const predictions = GRID.map((x) =>
      neighbourMean(nearest(X, [x], this.k()), Y, this.weights()),
    );
    const squared = GRID.map((x, i) => (predictions[i] - Math.sin(x)) ** 2);
    return {
      line: path(GRID.map((x, i) => [sx(x), sy(predictions[i])])),
      mse: squared.reduce((sum, v) => sum + v, 0) / GRID.length,
    };
  });

  protected readonly description = computed(
    () =>
      `k-nearest neighbours regression with k = ${this.k()} on ${ROWS} noisy samples of a sine ` +
      `curve; mean squared error against the curve is ${this.fit().mse.toFixed(3)}.`,
  );
}
