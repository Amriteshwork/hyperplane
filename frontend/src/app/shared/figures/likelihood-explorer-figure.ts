import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { logNormal } from './lib/naive-bayes';
import { linear, path, range } from './lib/plot';
import { normal, seeded } from './lib/random';

const WIDTH = 460;
const HEIGHT = 200;
const LOW = 0;
const HIGH = 10;
const sx = linear(LOW, HIGH, 0, WIDTH);
const sy = linear(0, 0.8, HEIGHT - 14, 6);

const rng = seeded(3);
const DATA = Array.from({ length: 14 }, () => 6 + normal(rng));
const MEAN = DATA.reduce((sum, x) => sum + x, 0) / DATA.length;
const VARIANCE = DATA.reduce((sum, x) => sum + (x - MEAN) ** 2, 0) / DATA.length;
const GRID = range(LOW, HIGH, 201);

const logLikelihood = (mean: number, variance: number) =>
  DATA.reduce((sum, x) => sum + logNormal(x, mean, variance), 0);

/** Fix the data and move the curve: its fit is the sum of the log heights at the data. */
@Component({
  selector: 'app-likelihood-explorer-figure',
  imports: [FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Find the most likely curve"
      hint="Move the curve over the fixed data."
    >
      <svg
        [attr.viewBox]="'0 0 ' + width + ' ' + height"
        class="w-full rounded-md border"
        role="img"
        [attr.aria-label]="description()"
      >
        @for (x of data; track $index) {
          <line
            [attr.x1]="sx(x)"
            [attr.x2]="sx(x)"
            [attr.y1]="sy(0)"
            [attr.y2]="sy(height_(x))"
            class="stroke-fig-2"
            stroke-width="1.2"
          />
          <circle [attr.cx]="sx(x)" [attr.cy]="sy(0)" r="3" class="fill-fig-2" />
        }
        <path [attr.d]="curve()" fill="none" class="stroke-fig-1" stroke-width="2" />
      </svg>
      <div class="mt-3 grid gap-3 sm:grid-cols-2">
        <app-figure-slider
          label="Centre (μ)"
          [(value)]="mean"
          [min]="1"
          [max]="9"
          [step]="0.05"
          [readout]="mean().toFixed(2)"
        />
        <app-figure-slider
          label="Spread (σ)"
          [(value)]="std"
          [min]="0.3"
          [max]="3"
          [step]="0.01"
          [readout]="std().toFixed(2)"
        />
      </div>
      <div class="mt-2 flex flex-wrap items-center gap-3 text-xs" aria-live="polite">
        <p>
          Log likelihood:
          <span class="font-mono tabular-nums" data-testid="log-likelihood">{{
            score().toFixed(2)
          }}</span>
          (the best possible is
          <span class="font-mono tabular-nums" data-testid="best-log-likelihood">{{
            best.toFixed(2)
          }}</span
          >)
        </p>
        <button
          type="button"
          class="rounded-md border px-2.5 py-1 hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
          (click)="fit()"
        >
          Jump to the best fit
        </button>
      </div>
      <p caption>
        The orange lines are the curve's height at each row, and the log likelihood adds up their
        logs. A narrow curve is tall, which rewards rows near its centre, but a single stray row
        costs a lot. The best fit is the data's own mean and standard deviation: no search needed.
      </p>
    </app-figure-frame>
  `,
})
export class LikelihoodExplorerFigure {
  protected readonly width = WIDTH;
  protected readonly height = HEIGHT;
  protected readonly data = DATA;
  protected readonly sx = sx;
  protected readonly sy = (y: number) => sy(Math.min(y, 0.8));
  protected readonly best = logLikelihood(MEAN, VARIANCE);

  protected readonly mean = signal(3);
  protected readonly std = signal(2);

  protected readonly score = computed(() => logLikelihood(this.mean(), this.std() ** 2));
  protected readonly curve = computed(() =>
    path(GRID.map((x) => [sx(x), this.sy(this.height_(x))])),
  );
  protected readonly description = computed(
    () =>
      `A normal curve with centre ${this.mean().toFixed(2)} and spread ${this.std().toFixed(2)} ` +
      `over ${DATA.length} data points has log likelihood ${this.score().toFixed(2)}; the best is ` +
      `${this.best.toFixed(2)}.`,
  );

  protected height_(x: number): number {
    return Math.exp(logNormal(x, this.mean(), this.std() ** 2));
  }

  protected fit(): void {
    this.mean.set(MEAN);
    this.std.set(Math.sqrt(VARIANCE));
  }
}
