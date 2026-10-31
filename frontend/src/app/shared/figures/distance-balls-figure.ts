import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { Metric, nearest } from './lib/knn';
import { linear } from './lib/plot';
import { seeded, uniform } from './lib/random';

const SIZE = 300;
const s = linear(-1.1, 1.1, 0, SIZE);
const PER_UNIT = SIZE / 2.2;
const rng = seeded(21);
const POINTS = Array.from({ length: 16 }, () => [uniform(rng, -1, 1), uniform(rng, -1, 1)]);
const METRICS: { id: Metric; label: string }[] = [
  { id: 'euclidean', label: 'Euclidean' },
  { id: 'manhattan', label: 'Manhattan' },
  { id: 'chebyshev', label: 'Chebyshev' },
];

/** The set of points at one distance, as an SVG path; `stretch` squeezes it along the first axis. */
function ball(metric: Metric, cx: number, cy: number, radius: number, stretch: number): string {
  const [rx, ry] = [(radius * PER_UNIT) / stretch, radius * PER_UNIT];
  if (metric === 'euclidean') {
    return `M${cx - rx},${cy}a${rx},${ry} 0 1,0 ${2 * rx},0a${rx},${ry} 0 1,0 ${-2 * rx},0Z`;
  }
  if (metric === 'manhattan') {
    return `M${cx - rx},${cy}L${cx},${cy - ry}L${cx + rx},${cy}L${cx},${cy + ry}Z`;
  }
  return `M${cx - rx},${cy - ry}L${cx + rx},${cy - ry}L${cx + rx},${cy + ry}L${cx - rx},${cy + ry}Z`;
}

/** Each metric has its own "circle", so each can pick a different nearest point. */
@Component({
  selector: 'app-distance-balls-figure',
  imports: [DecimalPipe, FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="What 'nearest' means"
      hint="Move the query, switch metric, then stretch the first feature."
    >
      <div class="grid gap-4 sm:grid-cols-[minmax(0,300px)_1fr]">
        <svg
          [attr.viewBox]="'0 0 ' + size + ' ' + size"
          class="w-full rounded-md border"
          role="img"
          [attr.aria-label]="description()"
        >
          <path [attr.d]="shape()" class="fill-fig-1 stroke-fig-1" fill-opacity="0.12" />
          @for (p of points; track $index) {
            <circle
              [attr.cx]="sx(p[0])"
              [attr.cy]="sx(-p[1])"
              [attr.r]="$index === winner().index ? 5 : 3.5"
              [class]="$index === winner().index ? 'fill-fig-2' : 'fill-muted-foreground'"
            />
          }
          <circle
            [attr.cx]="sx(queryX())"
            [attr.cy]="sx(-queryY())"
            r="4"
            class="fill-card stroke-foreground"
            stroke-width="2"
          />
        </svg>

        <div class="space-y-3">
          <div>
            <p class="text-xs text-muted-foreground">Metric</p>
            <div class="mt-1 flex flex-wrap gap-1.5" role="group" aria-label="Metric">
              @for (m of metrics; track m.id) {
                <button
                  type="button"
                  [class]="chip"
                  [class.bg-muted]="metric() === m.id"
                  [attr.aria-pressed]="metric() === m.id"
                  (click)="metric.set(m.id)"
                >
                  {{ m.label }}
                </button>
              }
            </div>
          </div>
          <app-figure-slider
            label="Query, first feature"
            [(value)]="queryX"
            [min]="-1"
            [max]="1"
            [step]="0.01"
          />
          <app-figure-slider
            label="Query, second feature"
            [(value)]="queryY"
            [min]="-1"
            [max]="1"
            [step]="0.01"
          />
          <app-figure-slider
            label="Stretch the first feature (bad units)"
            [(value)]="stretch"
            [min]="1"
            [max]="20"
            [readout]="'× ' + stretch()"
          />
          <dl class="grid grid-cols-2 gap-2 text-xs" aria-live="polite">
            <div>
              <dt class="text-muted-foreground">distance to the nearest point</dt>
              <dd class="font-mono tabular-nums" data-testid="nearest-distance">
                {{ winner().distance | number: '1.3-3' }}
              </dd>
            </div>
            <div>
              <dt class="text-muted-foreground">metrics agreeing on it</dt>
              <dd class="font-mono tabular-nums" data-testid="agreement">{{ agreement() }} of 3</dd>
            </div>
          </dl>
        </div>
      </div>
      <p caption>
        The shape is every location exactly as far away as the nearest point (orange). Euclidean
        draws a circle, Manhattan a diamond, Chebyshev a square; move the query until two of them
        pick different points. Now stretch the first feature, as if it were measured in smaller
        units. The shape collapses into a sliver: any difference in that feature costs so much that
        it alone chooses the neighbour, and the other feature hardly counts. That is why you scale
        features first.
      </p>
    </app-figure-frame>
  `,
})
export class DistanceBallsFigure {
  protected readonly size = SIZE;
  protected readonly sx = s;
  protected readonly points = POINTS;
  protected readonly metrics = METRICS;
  protected readonly chip =
    'rounded-md border px-2.5 py-1 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring';

  protected readonly metric = signal<Metric>('euclidean');
  protected readonly queryX = signal(0);
  protected readonly queryY = signal(0);
  protected readonly stretch = signal(1);

  private readonly query = computed(() => [this.queryX(), this.queryY()]);

  protected readonly winner = computed(
    () => nearest(POINTS, this.query(), 1, this.metric(), this.stretch())[0],
  );
  protected readonly agreement = computed(
    () =>
      METRICS.filter(
        (m) =>
          nearest(POINTS, this.query(), 1, m.id, this.stretch())[0].index === this.winner().index,
      ).length,
  );
  protected readonly shape = computed(() =>
    ball(
      this.metric(),
      s(this.queryX()),
      s(-this.queryY()),
      this.winner().distance,
      this.stretch(),
    ),
  );
  protected readonly description = computed(
    () =>
      `Under the ${this.metric()} distance with the first feature stretched ${this.stretch()} ` +
      `times, the nearest point is ${this.winner().distance.toFixed(3)} away.`,
  );
}
