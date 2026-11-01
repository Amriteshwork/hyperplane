import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { correlatedBlobs, fitGaussians, logNormal, predictProba } from './lib/naive-bayes';
import { linear } from './lib/plot';
import { seeded } from './lib/random';

const SIZE = 300;
const CELLS = 30;
const CELL = SIZE / CELLS;
const LOW = -4;
const HIGH = 4;
const ROWS = 160;
const CENTRES: readonly (readonly [number, number])[] = [
  [-0.8, -0.8],
  [0.8, 0.8],
];
const s = linear(LOW, HIGH, 0, SIZE);
const sy = linear(LOW, HIGH, SIZE, 0);
/** Cell centres, row by row, in data coordinates. */
const CENTRE_POINTS = Array.from({ length: CELLS * CELLS }, (_, i) => {
  const [col, row] = [i % CELLS, Math.floor(i / CELLS)];
  return [LOW + ((HIGH - LOW) * (col + 0.5)) / CELLS, HIGH - ((HIGH - LOW) * (row + 0.5)) / CELLS];
});
const ARROWS: Readonly<Record<string, readonly [number, number]>> = {
  ArrowLeft: [-0.2, 0],
  ArrowRight: [0.2, 0],
  ArrowUp: [0, 0.2],
  ArrowDown: [0, -0.2],
};
const CLASS_FILL = ['fill-fig-2', 'fill-fig-1'];
const CLASS_STROKE = ['stroke-fig-2', 'stroke-fig-1'];

/** Gaussian NB's axis-aligned curves against data that may not be: labels hold, confidence doesn't. */
@Component({
  selector: 'app-naive-bayes-playground-figure',
  imports: [FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Gaussian Naive Bayes playground"
      hint="Click the plot or use its arrow keys to move the query, then raise the correlation."
    >
      <div class="grid gap-4 sm:grid-cols-[minmax(0,300px)_1fr]">
        <svg
          [attr.viewBox]="'0 0 ' + size + ' ' + size"
          class="w-full cursor-crosshair rounded-md border"
          role="img"
          [attr.aria-label]="description()"
          tabindex="0"
          (click)="place($event)"
          (keydown)="nudge($event)"
        >
          @for (p of shading(); track $index) {
            <rect
              [attr.x]="($index % cells) * cell"
              [attr.y]="floor($index / cells) * cell"
              [attr.width]="cell"
              [attr.height]="cell"
              shape-rendering="crispEdges"
              [class]="p > 0.5 ? 'fill-fig-1' : 'fill-fig-2'"
              [attr.fill-opacity]="0.04 + 0.45 * abs(p - 0.5) * 2"
            />
          }
          @for (p of train().X; track $index) {
            <circle
              [attr.cx]="sx(p[0])"
              [attr.cy]="sy(p[1])"
              r="2.5"
              [class]="classFill[train().y[$index]] + ' stroke-card'"
              stroke-width="0.8"
            />
          }
          @for (e of ellipses(); track $index) {
            <ellipse
              [attr.cx]="e.cx"
              [attr.cy]="e.cy"
              [attr.rx]="e.rx"
              [attr.ry]="e.ry"
              fill="none"
              [class]="classStroke[$index]"
              stroke-width="2"
            />
          }
          <circle
            [attr.cx]="sx(queryX())"
            [attr.cy]="sy(queryY())"
            r="5"
            class="fill-card stroke-foreground"
            stroke-width="2"
          />
        </svg>

        <div class="space-y-3">
          <app-figure-slider
            label="Correlation within each class"
            [(value)]="correlation"
            [min]="0"
            [max]="0.95"
            [step]="0.05"
            [readout]="correlation().toFixed(2)"
          />
          <div>
            <p class="text-xs text-muted-foreground">
              Log likelihood gap at the query (blue − orange)
            </p>
            <dl class="mt-1 grid grid-cols-2 gap-2 text-xs">
              <div>
                <dt class="text-muted-foreground">first feature</dt>
                <dd class="font-mono tabular-nums">{{ gaps()[0].toFixed(2) }}</dd>
              </div>
              <div>
                <dt class="text-muted-foreground">second feature</dt>
                <dd class="font-mono tabular-nums">{{ gaps()[1].toFixed(2) }}</dd>
              </div>
            </dl>
          </div>
          <dl class="grid grid-cols-3 gap-2 text-xs" aria-live="polite">
            <div>
              <dt class="text-muted-foreground">P(blue) here</dt>
              <dd class="font-mono tabular-nums">
                <span data-testid="query-p">{{ (100 * queryP()).toFixed(1) }}</span
                >%
              </dd>
            </div>
            <div>
              <dt class="text-muted-foreground">avg confidence</dt>
              <dd class="font-mono tabular-nums">
                <span data-testid="confidence">{{ (100 * scores().confidence).toFixed(0) }}</span
                >%
              </dd>
            </div>
            <div>
              <dt class="text-muted-foreground">accuracy, new rows</dt>
              <dd class="font-mono tabular-nums">
                <span data-testid="accuracy">{{ (100 * scores().accuracy).toFixed(0) }}</span
                >%
              </dd>
            </div>
          </dl>
        </div>
      </div>
      <p caption>
        The rings show what the model fits: a 2σ ellipse for each class, always upright, because it
        treats the features separately. The shading is its probability of blue, and the two gaps add
        up to its verdict at the query. Raise the correlation and the clouds tilt along the line
        joining them, but the rings can't follow. Each feature now repeats the other, so the same
        evidence counts twice: confidence climbs above accuracy.
      </p>
    </app-figure-frame>
  `,
})
export class NaiveBayesPlaygroundFigure {
  protected readonly size = SIZE;
  protected readonly cells = CELLS;
  protected readonly cell = CELL;
  protected readonly sx = s;
  protected readonly sy = sy;
  protected readonly floor = Math.floor;
  protected readonly abs = Math.abs;
  protected readonly classFill = CLASS_FILL;
  protected readonly classStroke = CLASS_STROKE;

  protected readonly correlation = signal(0);
  protected readonly queryX = signal(0.5);
  protected readonly queryY = signal(-0.3);

  protected readonly train = computed(() =>
    correlatedBlobs(seeded(9), ROWS, CENTRES, this.correlation()),
  );
  private readonly model = computed(() => fitGaussians(this.train().X, this.train().y, 2));

  protected readonly shading = computed(() =>
    CENTRE_POINTS.map((point) => predictProba(this.model(), point)[1]),
  );
  protected readonly ellipses = computed(() => {
    const { means, variances } = this.model();
    const unit = SIZE / (HIGH - LOW);
    return means.map((mean, c) => ({
      cx: s(mean[0]),
      cy: sy(mean[1]),
      rx: 2 * Math.sqrt(variances[c][0]) * unit,
      ry: 2 * Math.sqrt(variances[c][1]) * unit,
    }));
  });
  protected readonly queryP = computed(
    () => predictProba(this.model(), [this.queryX(), this.queryY()])[1],
  );
  protected readonly gaps = computed(() => {
    const { means, variances } = this.model();
    return [this.queryX(), this.queryY()].map(
      (x, j) =>
        logNormal(x, means[1][j], variances[1][j]) - logNormal(x, means[0][j], variances[0][j]),
    );
  });
  protected readonly scores = computed(() => {
    const test = correlatedBlobs(seeded(21), 2000, CENTRES, this.correlation());
    const probabilities = test.X.map((row) => predictProba(this.model(), row)[1]);
    const accuracy =
      probabilities.filter((p, i) => (p > 0.5 ? 1 : 0) === test.y[i]).length / probabilities.length;
    const confidence =
      probabilities.reduce((sum, p) => sum + Math.max(p, 1 - p), 0) / probabilities.length;
    return { accuracy, confidence };
  });

  protected readonly description = computed(
    () =>
      `Gaussian Naive Bayes on two classes with within-class correlation ` +
      `${this.correlation().toFixed(2)}: accuracy ${(100 * this.scores().accuracy).toFixed(0)}%, ` +
      `average confidence ${(100 * this.scores().confidence).toFixed(0)}%. The query is blue with ` +
      `probability ${this.queryP().toFixed(2)}.`,
  );

  protected nudge(event: KeyboardEvent): void {
    const [dx, dy] = ARROWS[event.key] ?? [0, 0];
    if (!dx && !dy) return;
    event.preventDefault();
    this.queryX.update((x) => Math.min(HIGH, Math.max(LOW, x + dx)));
    this.queryY.update((y) => Math.min(HIGH, Math.max(LOW, y + dy)));
  }

  protected place(event: MouseEvent): void {
    const box = (event.currentTarget as SVGSVGElement).getBoundingClientRect();
    if (!box.width) return;
    const scale = (HIGH - LOW) / box.width;
    this.queryX.set(LOW + (event.clientX - box.left) * scale);
    this.queryY.set(HIGH - (event.clientY - box.top) * scale);
  }
}
