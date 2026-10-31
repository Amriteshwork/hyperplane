import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { Weights, accuracy, flipLabels, nearest, neighbourMean } from './lib/knn';
import { linear } from './lib/plot';
import { seeded } from './lib/random';
import { ToyDataset, toyData } from './lib/toy-data';

const SIZE = 300;
const CELLS = 30;
const ROWS = 150;
const LABEL_NOISE = 0.15;
const LOW = -1.1;
const HIGH = 1.1;
const s = linear(LOW, HIGH, 0, SIZE);
const CELL = SIZE / CELLS;
/** Cell centres, row by row, in data coordinates. */
const CENTRES = Array.from({ length: CELLS * CELLS }, (_, i) => {
  const [col, row] = [i % CELLS, Math.floor(i / CELLS)];
  return [LOW + ((HIGH - LOW) * (col + 0.5)) / CELLS, HIGH - ((HIGH - LOW) * (row + 0.5)) / CELLS];
});

const ARROWS: Readonly<Record<string, readonly [number, number]>> = {
  ArrowLeft: [-0.05, 0],
  ArrowRight: [0.05, 0],
  ArrowUp: [0, 0.05],
  ArrowDown: [0, -0.05],
};

const DATASETS: { id: ToyDataset; label: string }[] = [
  { id: 'moons', label: 'Moons' },
  { id: 'circle', label: 'Circle' },
  { id: 'xor', label: 'XOR' },
  { id: 'spiral', label: 'Spiral' },
];
const WEIGHTS: { id: Weights; label: string }[] = [
  { id: 'uniform', label: 'Equal votes' },
  { id: 'distance', label: 'Closer counts more' },
];

/** Move a query, change k and see who votes, how the boundary reshapes and what it does to accuracy. */
@Component({
  selector: 'app-knn-playground-figure',
  imports: [DecimalPipe, FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="k-nearest neighbours playground"
      hint="Click the plot or use its arrow keys to move the query, then change k."
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
              [attr.fill-opacity]="0.08 + 0.5 * abs(p - 0.5) * 2"
            />
          }
          @for (p of train().X; track $index) {
            <circle
              [attr.cx]="sx(p[0])"
              [attr.cy]="sx(-p[1])"
              r="3"
              [class]="labels()[$index] ? 'fill-fig-1 stroke-card' : 'fill-fig-2 stroke-card'"
              stroke-width="1"
            />
          }
          @for (n of neighbours(); track n.index) {
            <line
              [attr.x1]="sx(queryX())"
              [attr.y1]="sx(-queryY())"
              [attr.x2]="sx(train().X[n.index][0])"
              [attr.y2]="sx(-train().X[n.index][1])"
              class="stroke-foreground"
              stroke-width="0.8"
            />
          }
          <circle
            [attr.cx]="sx(queryX())"
            [attr.cy]="sx(-queryY())"
            r="5"
            class="fill-card stroke-foreground"
            stroke-width="2"
          />
        </svg>

        <div class="space-y-3">
          <div>
            <p class="text-xs text-muted-foreground">Dataset</p>
            <div class="mt-1 flex flex-wrap gap-1.5" role="group" aria-label="Dataset">
              @for (d of datasets; track d.id) {
                <button
                  type="button"
                  [class]="chip"
                  [class.bg-muted]="dataset() === d.id"
                  [attr.aria-pressed]="dataset() === d.id"
                  (click)="dataset.set(d.id)"
                >
                  {{ d.label }}
                </button>
              }
            </div>
          </div>
          <app-figure-slider label="Neighbours (k)" [(value)]="k" [min]="1" [max]="rows" />
          <div>
            <p class="text-xs text-muted-foreground">Votes</p>
            <div class="mt-1 flex flex-wrap gap-1.5" role="group" aria-label="Votes">
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
          <dl class="grid grid-cols-3 gap-2 text-xs" aria-live="polite">
            <div>
              <dt class="text-muted-foreground">P(blue) here</dt>
              <dd class="font-mono tabular-nums" data-testid="query-p">
                {{ queryP() | number: '1.2-2' }}
              </dd>
            </div>
            <div>
              <dt class="text-muted-foreground">training rows</dt>
              <dd class="font-mono tabular-nums" data-testid="train-score">
                {{ 100 * scores().train | number: '1.0-0' }}%
              </dd>
            </div>
            <div>
              <dt class="text-muted-foreground">new rows</dt>
              <dd class="font-mono tabular-nums" data-testid="test-score">
                {{ 100 * scores().test | number: '1.0-0' }}%
              </dd>
            </div>
          </dl>
        </div>
      </div>
      <p caption>
        The shading is the share of each spot's k neighbours that are blue, and the lines join the
        query to the rows voting on it. {{ noisePct }}% of the training labels were flipped on
        purpose, so there is noise to learn. At <strong>k = 1</strong> the boundary wraps each
        flipped point in its own island: the training score is a perfect 100% while new rows suffer.
        Raise k and the islands melt away and the score on new rows climbs. Keep going and the
        boundary blurs until, at k = {{ rows }}, every spot gets the same answer.
      </p>
    </app-figure-frame>
  `,
})
export class KnnPlaygroundFigure {
  protected readonly size = SIZE;
  protected readonly cells = CELLS;
  protected readonly cell = CELL;
  protected readonly rows = ROWS;
  protected readonly noisePct = 100 * LABEL_NOISE;
  protected readonly sx = s;
  protected readonly datasets = DATASETS;
  protected readonly weightings = WEIGHTS;
  protected readonly floor = Math.floor;
  protected readonly abs = Math.abs;
  protected readonly chip =
    'rounded-md border px-2.5 py-1 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring';

  protected readonly dataset = signal<ToyDataset>('moons');
  protected readonly k = signal(1);
  protected readonly weights = signal<Weights>('uniform');
  protected readonly queryX = signal(0.2);
  protected readonly queryY = signal(0.1);

  protected readonly train = computed(() => toyData(this.dataset(), seeded(7), ROWS));
  protected readonly labels = computed(() => flipLabels(this.train().y, LABEL_NOISE, seeded(5)));
  private readonly test = computed(() => toyData(this.dataset(), seeded(11), ROWS));

  protected readonly neighbours = computed(() =>
    nearest(this.train().X, [this.queryX(), this.queryY()], this.k()),
  );
  protected readonly queryP = computed(() =>
    neighbourMean(this.neighbours(), this.labels(), this.weights()),
  );
  protected readonly shading = computed(() =>
    CENTRES.map((centre) =>
      neighbourMean(nearest(this.train().X, centre, this.k()), this.labels(), this.weights()),
    ),
  );
  protected readonly scores = computed(() => {
    const { X } = this.train();
    const test = this.test();
    return {
      train: accuracy(X, this.labels(), X, this.labels(), this.k(), this.weights()),
      test: accuracy(X, this.labels(), test.X, test.y, this.k(), this.weights()),
    };
  });

  protected readonly description = computed(
    () =>
      `Decision boundary of k-nearest neighbours with k = ${this.k()} on the ${this.dataset()} ` +
      `dataset: ${(100 * this.scores().train).toFixed(0)}% on training rows, ` +
      `${(100 * this.scores().test).toFixed(0)}% on new rows. The query is blue with ` +
      `probability ${this.queryP().toFixed(2)}.`,
  );

  protected nudge(event: KeyboardEvent): void {
    const [dx, dy] = ARROWS[event.key] ?? [0, 0];
    if (!dx && !dy) return;
    event.preventDefault();
    this.queryX.update((x) => Math.min(1, Math.max(-1, x + dx)));
    this.queryY.update((y) => Math.min(1, Math.max(-1, y + dy)));
  }

  protected place(event: MouseEvent): void {
    const box = (event.currentTarget as SVGSVGElement).getBoundingClientRect();
    if (!box.width) return;
    const scale = (HIGH - LOW) / box.width;
    this.queryX.set(LOW + (event.clientX - box.left) * scale);
    this.queryY.set(HIGH - (event.clientY - box.top) * scale);
  }
}
