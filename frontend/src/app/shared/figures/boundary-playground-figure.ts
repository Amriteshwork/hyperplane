import { DecimalPipe } from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  computed,
  inject,
  signal,
} from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { linear } from './lib/plot';
import { seeded } from './lib/random';
import { ToyDataset, toyData } from './lib/toy-data';
import { Activation, Adam, Network, createNetwork, gradients, loss, predict } from './lib/tiny-mlp';

const SIZE = 300;
const CELLS = 25;
const PER_FRAME = 5;
const MAX_EPOCHS = 3000;
const s = linear(-1.1, 1.1, 0, SIZE);
const CELL = SIZE / CELLS;
/** Cell centres, row by row, in data coordinates. */
const CENTRES = Array.from({ length: CELLS * CELLS }, (_, i) => {
  const [col, row] = [i % CELLS, Math.floor(i / CELLS)];
  return [-1.1 + (2.2 * (col + 0.5)) / CELLS, 1.1 - (2.2 * (row + 0.5)) / CELLS];
});

const DATASETS: { id: ToyDataset; label: string }[] = [
  { id: 'circle', label: 'Circle' },
  { id: 'xor', label: 'XOR' },
  { id: 'moons', label: 'Moons' },
  { id: 'spiral', label: 'Spiral' },
];
const ACTIVATIONS: { id: Activation; label: string }[] = [
  { id: 'relu', label: 'ReLU' },
  { id: 'tanh', label: 'tanh' },
  { id: 'logistic', label: 'sigmoid' },
  { id: 'identity', label: 'identity (linear)' },
];

/** Train a small classifier in the page and watch its decision boundary bend around the data. */
@Component({
  selector: 'app-boundary-playground-figure',
  imports: [DecimalPipe, FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Decision-boundary playground"
      hint="Pick a dataset and a network, press Train, and watch the boundary bend."
    >
      <div class="grid gap-4 sm:grid-cols-[minmax(0,300px)_1fr]">
        <svg
          [attr.viewBox]="'0 0 ' + size + ' ' + size"
          class="w-full rounded-md border"
          role="img"
          [attr.aria-label]="description()"
        >
          @for (p of shading(); track $index) {
            <rect
              [attr.x]="($index % cells) * cell"
              [attr.y]="floor($index / cells) * cell"
              [attr.width]="cell + 0.5"
              [attr.height]="cell + 0.5"
              [class]="p > 0.5 ? 'fill-fig-1' : 'fill-fig-2'"
              [attr.fill-opacity]="0.08 + 0.5 * abs(p - 0.5) * 2"
            />
          }
          @for (p of data().X; track $index) {
            <circle
              [attr.cx]="sx(p[0])"
              [attr.cy]="sx(-p[1])"
              r="3.2"
              [class]="data().y[$index] ? 'fill-fig-1 stroke-card' : 'fill-fig-2 stroke-card'"
              stroke-width="1"
            />
          }
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
                  (click)="choose(d.id)"
                >
                  {{ d.label }}
                </button>
              }
            </div>
          </div>
          <div>
            <p class="text-xs text-muted-foreground">Activation</p>
            <div class="mt-1 flex flex-wrap gap-1.5" role="group" aria-label="Activation">
              @for (a of activations; track a.id) {
                <button
                  type="button"
                  [class]="chip"
                  [class.bg-muted]="activation() === a.id"
                  [attr.aria-pressed]="activation() === a.id"
                  (click)="activation.set(a.id); reset()"
                >
                  {{ a.label }}
                </button>
              }
            </div>
          </div>
          <app-figure-slider
            label="Hidden layers"
            [value]="layers()"
            (valueChange)="layers.set($event); reset()"
            [min]="1"
            [max]="3"
          />
          <app-figure-slider
            label="Neurons per layer"
            [value]="units()"
            (valueChange)="units.set($event); reset()"
            [min]="1"
            [max]="12"
          />
          <div class="flex flex-wrap gap-2">
            <button
              type="button"
              class="rounded-md bg-primary px-3 py-1.5 text-xs text-primary-foreground hover:opacity-90"
              (click)="toggle()"
            >
              {{ running() ? 'Pause' : epoch() ? 'Continue' : 'Train' }}
            </button>
            <button type="button" [class]="chip" (click)="reset()">Reset weights</button>
          </div>
          <dl class="grid grid-cols-3 gap-2 text-xs" aria-live="polite">
            <div>
              <dt class="text-muted-foreground">epoch</dt>
              <dd class="font-mono tabular-nums">{{ epoch() }}</dd>
            </div>
            <div>
              <dt class="text-muted-foreground">log-loss</dt>
              <dd class="font-mono tabular-nums">{{ stats().loss | number: '1.3-3' }}</dd>
            </div>
            <div>
              <dt class="text-muted-foreground">accuracy</dt>
              <dd class="font-mono tabular-nums">{{ stats().accuracy | number: '1.0-0' }}%</dd>
            </div>
          </dl>
        </div>
      </div>
      <p caption>
        The shading is the network's probability of blue: the boundary is where it turns orange.
        With the <strong>identity</strong> activation, or a single neuron, the boundary can only be
        a straight line, whatever you do: that is Logistic Regression. Add neurons and a bend and it
        curves around the circle or the moons; the spiral needs more neurons and patience. Every
        hidden neuron contributes one soft "which side of my line?" signal, and the output layer
        combines them.
      </p>
    </app-figure-frame>
  `,
})
export class BoundaryPlaygroundFigure {
  protected readonly size = SIZE;
  protected readonly cells = CELLS;
  protected readonly cell = CELL;
  protected readonly sx = s;
  protected readonly datasets = DATASETS;
  protected readonly activations = ACTIVATIONS;
  protected readonly floor = Math.floor;
  protected readonly abs = Math.abs;
  protected readonly chip =
    'rounded-md border px-2.5 py-1 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring';

  protected readonly dataset = signal<ToyDataset>('circle');
  protected readonly activation = signal<Activation>('tanh');
  protected readonly layers = signal(1);
  protected readonly units = signal(4);
  protected readonly running = signal(false);
  protected readonly epoch = signal(0);
  protected readonly data = computed(() => toyData(this.dataset(), seeded(7)));

  private net: Network = this.freshNetwork();
  private adam = new Adam(this.net, 0.03);
  /** Bumped whenever the weights change, so the views below recompute. */
  private readonly version = signal(0);
  private frame = 0;

  protected readonly shading = computed(() => {
    this.version();
    return predict(this.net, CENTRES);
  });

  protected readonly stats = computed(() => {
    this.version();
    const { X, y } = this.data();
    const p = predict(this.net, X);
    const correct = p.filter((v, i) => (v > 0.5 ? 1 : 0) === y[i]).length;
    return { loss: loss(this.net, X, y), accuracy: (100 * correct) / y.length };
  });

  protected readonly description = computed(() => {
    const { loss: l, accuracy } = this.stats();
    return (
      `Decision boundary of a ${this.layers()}×${this.units()} ${this.activation()} network on ` +
      `the ${this.dataset()} dataset after ${this.epoch()} epochs: accuracy ${accuracy.toFixed(0)}%, ` +
      `log-loss ${l.toFixed(3)}.`
    );
  });

  constructor() {
    inject(DestroyRef).onDestroy(() => cancelAnimationFrame(this.frame));
  }

  protected choose(dataset: ToyDataset): void {
    this.dataset.set(dataset);
    this.reset();
  }

  protected toggle(): void {
    if (this.running()) {
      this.running.set(false);
      cancelAnimationFrame(this.frame);
      return;
    }
    this.running.set(true);
    const run = () => {
      const { X, y } = this.data();
      for (let i = 0; i < PER_FRAME; i++) this.adam.step(this.net, gradients(this.net, X, y));
      this.epoch.update((e) => e + PER_FRAME);
      this.version.update((v) => v + 1);
      if (this.epoch() >= MAX_EPOCHS) this.running.set(false);
      if (this.running()) this.frame = requestAnimationFrame(run);
    };
    this.frame = requestAnimationFrame(run);
  }

  /** New random weights for the current settings; training continues if it was running. */
  protected reset(): void {
    this.net = this.freshNetwork();
    this.adam = new Adam(this.net, 0.03);
    this.epoch.set(0);
    this.version.update((v) => v + 1);
  }

  private freshNetwork(): Network {
    const hidden = Array<number>(this.layers()).fill(this.units());
    return createNetwork([2, ...hidden, 1], this.activation(), 'sigmoid', seeded(3));
  }
}
