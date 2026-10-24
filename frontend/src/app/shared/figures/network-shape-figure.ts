import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';

const WIDTH = 360;
const HEIGHT = 200;
/** Wider layers are drawn with this many neurons and a "…" (the count is still exact). */
const DRAWN = 8;

interface Column {
  label: string;
  size: number;
  x: number;
  ys: number[];
}

/** The shape of an MLP: inputs, hidden layers and outputs, and how many weights that is. */
@Component({
  selector: 'app-network-shape-figure',
  imports: [DecimalPipe, FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="The shape of a network, and what it costs"
      hint="Change the layers and watch the number of weights to learn."
    >
      <svg
        [attr.viewBox]="'0 0 ' + width + ' ' + height"
        class="w-full"
        role="img"
        [attr.aria-label]="description()"
      >
        @for (pair of links(); track $index) {
          @for (y1 of pair[0].ys; track $index) {
            @for (y2 of pair[1].ys; track $index) {
              <line
                [attr.x1]="pair[0].x"
                [attr.y1]="y1"
                [attr.x2]="pair[1].x"
                [attr.y2]="y2"
                class="stroke-muted-foreground"
                stroke-opacity="0.25"
                stroke-width="0.6"
              />
            }
          }
        }
        @for (column of columns(); track $index; let i = $index) {
          @for (y of column.ys; track $index) {
            <circle
              [attr.cx]="column.x"
              [attr.cy]="y"
              r="6"
              [class]="
                i === 0 ? 'fill-fig-3' : i === columns().length - 1 ? 'fill-fig-2' : 'fill-fig-1'
              "
            />
          }
          @if (column.size > drawn) {
            <text
              [attr.x]="column.x"
              [attr.y]="height - 24"
              text-anchor="middle"
              class="fill-muted-foreground text-[10px]"
            >
              ⋮
            </text>
          }
          <text
            [attr.x]="column.x"
            [attr.y]="height - 6"
            text-anchor="middle"
            class="fill-foreground text-[10px]"
          >
            {{ column.label }} ({{ column.size }})
          </text>
        }
      </svg>

      <div class="mt-3 grid gap-3 sm:grid-cols-2">
        <app-figure-slider label="Input features" [(value)]="features" [min]="1" [max]="20" />
        <app-figure-slider label="Hidden layers" [(value)]="layers" [min]="1" [max]="3" />
        <app-figure-slider
          label="Neurons per hidden layer"
          [(value)]="units"
          [min]="1"
          [max]="128"
        />
        <app-figure-slider label="Outputs" [(value)]="outputs" [min]="1" [max]="5" />
      </div>

      <dl class="mt-3 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
        @for (row of counts(); track $index) {
          <dt class="text-muted-foreground">{{ row.label }}</dt>
          <dd class="font-mono tabular-nums">
            ({{ row.inputs }} + 1) × {{ row.outputs }} = {{ row.count | number }}
          </dd>
        }
        <dt class="font-semibold">Total</dt>
        <dd class="font-mono font-semibold tabular-nums">
          {{ total() | number }} weights and biases
        </dd>
      </dl>
      <p caption>
        Every neuron has one weight per neuron in the layer before it, plus a bias: that is the "+
        1". Inputs (green) feed the hidden layers (blue), which feed the outputs (orange: one for
        regression or two classes, one per class otherwise). The Lab's default, one layer of 100
        neurons on 10 features, already has 1,201 numbers to learn, which is why networks need
        plenty of rows.
      </p>
    </app-figure-frame>
  `,
})
export class NetworkShapeFigure {
  protected readonly width = WIDTH;
  protected readonly height = HEIGHT;
  protected readonly drawn = DRAWN;
  protected readonly features = signal(4);
  protected readonly layers = signal(1);
  protected readonly units = signal(6);
  protected readonly outputs = signal(1);

  private readonly sizes = computed(() => [
    this.features(),
    ...Array<number>(this.layers()).fill(this.units()),
    this.outputs(),
  ]);

  protected readonly columns = computed<Column[]>(() => {
    const sizes = this.sizes();
    return sizes.map((size, i) => {
      const shown = Math.min(size, DRAWN);
      const top = 16;
      const bottom = HEIGHT - 40;
      const gap = (bottom - top) / DRAWN;
      const start = (top + bottom) / 2 - ((shown - 1) * gap) / 2;
      return {
        label: i === 0 ? 'inputs' : i === sizes.length - 1 ? 'outputs' : `hidden ${i}`,
        size,
        x: 30 + (i * (WIDTH - 60)) / (sizes.length - 1),
        ys: Array.from({ length: shown }, (_, k) => start + k * gap),
      };
    });
  });

  protected readonly links = computed(() => {
    const columns = this.columns();
    return columns.slice(1).map((column, i) => [columns[i], column] as const);
  });

  protected readonly counts = computed(() => {
    const sizes = this.sizes();
    return sizes.slice(1).map((outputs, i) => ({
      label: i === sizes.length - 2 ? 'Output layer' : `Hidden layer ${i + 1}`,
      inputs: sizes[i],
      outputs,
      count: (sizes[i] + 1) * outputs,
    }));
  });

  protected readonly total = computed(() => this.counts().reduce((s, row) => s + row.count, 0));

  protected readonly description = computed(
    () =>
      `Network diagram with ${this.sizes().join(', ')} neurons per layer, ` +
      `${this.total()} weights and biases in total.`,
  );
}
