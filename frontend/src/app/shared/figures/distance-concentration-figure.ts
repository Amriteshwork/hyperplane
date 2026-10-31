import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { cubeDistances, edgeForFraction, histogram } from './lib/knn';
import { linear } from './lib/plot';
import { seeded } from './lib/random';

const WIDTH = 460;
const HEIGHT = 170;
const BINS = 40;
const ROWS = 1000;
/** Bars run over [0, 2] × the mean distance. */
const MAX_RELATIVE = 2;
const DIMENSIONS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000];
const sx = linear(0, BINS, 0, WIDTH);

/** Drag the number of features up and watch every distance pile onto the same value. */
@Component({
  selector: 'app-distance-concentration-figure',
  imports: [DecimalPipe, FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Every row ends up equally far away"
      hint="Add features and watch the distances from one query to 1,000 random rows."
    >
      <svg
        [attr.viewBox]="'0 0 ' + width + ' ' + (height + 18)"
        class="w-full rounded-md border"
        role="img"
        [attr.aria-label]="description()"
      >
        @for (count of view().counts; track $index) {
          <rect
            [attr.x]="sx($index) + 0.5"
            [attr.y]="height - (count / view().tallest) * (height - 6)"
            [attr.width]="sx(1) - 1"
            [attr.height]="(count / view().tallest) * (height - 6)"
            class="fill-fig-1"
          />
        }
        <line
          [attr.x1]="width / 2"
          [attr.x2]="width / 2"
          y1="0"
          [attr.y2]="height"
          class="stroke-foreground"
          stroke-dasharray="3 3"
        />
        <text x="4" [attr.y]="height + 13" class="fill-muted-foreground text-[11px]">0</text>
        <text
          [attr.x]="width / 2"
          [attr.y]="height + 13"
          text-anchor="middle"
          class="fill-muted-foreground text-[11px]"
        >
          mean distance
        </text>
        <text
          [attr.x]="width - 4"
          [attr.y]="height + 13"
          text-anchor="end"
          class="fill-muted-foreground text-[11px]"
        >
          2 × mean
        </text>
      </svg>
      <div class="mt-3">
        <app-figure-slider
          label="Features (dimensions)"
          [(value)]="step"
          [min]="0"
          [max]="steps"
          [readout]="dimensions() + ''"
        />
      </div>
      <dl class="mt-3 grid grid-cols-2 gap-2 text-xs" aria-live="polite">
        <div>
          <dt class="text-muted-foreground">farthest ÷ nearest</dt>
          <dd class="font-mono tabular-nums" data-testid="contrast">
            {{ view().contrast | number: '1.2-2' }}
          </dd>
        </div>
        <div>
          <dt class="text-muted-foreground">side of a cube holding 1% of the rows</dt>
          <dd class="font-mono tabular-nums" data-testid="edge">
            {{ view().edge | number: '1.2-2' }}
          </dd>
        </div>
      </dl>
      <p caption>
        Distances are shown relative to their mean, so what changes is their spread. With two
        features they range widely, and the nearest row really is near. Each extra feature adds
        another independent term to the sum, the terms average out, and the distances pile up around
        the mean. By 1,000 features the farthest row is barely farther than the nearest, and "the
        most similar rows" means very little.
      </p>
    </app-figure-frame>
  `,
})
export class DistanceConcentrationFigure {
  protected readonly width = WIDTH;
  protected readonly height = HEIGHT;
  protected readonly steps = DIMENSIONS.length - 1;
  protected readonly sx = sx;

  protected readonly step = signal(1);
  protected readonly dimensions = computed(() => DIMENSIONS[this.step()]);

  protected readonly view = computed(() => {
    const d = this.dimensions();
    const distances = cubeDistances(d, ROWS, seeded(d));
    const mean = distances.reduce((sum, v) => sum + v, 0) / ROWS;
    const counts = histogram(
      distances.map((v) => v / mean),
      0,
      MAX_RELATIVE,
      BINS,
    );
    return {
      counts,
      tallest: Math.max(...counts),
      contrast: Math.max(...distances) / Math.min(...distances),
      edge: edgeForFraction(0.01, d),
    };
  });

  protected readonly description = computed(
    () =>
      `With ${this.dimensions()} features, the farthest of ${ROWS} random rows is ` +
      `${this.view().contrast.toFixed(2)} times as far as the nearest.`,
  );
}
