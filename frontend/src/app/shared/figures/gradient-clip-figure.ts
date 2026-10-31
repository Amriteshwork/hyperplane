import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { linear } from './lib/plot';
import { clipByNorm } from './lib/training';

const SIZE = 220;
const scale = linear(0, 10, 0, SIZE / 2 - 6);

/** A gradient as an arrow, and the circle it isn't allowed to leave. */
@Component({
  selector: 'app-gradient-clip-figure',
  imports: [FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame heading="Clipping: same direction, shorter" hint="Make the gradient spike.">
      <svg
        [attr.viewBox]="'0 0 ' + size + ' ' + size"
        class="mx-auto w-full max-w-xs"
        role="img"
        [attr.aria-label]="
          'A gradient of length ' +
          length().toFixed(1) +
          ' becomes length ' +
          result().length.toFixed(1) +
          ' with the limit at ' +
          limit().toFixed(1)
        "
      >
        <circle
          [attr.cx]="c"
          [attr.cy]="c"
          [attr.r]="r(limit())"
          class="fill-fig-3/10 stroke-fig-3"
          stroke-dasharray="3 3"
        />
        <line
          [attr.x1]="c"
          [attr.y1]="c"
          [attr.x2]="tip(length()).x"
          [attr.y2]="tip(length()).y"
          class="stroke-fig-negative stroke-2"
          stroke-dasharray="3 2"
        />
        <line
          [attr.x1]="c"
          [attr.y1]="c"
          [attr.x2]="tip(result().length).x"
          [attr.y2]="tip(result().length).y"
          class="stroke-fig-1"
          stroke-width="3"
        />
        <circle
          [attr.cx]="tip(result().length).x"
          [attr.cy]="tip(result().length).y"
          r="3.5"
          class="fill-fig-1"
        />
      </svg>
      <div class="mt-2 grid gap-3 sm:grid-cols-3">
        <app-figure-slider
          label="Gradient length"
          [(value)]="length"
          [min]="0.5"
          [max]="10"
          [step]="0.5"
          [readout]="length().toFixed(1)"
        />
        <app-figure-slider
          label="Direction"
          [(value)]="angle"
          [min]="0"
          [max]="360"
          [step]="5"
          [readout]="angle() + '°'"
        />
        <app-figure-slider
          label="Limit c"
          [(value)]="limit"
          [min]="0.5"
          [max]="6"
          [step]="0.5"
          [readout]="limit().toFixed(1)"
        />
      </div>
      <p class="mt-2 rounded-md bg-muted/50 p-2 text-sm" aria-live="polite">
        @if (length() > limit()) {
          Too long: multiplied by {{ (limit() / length()).toFixed(2) }}, so it's now length
          {{ result().length.toFixed(1) }}.
        } @else {
          Inside the limit, so it's not touched.
        }
      </p>
      <p caption>
        The dashed line is the raw gradient, and the blue arrow is what the optimizer gets. Turning
        the direction slider moves both together: clipping never changes which way to go, only how
        far. A normal gradient passes through unchanged, so clipping costs nothing until a spike
        hits.
      </p>
    </app-figure-frame>
  `,
})
export class GradientClipFigure {
  protected readonly size = SIZE;
  protected readonly c = SIZE / 2;
  protected readonly length = signal(8);
  protected readonly angle = signal(35);
  protected readonly limit = signal(3);
  protected readonly r = scale;
  private readonly gradient = computed((): [number, number] => {
    const a = (this.angle() * Math.PI) / 180;
    return [this.length() * Math.cos(a), this.length() * Math.sin(a)];
  });
  protected readonly result = computed(() => {
    const { clipped } = clipByNorm(this.gradient(), this.limit());
    return { length: Math.hypot(...clipped) };
  });

  protected tip(length: number): { x: number; y: number } {
    const a = (this.angle() * Math.PI) / 180;
    return { x: this.c + scale(length) * Math.cos(a), y: this.c - scale(length) * Math.sin(a) };
  }
}
