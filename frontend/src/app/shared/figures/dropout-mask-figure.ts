import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { seeded } from './lib/random';
import { dropoutMask } from './lib/training';

const LAYERS = [6, 8, 8, 4];
const W = 300;
const H = 170;

/** A small network with a fresh random set of units silenced at each training step. */
@Component({
  selector: 'app-dropout-mask-figure',
  imports: [FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame heading="Dropout: a different network each step" hint="Press Next batch.">
      <svg
        [attr.viewBox]="'0 0 ' + W + ' ' + H"
        class="w-full"
        role="img"
        [attr.aria-label]="
          'Step ' + step() + ': ' + silent() + ' of ' + hidden + ' hidden units silenced.'
        "
      >
        @for (edge of edges(); track edge.id) {
          <line
            [attr.x1]="edge.x1"
            [attr.y1]="edge.y1"
            [attr.x2]="edge.x2"
            [attr.y2]="edge.y2"
            [class]="edge.on ? 'stroke-muted-foreground/40' : 'stroke-transparent'"
          />
        }
        @for (node of nodes(); track node.id) {
          <circle
            [attr.cx]="node.x"
            [attr.cy]="node.y"
            r="6"
            [class]="node.on ? 'fill-fig-1' : 'fill-muted stroke-border'"
          />
        }
      </svg>
      <div class="mt-2 grid items-end gap-3 sm:grid-cols-2">
        <app-figure-slider
          label="Dropout rate"
          [(value)]="rate"
          [min]="0"
          [max]="0.8"
          [step]="0.05"
          [readout]="rate().toFixed(2)"
        />
        <button
          type="button"
          class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
          (click)="step.set(step() + 1)"
        >
          Next batch
        </button>
      </div>
      <p class="mt-2 text-sm" aria-live="polite">
        {{ silent() }} of {{ hidden }} hidden units are silent, so the others are multiplied by
        {{ (1 / (1 - rate())).toFixed(2) }} to keep the average signal the same.
      </p>
      <p caption>
        A new random set goes quiet at every step, so no unit can count on any other being there. At
        test time everything's on. It often helps larger models on limited data. In our tiny GPT
        test it made validation loss worse, because that small model needs all the capacity it has.
      </p>
    </app-figure-frame>
  `,
})
export class DropoutMaskFigure {
  protected readonly W = W;
  protected readonly H = H;
  protected readonly hidden = LAYERS[1] + LAYERS[2];
  protected readonly rate = signal(0.3);
  protected readonly step = signal(1);

  private readonly masks = computed(() => {
    const rng = seeded(this.step() * 97 + 1);
    return LAYERS.map((size, layer) =>
      layer === 0 || layer === LAYERS.length - 1
        ? new Array<boolean>(size).fill(true)
        : dropoutMask(size, this.rate(), rng),
    );
  });
  protected readonly silent = computed(() =>
    this.masks().reduce((sum, m) => sum + m.filter((on) => !on).length, 0),
  );
  protected readonly nodes = computed(() =>
    LAYERS.flatMap((size, layer) =>
      Array.from({ length: size }, (_, i) => ({
        id: `${layer}-${i}`,
        x: 24 + (layer * (W - 48)) / (LAYERS.length - 1),
        y: H / 2 + (i - (size - 1) / 2) * 19,
        on: this.masks()[layer][i],
      })),
    ),
  );
  protected readonly edges = computed(() => {
    const nodes = this.nodes();
    const layerOf = (layer: number) => nodes.filter((n) => n.id.startsWith(`${layer}-`));
    return LAYERS.slice(1).flatMap((_, layer) =>
      layerOf(layer).flatMap((a) =>
        layerOf(layer + 1).map((b) => ({
          id: `${a.id}:${b.id}`,
          x1: a.x,
          y1: a.y,
          x2: b.x,
          y2: b.y,
          on: a.on && b.on,
        })),
      ),
    );
  });
}
