import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { WARMUP_RUNS } from './lib/measured';
import { linear, path, range } from './lib/plot';
import { learningRate } from './lib/training';

const W = 320;
const H = 150;
const TOTAL = 200;
const x = linear(0, TOTAL, 34, W - 10);
const y = linear(0, 0.11, H - 22, 8);
const MAX_LOSS = 1.8;

/** Shape a learning-rate schedule, beside what each schedule really did to a training run. */
@Component({
  selector: 'app-lr-warmup-figure',
  imports: [FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Warmup and cosine decay"
      hint="Drag warmup to 0 to get a constant rate. Below, what four real runs did."
    >
      <svg
        [attr.viewBox]="'0 0 ' + W + ' ' + H"
        class="w-full"
        role="img"
        [attr.aria-label]="
          'Learning rate climbs to ' + peak() + ' over ' + warmup() + ' steps, then falls to 0.'
        "
      >
        @for (tick of [0, 0.05, 0.1]; track tick) {
          <line
            [attr.x1]="34"
            [attr.x2]="W - 10"
            [attr.y1]="y(tick)"
            [attr.y2]="y(tick)"
            class="stroke-border"
          />
          <text
            [attr.x]="30"
            [attr.y]="y(tick) + 3"
            text-anchor="end"
            class="fill-muted-foreground text-[8px]"
          >
            {{ tick }}
          </text>
        }
        <path [attr.d]="curve()" class="fill-none stroke-fig-1 stroke-2" />
        <text
          [attr.x]="(34 + W - 10) / 2"
          [attr.y]="H - 6"
          text-anchor="middle"
          class="fill-muted-foreground text-[8px]"
        >
          training step
        </text>
      </svg>
      <div class="mt-2 grid gap-3 sm:grid-cols-2">
        <app-figure-slider
          label="Peak learning rate"
          [(value)]="peak"
          [min]="0.01"
          [max]="0.1"
          [step]="0.01"
          [readout]="peak().toFixed(2)"
        />
        <app-figure-slider
          label="Warmup steps"
          [(value)]="warmup"
          [min]="0"
          [max]="100"
          [step]="5"
          [readout]="warmup() ? warmup() + ' steps' : 'off (constant)'"
        />
      </div>

      <p class="mt-4 text-xs font-semibold">Validation loss after 200 steps (lower is better)</p>
      <ul class="list-none pl-0 mt-1 space-y-1.5 text-xs">
        @for (run of runs; track run.label) {
          <li>
            <div class="flex justify-between gap-2">
              <span>{{ run.label }}</span>
              <span class="font-semibold tabular-nums">{{ run.loss.toFixed(2) }}</span>
            </div>
            <div class="mt-0.5 h-2 rounded bg-muted">
              <div
                class="h-2 rounded"
                [class]="run.loss > 1 ? 'bg-fig-negative' : 'bg-fig-3'"
                [style.width.%]="(run.loss / maxLoss) * 100"
              ></div>
            </div>
            <p class="text-[11px] text-muted-foreground">{{ run.note }}</p>
          </li>
        }
      </ul>
      <p caption>
        These four runs share a model, a seed and data. The second and fourth reach the same peak of
        0.1. The only difference is the road there: one starts at full speed on random weights and
        never recovers, and the other starts gently and does as well as a safe rate. Clipping
        shortens gradients but can't fix a rate that's wrong on every step, because Adam already
        sets the size of its own steps.
      </p>
    </app-figure-frame>
  `,
})
export class LrWarmupFigure {
  protected readonly W = W;
  protected readonly H = H;
  protected readonly y = y;
  protected readonly runs = WARMUP_RUNS;
  protected readonly maxLoss = MAX_LOSS;
  protected readonly peak = signal(0.1);
  protected readonly warmup = signal(40);
  protected readonly curve = computed(() =>
    path(
      range(0, TOTAL - 1, 120).map((s): [number, number] => [
        x(s),
        y(learningRate(s, this.peak(), this.warmup(), TOTAL)),
      ]),
    ),
  );
}
