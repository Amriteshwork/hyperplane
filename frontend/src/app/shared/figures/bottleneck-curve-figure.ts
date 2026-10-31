import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { BOTTLENECK } from './lib/measured';
import { linear, path } from './lib/plot';

const W = 320;
const H = 180;
const x = linear(1, 12, 34, W - 12);
const y = linear(0, 1, H - 26, 10);
const curve = (values: readonly number[]) =>
  path(values.map((v, i): [number, number] => [x(BOTTLENECK.lengths[i]), y(v)]));
const shown = BOTTLENECK.lengths.map((length, i) => ({
  length,
  plain: BOTTLENECK.plain[i],
  attention: BOTTLENECK.attention[i],
}));

/** The measured wall: exact translations as lists get longer, with and without attention. */
@Component({
  selector: 'app-bottleneck-curve-figure',
  imports: [FigureFrame, FigureSlider],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="The wall, measured"
      hint="Drag the list length. Same LSTM size, data and steps for both models."
    >
      <svg
        [attr.viewBox]="'0 0 ' + W + ' ' + H"
        class="w-full"
        role="img"
        [attr.aria-label]="summary()"
      >
        @for (tick of [0, 0.5, 1]; track tick) {
          <line
            [attr.x1]="34"
            [attr.x2]="W - 12"
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
            {{ tick * 100 }}%
          </text>
        }
        <path [attr.d]="plainLine" class="fill-none stroke-fig-2 stroke-2" />
        <path [attr.d]="attentionLine" class="fill-none stroke-fig-1 stroke-2" />
        <line
          [attr.x1]="x(row().length)"
          [attr.x2]="x(row().length)"
          y1="10"
          [attr.y2]="H - 26"
          class="stroke-muted-foreground"
          stroke-dasharray="2 2"
        />
        <circle [attr.cx]="x(row().length)" [attr.cy]="y(row().plain)" r="3.5" class="fill-fig-2" />
        <circle
          [attr.cx]="x(row().length)"
          [attr.cy]="y(row().attention)"
          r="3.5"
          class="fill-fig-1"
        />
        <text
          [attr.x]="(34 + W - 12) / 2"
          [attr.y]="H - 6"
          text-anchor="middle"
          class="fill-muted-foreground text-[8px]"
        >
          words in the list
        </text>
      </svg>
      <p class="mt-1 flex flex-wrap gap-3 text-xs">
        <span class="text-fig-1">● with attention</span>
        <span class="text-fig-2">● one vector only</span>
      </p>
      <app-figure-slider
        class="mt-2"
        label="List length"
        [(value)]="index"
        [min]="0"
        [max]="rows.length - 1"
        [readout]="row().length + ' words'"
      />

      <div class="mt-3 grid gap-2 sm:grid-cols-2" aria-live="polite">
        <div class="rounded-md border-l-4 border-fig-2 bg-muted/50 p-2">
          <p class="text-xs text-muted-foreground">One vector only</p>
          <p class="text-lg font-semibold tabular-nums">{{ percent(row().plain) }}</p>
        </div>
        <div class="rounded-md border-l-4 border-fig-1 bg-muted/50 p-2">
          <p class="text-xs text-muted-foreground">With attention</p>
          <p class="text-lg font-semibold tabular-nums">{{ percent(row().attention) }}</p>
        </div>
      </div>
      <p caption>
        The share of unseen lists translated with every word right. Both are perfect up to 3 words,
        so both models can translate. Then the plain model's vector runs out of room and it falls to
        0% by 12 words, while attention drops only to 70%.
      </p>
    </app-figure-frame>
  `,
})
export class BottleneckCurveFigure {
  protected readonly W = W;
  protected readonly H = H;
  protected readonly x = x;
  protected readonly y = y;
  protected readonly rows = shown;
  protected readonly plainLine = curve(BOTTLENECK.plain);
  protected readonly attentionLine = curve(BOTTLENECK.attention);
  protected readonly index = signal(4);
  protected readonly row = computed(() => this.rows[this.index()]);
  protected readonly summary = computed(
    () =>
      `At ${this.row().length} words: one vector ${this.percent(this.row().plain)}, with attention ${this.percent(this.row().attention)}.`,
  );

  protected percent(value: number): string {
    return `${Math.round(value * 100)}%`;
  }
}
