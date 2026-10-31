import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { TEMPERATURE_RUNS } from './lib/measured';
import { linear, path } from './lib/plot';

const W = 320;
const H = 170;
const x = linear(0, 2.5, 34, W - 34);
const y = linear(0, 1, H - 24, 10);
const line = (key: 'valid' | 'distinct') =>
  path(
    TEMPERATURE_RUNS.map((run): [number, number] => [
      x(run.temperature),
      y(key === 'valid' ? run.valid : run.distinct / 200),
    ]),
  );

/** What each temperature really gave: sentences that follow the rule against sentences that differ. */
@Component({
  selector: 'app-temperature-tradeoff-figure',
  imports: [FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Sense against variety, measured"
      hint="Pick a temperature. 200 sentences each, from the tiny GPT on this page."
    >
      <svg
        [attr.viewBox]="'0 0 ' + W + ' ' + H"
        class="w-full"
        role="img"
        [attr.aria-label]="
          'At temperature ' +
          run().temperature +
          ', ' +
          percent(run().valid) +
          ' of sentences are valid and ' +
          run().distinct +
          ' of 200 are distinct.'
        "
      >
        @for (tick of [0, 0.5, 1]; track tick) {
          <line
            [attr.x1]="34"
            [attr.x2]="W - 34"
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
        <path [attr.d]="validLine" class="fill-none stroke-fig-1 stroke-2" />
        <path [attr.d]="distinctLine" class="fill-none stroke-fig-2 stroke-2" />
        <line
          [attr.x1]="x(run().temperature)"
          [attr.x2]="x(run().temperature)"
          y1="10"
          [attr.y2]="H - 24"
          class="stroke-muted-foreground"
          stroke-dasharray="2 2"
        />
        @for (r of runs; track r.temperature) {
          <circle [attr.cx]="x(r.temperature)" [attr.cy]="y(r.valid)" r="3" class="fill-fig-1" />
          <circle
            [attr.cx]="x(r.temperature)"
            [attr.cy]="y(r.distinct / 200)"
            r="3"
            class="fill-fig-2"
          />
        }
        <text
          [attr.x]="W / 2"
          [attr.y]="H - 6"
          text-anchor="middle"
          class="fill-muted-foreground text-[8px]"
        >
          temperature τ
        </text>
      </svg>
      <p class="mt-1 flex flex-wrap gap-3 text-xs">
        <span class="text-fig-1">● follows the rule</span>
        <span class="text-fig-2">● different sentences</span>
      </p>

      <div class="mt-2 flex flex-wrap gap-1.5" role="group" aria-label="Temperature">
        @for (r of runs; track r.temperature; let i = $index) {
          <button
            type="button"
            class="rounded-md border px-2.5 py-1 text-xs tabular-nums hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
            [class.bg-muted]="at() === i"
            [attr.aria-pressed]="at() === i"
            (click)="at.set(i)"
          >
            τ = {{ r.temperature }}
          </button>
        }
      </div>
      <p class="mt-2 rounded-md bg-muted/50 p-2 text-sm" aria-live="polite">
        <span class="font-semibold">{{ percent(run().valid) }}</span> valid,
        <span class="font-semibold">{{ run().distinct }}</span> of 200 different. For example:
        <span class="font-mono text-xs">“{{ run().example }}”</span>
      </p>
      <p caption>
        A chatbot's temperature sits somewhere in the middle, where the blue line hasn't fallen far
        and the orange line has risen. There's no setting that gets you both at the top.
      </p>
    </app-figure-frame>
  `,
})
export class TemperatureTradeoffFigure {
  protected readonly W = W;
  protected readonly H = H;
  protected readonly x = x;
  protected readonly y = y;
  protected readonly runs = TEMPERATURE_RUNS;
  protected readonly validLine = line('valid');
  protected readonly distinctLine = line('distinct');
  protected readonly at = signal(2);
  protected readonly run = computed(() => this.runs[this.at()]);

  protected percent(value: number): string {
    return `${Math.round(value * 100)}%`;
  }
}
