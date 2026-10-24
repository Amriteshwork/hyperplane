import { ChangeDetectionStrategy, Component, input, model } from '@angular/core';

import { sliderValue } from './lib/events';

/** A labelled range input for figures; `readout` is shown next to the label. */
@Component({
  selector: 'app-figure-slider',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block' },
  template: `
    <label class="block text-xs">
      <span class="flex justify-between gap-2">
        <span>{{ label() }}</span>
        <span class="tabular-nums text-muted-foreground">{{ readout() ?? value() }}</span>
      </span>
      <input
        type="range"
        class="mt-1 block w-full accent-fig-1"
        [min]="min()"
        [max]="max()"
        [step]="step()"
        [value]="value()"
        (input)="value.set(read($event))"
      />
    </label>
  `,
})
export class FigureSlider {
  readonly label = input.required<string>();
  readonly value = model.required<number>();
  readonly min = input(0);
  readonly max = input(1);
  readonly step = input(1);
  readonly readout = input<string>();

  protected readonly read = sliderValue;
}
