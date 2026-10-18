import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';

import { SchemaField } from '../json-schema';
import type { SchemaWidget } from '../widget-registry';
import { CONTROL_CLASS } from './control-class';

@Component({
  selector: 'app-number-widget',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <input
      type="number"
      inputmode="decimal"
      class="max-w-48"
      [class]="controlClass"
      [id]="inputId()"
      [value]="display()"
      [attr.min]="field().minimum"
      [attr.max]="field().maximum"
      [attr.step]="field().type === 'integer' ? 1 : 'any'"
      [attr.placeholder]="field().nullable ? 'None' : null"
      [attr.aria-describedby]="describedBy()"
      [attr.aria-invalid]="invalid()"
      [attr.aria-required]="field().required && !field().nullable"
      (input)="onInput($event)"
    />
  `,
})
export class NumberWidget implements SchemaWidget {
  readonly field = input.required<SchemaField>();
  readonly value = input<unknown>();
  readonly inputId = input.required<string>();
  readonly describedBy = input('');
  readonly invalid = input(false);
  readonly valueChange = output<unknown>();

  protected readonly controlClass = CONTROL_CLASS;

  protected readonly display = computed(() => {
    const value = this.value();
    return typeof value === 'number' && !Number.isNaN(value) ? String(value) : '';
  });

  protected onInput(event: Event): void {
    const el = event.target as HTMLInputElement;
    // `badInput` is text the browser can't parse ("1e"); keep it invalid rather than empty.
    if (el.validity.badInput) this.valueChange.emit(Number.NaN);
    else this.valueChange.emit(el.value === '' ? null : Number(el.value));
  }
}
