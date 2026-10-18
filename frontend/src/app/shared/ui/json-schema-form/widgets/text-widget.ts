import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';

import { SchemaField } from '../json-schema';
import type { SchemaWidget } from '../widget-registry';
import { CONTROL_CLASS } from './control-class';

@Component({
  selector: 'app-text-widget',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <input
      type="text"
      [class]="controlClass"
      [id]="inputId()"
      [value]="display()"
      [attr.minlength]="field().minLength"
      [attr.maxlength]="field().maxLength"
      [attr.placeholder]="field().nullable ? 'None' : null"
      [attr.aria-describedby]="describedBy()"
      [attr.aria-invalid]="invalid()"
      [attr.aria-required]="field().required && !field().nullable"
      (input)="onInput($event)"
    />
  `,
})
export class TextWidget implements SchemaWidget {
  readonly field = input.required<SchemaField>();
  readonly value = input<unknown>();
  readonly inputId = input.required<string>();
  readonly describedBy = input('');
  readonly invalid = input(false);
  readonly valueChange = output<unknown>();

  protected readonly controlClass = CONTROL_CLASS;

  protected readonly display = computed(() => {
    const value = this.value();
    return value === null || value === undefined ? '' : String(value);
  });

  protected onInput(event: Event): void {
    const value = (event.target as HTMLInputElement).value;
    this.valueChange.emit(value === '' && this.field().nullable ? null : value);
  }
}
