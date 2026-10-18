import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';

import { SchemaField } from '../json-schema';
import type { SchemaWidget } from '../widget-registry';
import { CONTROL_CLASS } from './control-class';

/** Native <select> keyed by option index, so non-string enum values (numbers, null) survive. */
@Component({
  selector: 'app-enum-widget',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <select
      class="max-w-xs"
      [class]="controlClass"
      [id]="inputId()"
      [attr.aria-describedby]="describedBy()"
      [attr.aria-invalid]="invalid()"
      (change)="onChange($event)"
    >
      @if (field().nullable) {
        <option value="" [selected]="selectedIndex() === -1">None</option>
      }
      @for (option of options(); track $index) {
        <option [value]="$index" [selected]="$index === selectedIndex()">{{ option }}</option>
      }
    </select>
  `,
})
export class EnumWidget implements SchemaWidget {
  readonly field = input.required<SchemaField>();
  readonly value = input<unknown>();
  readonly inputId = input.required<string>();
  readonly describedBy = input('');
  readonly invalid = input(false);
  readonly valueChange = output<unknown>();

  protected readonly controlClass = CONTROL_CLASS;

  protected readonly options = computed(() => this.field().enum ?? []);
  protected readonly selectedIndex = computed(() =>
    this.options().findIndex((option) => option === this.value()),
  );

  protected onChange(event: Event): void {
    const raw = (event.target as HTMLSelectElement).value;
    this.valueChange.emit(raw === '' ? null : this.options()[Number(raw)]);
  }
}
