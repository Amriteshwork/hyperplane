import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';
import { HlmSwitch } from '@spartan-ng/helm/switch';

import { SchemaField } from '../json-schema';
import type { SchemaWidget } from '../widget-registry';

@Component({
  selector: 'app-boolean-widget',
  imports: [HlmSwitch],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <hlm-switch
      [inputId]="inputId()"
      [checked]="value() === true"
      [aria-describedby]="describedBy() || null"
      (checkedChange)="valueChange.emit($event)"
    />
  `,
})
export class BooleanWidget implements SchemaWidget {
  readonly field = input.required<SchemaField>();
  readonly value = input<unknown>();
  readonly inputId = input.required<string>();
  readonly describedBy = input('');
  readonly invalid = input(false);
  readonly valueChange = output<unknown>();
}
