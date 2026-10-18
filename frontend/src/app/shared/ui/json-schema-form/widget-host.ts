import {
  Directive,
  Type,
  ViewContainerRef,
  effect,
  inject,
  input,
  inputBinding,
  output,
  outputBinding,
  untracked,
} from '@angular/core';

import { SchemaField } from './json-schema';
import { SchemaWidget } from './widget-registry';

/** Renders the resolved widget in place and forwards the widget contract both ways. */
@Directive({ selector: '[appWidgetHost]' })
export class WidgetHost {
  private readonly vcr = inject(ViewContainerRef);

  readonly appWidgetHost = input.required<Type<SchemaWidget>>();
  readonly field = input.required<SchemaField>();
  readonly value = input<unknown>();
  readonly inputId = input.required<string>();
  readonly describedBy = input('');
  readonly invalid = input(false);
  readonly valueChange = output<unknown>();

  constructor() {
    effect(() => {
      const widget = this.appWidgetHost();
      untracked(() => {
        this.vcr.clear();
        this.vcr.createComponent(widget, {
          bindings: [
            inputBinding('field', this.field),
            inputBinding('value', this.value),
            inputBinding('inputId', this.inputId),
            inputBinding('describedBy', this.describedBy),
            inputBinding('invalid', this.invalid),
            outputBinding<unknown>('valueChange', (value) => this.valueChange.emit(value)),
          ],
        });
      });
    });
  }
}
