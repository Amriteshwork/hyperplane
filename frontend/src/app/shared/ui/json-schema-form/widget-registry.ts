import { InjectionToken, InputSignal, OutputEmitterRef, Type } from '@angular/core';

import { SchemaField } from './json-schema';
import { BooleanWidget } from './widgets/boolean-widget';
import { EnumWidget } from './widgets/enum-widget';
import { NumberWidget } from './widgets/number-widget';
import { TextWidget } from './widgets/text-widget';

/** What every widget accepts; the form binds these by name. */
export interface SchemaWidget {
  readonly field: InputSignal<SchemaField>;
  readonly value: InputSignal<unknown>;
  readonly inputId: InputSignal<string>;
  readonly describedBy: InputSignal<string>;
  readonly invalid: InputSignal<boolean>;
  readonly valueChange: OutputEmitterRef<unknown>;
}

/** Widgets keyed by schema `format`, `enum`, or `type` (ARCHITECTURE §12 O). */
export type WidgetRegistry = Readonly<Record<string, Type<SchemaWidget>>>;

export const DEFAULT_WIDGETS: WidgetRegistry = {
  enum: EnumWidget,
  boolean: BooleanWidget,
  integer: NumberWidget,
  number: NumberWidget,
  string: TextWidget,
};

/** Override (or extend with a new `format`) by providing this token. */
export const SCHEMA_WIDGETS = new InjectionToken<WidgetRegistry>('SCHEMA_WIDGETS', {
  factory: () => DEFAULT_WIDGETS,
});

/** Most specific key wins: a custom `format`, then `enum`, then the JSON `type`. */
export function resolveWidget(
  registry: WidgetRegistry,
  field: SchemaField,
): Type<SchemaWidget> | null {
  const keys = [field.format, field.enum ? 'enum' : null, field.type];
  for (const key of keys) {
    if (key && registry[key]) return registry[key];
  }
  return null;
}
