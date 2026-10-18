import { NgTemplateOutlet } from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  computed,
  effect,
  inject,
  input,
  model,
  output,
} from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideChevronDown } from '@ng-icons/lucide';
import { HlmCollapsibleImports } from '@spartan-ng/helm/collapsible';

import { JsonSchema, SchemaField, SchemaValues, schemaDefaults, schemaFields } from './json-schema';
import { validateValues } from './validation';
import { SCHEMA_WIDGETS, resolveWidget } from './widget-registry';
import { WidgetHost } from './widget-host';

let nextFormId = 0;

/**
 * Renders a Pydantic JSON Schema as a form. Widgets come from `SCHEMA_WIDGETS`, so adding a
 * field kind is a registration, not an edit here. Emits the full value object on every change.
 */
@Component({
  selector: 'app-json-schema-form',
  imports: [NgTemplateOutlet, WidgetHost, NgIcon, HlmCollapsibleImports],
  providers: [provideIcons({ lucideChevronDown })],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <ng-template #fieldList let-fields>
      <div class="grid gap-5">
        @for (field of fields; track field.key) {
          <div class="grid gap-1.5" [attr.data-field]="field.key">
            <label class="text-sm font-medium" [for]="inputId(field)">
              {{ field.title }}
              @if (field.required && !field.nullable) {
                <span class="text-destructive" aria-hidden="true">*</span>
              }
            </label>
            @if (widgetFor(field); as widget) {
              <ng-container
                [appWidgetHost]="widget"
                [field]="field"
                [value]="values()[field.key]"
                [inputId]="inputId(field)"
                [describedBy]="describedBy(field)"
                [invalid]="!!errors()[field.key]"
                (valueChange)="setValue(field.key, $event)"
              />
            } @else {
              <p class="text-sm text-muted-foreground">
                Unsupported field type “{{ field.type ?? 'unknown' }}”.
              </p>
            }
            @if (field.description) {
              <p class="text-xs text-muted-foreground" [id]="inputId(field) + '-help'">
                {{ field.description }}
              </p>
            }
            @if (errors()[field.key]; as error) {
              <p class="text-xs text-destructive" role="alert" [id]="inputId(field) + '-error'">
                {{ error }}
              </p>
            }
          </div>
        }
      </div>
    </ng-template>

    <div class="grid gap-6">
      @if (beginnerFields().length) {
        <ng-container *ngTemplateOutlet="fieldList; context: { $implicit: beginnerFields() }" />
      } @else if (!advancedFields().length) {
        <p class="text-sm text-muted-foreground">This algorithm has no settings to tune.</p>
      }

      @if (advancedFields().length) {
        <div hlmCollapsible class="group rounded-md border">
          <button
            hlmCollapsibleTrigger
            type="button"
            class="flex w-full items-center justify-between gap-2 px-4 py-3 text-sm font-medium hover:bg-muted/50 focus-visible:outline-2 focus-visible:outline-ring"
          >
            Advanced ({{ advancedFields().length }})
            <ng-icon
              name="lucideChevronDown"
              aria-hidden="true"
              class="transition-transform group-data-[state=open]:rotate-180 motion-reduce:transition-none"
            />
          </button>
          <div hlmCollapsibleContent class="border-t px-4 py-4">
            <ng-container *ngTemplateOutlet="fieldList; context: { $implicit: advancedFields() }" />
          </div>
        </div>
      }
    </div>
  `,
})
export class JsonSchemaForm {
  private readonly widgets = inject(SCHEMA_WIDGETS);
  private readonly formId = `jsf-${nextFormId++}`;

  readonly schema = input.required<JsonSchema>();
  /** Partial values are fine: missing keys fall back to schema defaults. */
  readonly value = model<SchemaValues>({});
  readonly validChange = output<boolean>();

  protected readonly fields = computed(() => schemaFields(this.schema()));
  protected readonly beginnerFields = computed(() =>
    this.fields().filter((f) => f.level !== 'advanced'),
  );
  protected readonly advancedFields = computed(() =>
    this.fields().filter((f) => f.level === 'advanced'),
  );
  protected readonly values = computed(() => ({
    ...schemaDefaults(this.schema()),
    ...this.value(),
  }));
  protected readonly errors = computed(() => validateValues(this.fields(), this.values()));
  readonly valid = computed(() => Object.keys(this.errors()).length === 0);

  constructor() {
    effect(() => this.validChange.emit(this.valid()));
  }

  protected setValue(key: string, fieldValue: unknown): void {
    this.value.set({ ...this.values(), [key]: fieldValue });
  }

  protected widgetFor(field: SchemaField) {
    return resolveWidget(this.widgets, field);
  }

  protected inputId(field: SchemaField): string {
    return `${this.formId}-${field.key}`;
  }

  protected describedBy(field: SchemaField): string {
    const ids = [];
    if (field.description) ids.push(`${this.inputId(field)}-help`);
    if (this.errors()[field.key]) ids.push(`${this.inputId(field)}-error`);
    return ids.join(' ');
  }
}
