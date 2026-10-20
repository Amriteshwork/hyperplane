import { ChangeDetectionStrategy, Component, inject, output } from '@angular/core';
import { HlmBadge } from '@spartan-ng/helm/badge';
import { HlmButton } from '@spartan-ng/helm/button';

import { CONTROL_CLASS } from '../../shared/ui/json-schema-form/widgets/control-class';
import { LabStore } from './lab.store';

/** Choose the column to predict and the columns to learn from. */
@Component({
  selector: 'app-setup-step',
  imports: [HlmBadge, HlmButton],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    @if (store.optionsError(); as error) {
      <p class="text-sm text-destructive" role="alert">{{ error }}</p>
    } @else if (store.options(); as options) {
      <div class="grid gap-2">
        <label for="lab-target" class="text-sm font-medium">What do you want to predict?</label>
        <select
          id="lab-target"
          class="max-w-sm"
          [class]="controlClass"
          aria-describedby="lab-target-help"
          (change)="store.target.set($any($event.target).value)"
        >
          @for (target of options.targets; track target.column) {
            <option [value]="target.column" [selected]="target.column === store.target()">
              {{ target.column }} ({{ target.task_type }})
            </option>
          }
        </select>
        @if (store.targetOption(); as target) {
          <p id="lab-target-help" class="text-xs text-muted-foreground">
            {{ target.reason }}
            @if (!target.algorithms.length) {
              <span class="text-destructive">
                No algorithm can train {{ target.task_type }} targets yet; pick another column.
              </span>
            }
          </p>
        }
      </div>

      <fieldset class="mt-8">
        <legend class="text-sm font-medium">Features to learn from</legend>
        <p class="mt-1 text-xs text-muted-foreground">
          {{ store.features().length }} selected. Columns of a type we can’t use yet are disabled.
        </p>
        <ul class="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          @for (feature of options.features; track feature.column) {
            @if (feature.column !== store.target()) {
              <li>
                <label
                  class="flex items-center gap-2 rounded-md border px-3 py-2 text-sm has-[:disabled]:opacity-50"
                >
                  <input
                    type="checkbox"
                    class="size-4 accent-primary"
                    [checked]="store.features().includes(feature.column)"
                    [disabled]="feature.role === null"
                    (change)="store.toggleFeature(feature.column, $any($event.target).checked)"
                  />
                  <span class="truncate">{{ feature.column }}</span>
                  <span hlmBadge variant="outline" class="ms-auto">{{
                    feature.semantic_type
                  }}</span>
                </label>
              </li>
            }
          }
        </ul>
      </fieldset>

      <div class="mt-8 flex gap-2">
        <button hlmBtn variant="outline" type="button" (click)="back.emit()">Change dataset</button>
        <button hlmBtn type="button" [disabled]="!store.setupValid()" (click)="next.emit()">
          Next: settings
        </button>
      </div>
    } @else {
      <div class="h-40 animate-pulse rounded bg-muted motion-reduce:animate-none" aria-busy="true">
        <span class="sr-only">Loading…</span>
      </div>
    }
  `,
})
export class SetupStep {
  protected readonly store = inject(LabStore);
  protected readonly controlClass = CONTROL_CLASS;

  readonly back = output();
  readonly next = output();
}
