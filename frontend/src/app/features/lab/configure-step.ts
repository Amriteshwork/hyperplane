import { ChangeDetectionStrategy, Component, inject, output } from '@angular/core';
import { RouterLink } from '@angular/router';
import { HlmButton } from '@spartan-ng/helm/button';

import { JsonSchemaForm } from '../../shared/ui/json-schema-form/json-schema-form';
import { CONTROL_CLASS } from '../../shared/ui/json-schema-form/widgets/control-class';
import { LabStore } from './lab.store';

/** Algorithm, hyperparameters and the train/test split, then train. */
@Component({
  selector: 'app-configure-step',
  imports: [RouterLink, HlmButton, JsonSchemaForm],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <div class="grid gap-2">
      <label for="lab-algorithm" class="text-sm font-medium">Algorithm</label>
      <select
        id="lab-algorithm"
        class="max-w-sm"
        [class]="controlClass"
        (change)="store.algorithmId.set($any($event.target).value)"
      >
        @for (algorithm of store.targetOption()?.algorithms ?? []; track algorithm.id) {
          <option [value]="algorithm.id" [selected]="algorithm.id === store.algorithmId()">
            {{ algorithm.name }}
          </option>
        }
      </select>
      @if (store.algorithmId(); as id) {
        <a class="text-xs text-muted-foreground underline" [routerLink]="['/learn/algorithms', id]">
          How does it work?
        </a>
      }
    </div>

    <section class="mt-8" aria-labelledby="params-heading">
      <h2 id="params-heading" class="text-sm font-medium">Hyperparameters</h2>
      @if (store.paramsSchema(); as schema) {
        <app-json-schema-form
          class="mt-3 block"
          [schema]="schema"
          [(value)]="store.params"
          (validChange)="store.paramsValid.set($event)"
        />
      }
    </section>

    <section class="mt-8 grid gap-4 sm:grid-cols-2" aria-label="Train/test split">
      <div class="grid gap-1">
        <label for="lab-test-size" class="text-sm font-medium">Test set size (%)</label>
        <input
          id="lab-test-size"
          type="number"
          min="5"
          max="50"
          step="5"
          class="max-w-32"
          [class]="controlClass"
          aria-describedby="lab-test-size-help"
          [value]="store.testSize() * 100"
          (change)="setTestSize($any($event.target).valueAsNumber)"
        />
        <p id="lab-test-size-help" class="text-xs text-muted-foreground">
          Rows held out to measure the model on data it never saw (5–50%).
        </p>
      </div>
      <div class="grid gap-1">
        <label for="lab-seed" class="text-sm font-medium">Random seed</label>
        <input
          id="lab-seed"
          type="number"
          min="0"
          step="1"
          class="max-w-32"
          [class]="controlClass"
          aria-describedby="lab-seed-help"
          [value]="store.seed()"
          (change)="setSeed($any($event.target).valueAsNumber)"
        />
        <p id="lab-seed-help" class="text-xs text-muted-foreground">
          Same seed, same split: rerunning gives the same numbers.
        </p>
      </div>
    </section>

    @if (store.createRunError(); as error) {
      <p class="mt-6 text-sm text-destructive" role="alert">{{ error }}</p>
    }
    <div class="mt-8 flex gap-2">
      <button hlmBtn variant="outline" type="button" (click)="back.emit()">Back</button>
      <button
        hlmBtn
        type="button"
        [disabled]="!store.paramsValid() || store.creatingRun()"
        (click)="train.emit()"
      >
        {{ store.creatingRun() ? 'Starting…' : 'Train model' }}
      </button>
    </div>
  `,
})
export class ConfigureStep {
  protected readonly store = inject(LabStore);
  protected readonly controlClass = CONTROL_CLASS;

  readonly back = output();
  readonly train = output();

  protected setTestSize(percent: number): void {
    if (Number.isFinite(percent)) this.store.testSize.set(Math.min(50, Math.max(5, percent)) / 100);
  }

  protected setSeed(seed: number): void {
    if (Number.isInteger(seed) && seed >= 0) this.store.seed.set(seed);
  }
}
