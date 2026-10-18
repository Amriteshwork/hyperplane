import { ChangeDetectionStrategy, Component, input } from '@angular/core';

/** One headline number. Callers format `value`; `hint` explains it in plain words. */
@Component({
  selector: 'app-stat-card',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block rounded-lg border bg-card p-4' },
  template: `
    <p class="text-sm text-muted-foreground">{{ label() }}</p>
    <p class="mt-1 text-2xl font-semibold tracking-tight">{{ value() }}</p>
    @if (hint()) {
      <p class="mt-1 text-xs text-muted-foreground">{{ hint() }}</p>
    }
  `,
})
export class StatCard {
  readonly label = input.required<string>();
  readonly value = input.required<string>();
  readonly hint = input<string>();
}
