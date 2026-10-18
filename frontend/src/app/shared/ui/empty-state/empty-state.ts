import { ChangeDetectionStrategy, Component, input } from '@angular/core';

/** A quiet placeholder for empty lists, missing pages and errors; project actions as content. */
@Component({
  selector: 'app-empty-state',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block rounded-lg border border-dashed px-4 py-10 text-center' },
  template: `
    <p class="font-medium">{{ heading() }}</p>
    @if (description()) {
      <p class="mx-auto mt-1 max-w-prose text-sm text-muted-foreground">{{ description() }}</p>
    }
    <div class="mt-4 flex justify-center gap-2 empty:hidden"><ng-content /></div>
  `,
})
export class EmptyState {
  readonly heading = input.required<string>();
  readonly description = input<string>();
}
