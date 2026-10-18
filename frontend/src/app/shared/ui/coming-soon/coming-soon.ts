import { ChangeDetectionStrategy, Component, input } from '@angular/core';

/** Placeholder body for pages whose milestone hasn't landed yet. */
@Component({
  selector: 'app-coming-soon',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <section class="mx-auto max-w-6xl px-4 py-12 sm:py-16">
      <h1 class="text-3xl font-semibold tracking-tight">{{ heading() }}</h1>
      <p class="mt-3 max-w-prose text-muted-foreground">{{ description() }}</p>
      <p
        class="mt-6 inline-block rounded-md border border-dashed px-3 py-2 font-mono text-xs text-muted-foreground"
      >
        Coming in {{ milestone() }}
      </p>
    </section>
  `,
})
export class ComingSoon {
  readonly heading = input.required<string>();
  readonly description = input.required<string>();
  readonly milestone = input.required<string>();
}
