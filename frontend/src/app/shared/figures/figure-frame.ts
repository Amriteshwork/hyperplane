import { ChangeDetectionStrategy, Component, input } from '@angular/core';

/** The card around every figure: a title, the interactive part, then a caption to read after. */
@Component({
  selector: 'app-figure-frame',
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { class: 'block' },
  template: `
    <figure class="my-2 rounded-lg border bg-card p-3 sm:p-4">
      <p class="text-sm font-semibold">{{ heading() }}</p>
      @if (hint()) {
        <p class="mt-0.5 text-xs text-muted-foreground">{{ hint() }}</p>
      }
      <div class="mt-3"><ng-content /></div>
      <figcaption class="mt-3 text-sm leading-6 text-muted-foreground">
        <ng-content select="[caption]" />
      </figcaption>
    </figure>
  `,
})
export class FigureFrame {
  readonly heading = input.required<string>();
  /** What to do with the controls, e.g. "Drag the inputs". */
  readonly hint = input<string>();
}
