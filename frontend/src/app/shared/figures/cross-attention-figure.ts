import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { CROSS_ATTENTION, argmax } from './lib/measured';

/** Click a German word to see which English words it reads, from the trained translator. */
@Component({
  selector: 'app-cross-attention-figure',
  imports: [FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Cross-attention: which English does each German word read?"
      hint="Click a German word. These weights come from the encoder-decoder page's trained model."
    >
      <div class="overflow-x-auto">
        <table
          class="mx-auto border-separate border-spacing-0.5 text-[11px]"
          [attr.aria-label]="summary()"
        >
          <thead>
            <tr>
              <th scope="col"><span class="sr-only">German word</span></th>
              @for (word of data.english; track $index; let s = $index) {
                <th
                  scope="col"
                  class="px-0.5 pb-1 font-mono font-normal"
                  [class.font-bold]="s === topEnglish()"
                  [class.text-fig-1]="s === topEnglish()"
                >
                  {{ word }}
                </th>
              }
            </tr>
          </thead>
          <tbody>
            @for (cells of data.weights; track $index; let t = $index) {
              <tr [class.outline]="t === at()" [class.outline-fig-3]="t === at()">
                <th scope="row" class="pr-1 text-right font-normal">
                  <button
                    type="button"
                    class="rounded px-1 font-mono hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
                    [class.bg-muted]="t === at()"
                    [attr.aria-pressed]="t === at()"
                    (click)="at.set(t)"
                  >
                    {{ data.german[t] }}
                  </button>
                </th>
                @for (w of cells; track $index) {
                  <td class="relative size-7 rounded-sm bg-muted p-0 sm:size-9">
                    <span
                      class="absolute inset-0 rounded-sm bg-fig-1"
                      [style.opacity]="w"
                      aria-hidden="true"
                    ></span>
                    @if (t === at() && w >= 0.1) {
                      <span class="relative block text-center text-[9px] tabular-nums">{{
                        w
                      }}</span>
                    }
                  </td>
                }
              </tr>
            }
          </tbody>
        </table>
      </div>
      <p class="mt-2 rounded-md bg-muted/50 p-2 text-sm" aria-live="polite">{{ summary() }}</p>
      <p caption>
        Each row sums to 1: a German word spreads all its attention across the English. "suppe"
        reads "soup" at 0.96, and "pflaumen" reads "plums" at 0.64. These are real weights, so not
        every row is tidy: "isst" reads "so" most, and "bob" reads "eats" at 0.37 and "bob" at only
        0.16. Across 100 unseen sentences, names and foods read their own word 97% of the time.
        Nobody taught this alignment.
      </p>
    </app-figure-frame>
  `,
})
export class CrossAttentionFigure {
  protected readonly data = CROSS_ATTENTION;
  protected readonly at = signal(2);
  protected readonly row = computed(() => this.data.weights[this.at()]);
  protected readonly topEnglish = computed(() => argmax(this.row()));
  protected readonly summary = computed(() => {
    const top = this.topEnglish();
    return `"${this.data.german[this.at()]}" reads "${this.data.english[top]}" most, with weight ${this.row()[top]}.`;
  });
}
