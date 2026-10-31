import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { LOOKBACK, argmax } from './lib/measured';

/** Step through a translation and see which English words each German word really reads. */
@Component({
  selector: 'app-attention-lookback-figure',
  imports: [FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Where the decoder looks"
      hint="Click a German word, or step through. These are the trained model's real weights."
    >
      <p class="text-xs font-semibold">English: all the encoder states stay available</p>
      <div class="mt-1 grid grid-cols-6 gap-1" role="img" [attr.aria-label]="summary()">
        @for (word of data.english; track word; let s = $index) {
          <div class="min-w-0 text-center">
            <div class="relative h-16 overflow-hidden rounded bg-muted">
              <div
                class="absolute inset-x-0 bottom-0 bg-fig-1 transition-[height] motion-reduce:transition-none"
                [style.height.%]="row()[s] * 100"
              ></div>
            </div>
            <p class="mt-0.5 truncate font-mono text-[11px]">{{ word }}</p>
            <p class="text-[10px] tabular-nums text-muted-foreground">{{ percent(row()[s]) }}</p>
          </div>
        }
      </div>

      <p class="mt-3 text-xs font-semibold">German: written one word at a time</p>
      <div
        class="mt-1 grid grid-cols-3 gap-1 sm:grid-cols-6"
        role="group"
        aria-label="German words"
      >
        @for (word of data.german; track word; let t = $index) {
          <button
            type="button"
            class="min-w-0 truncate rounded-md border px-1 py-1 font-mono text-[11px] hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
            [class.bg-fig-3/20]="t === at()"
            [class.border-fig-3]="t === at()"
            [class.opacity-40]="t > at()"
            [attr.aria-pressed]="t === at()"
            (click)="at.set(t)"
          >
            {{ word }}
          </button>
        }
      </div>
      <button
        type="button"
        class="mt-2 rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
        (click)="next()"
      >
        Next word
      </button>

      <p class="mt-3 text-xs font-semibold">All the weights at once</p>
      <div
        class="mt-1 grid gap-0.5"
        [style.grid-template-columns]="'repeat(' + data.english.length + ', minmax(0, 1fr))'"
        role="img"
        aria-label="Attention weights form a diagonal: each German word reads its own English word most."
      >
        @for (cells of data.weights; track $index; let t = $index) {
          @for (w of cells; track $index) {
            <div
              class="h-4 rounded-sm bg-fig-1"
              [class.ring-1]="t === at()"
              [class.ring-fig-3]="t === at()"
              [style.opacity]="0.06 + w * 0.94"
            ></div>
          }
        }
      </div>
      <p caption>
        Each bar is how much of its attention the current German word spends on that English word,
        and the bars always add up to 100%. Stacked up, they make a diagonal, because a shopping
        list keeps its order. No one told the model which word goes with which, and the blurring
        near the diagonal shows it's less certain in the middle of a list. Real languages bend and
        cross this line wherever their word order differs.
      </p>
    </app-figure-frame>
  `,
})
export class AttentionLookbackFigure {
  protected readonly data = LOOKBACK;
  protected readonly at = signal(0);
  protected readonly row = computed(() => this.data.weights[this.at()]);
  protected readonly summary = computed(() => {
    const row = this.row();
    const top = argmax(row);
    return `"${this.data.german[this.at()]}" reads "${this.data.english[top]}" most, at ${this.percent(row[top])}.`;
  });

  protected next(): void {
    this.at.set((this.at() + 1) % this.data.german.length);
  }

  protected percent(value: number): string {
    return `${Math.round(value * 100)}%`;
  }
}
