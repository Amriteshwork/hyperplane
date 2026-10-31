import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { FigureSlider } from './figure-slider';
import { attentionWork } from './lib/sampling';

const MAX_DRAWN = 12;
const CELL = 14;

/** A grid of attention scores: which ones writing the next token has to compute. */
@Component({
  selector: 'app-kv-cache-figure',
  imports: [FigureFrame, FigureSlider, DecimalPipe],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="What the next token has to compute"
      hint="Drag the slider and switch the cache on and off."
    >
      <app-figure-slider label="Tokens written so far" [(value)]="tokens" [min]="2" [max]="12" />
      <div class="mt-2 flex gap-1.5" role="group" aria-label="Cache">
        @for (option of [false, true]; track option) {
          <button
            type="button"
            class="rounded-md border px-3 py-1.5 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
            [class.bg-muted]="cached() === option"
            [attr.aria-pressed]="cached() === option"
            (click)="cached.set(option)"
          >
            {{ option ? 'With the KV cache' : 'Rerun everything' }}
          </button>
        }
      </div>

      <svg
        [attr.viewBox]="'0 0 ' + size + ' ' + size"
        class="mx-auto mt-3 w-full max-w-60"
        role="img"
        [attr.aria-label]="
          'Writing token ' + tokens() + ' computes ' + thisStep() + ' attention scores'
        "
      >
        @for (row of rows(); track row) {
          @for (col of rows(); track col) {
            <rect
              [attr.x]="col * cell + 1"
              [attr.y]="row * cell + 1"
              [attr.width]="cell - 2"
              [attr.height]="cell - 2"
              rx="2"
              [class]="active(row) ? 'fill-fig-2' : 'fill-muted'"
            />
          }
        }
      </svg>
      <p class="mt-2 text-center text-sm" aria-live="polite">
        This step computes <span class="font-semibold">{{ thisStep() }}</span> scores.
      </p>

      <dl class="mt-3 grid grid-cols-3 gap-2 rounded-md bg-muted/50 p-2 text-center text-xs">
        <div>
          <dt class="text-muted-foreground">Length of reply</dt>
          <dd class="font-semibold">256 tokens</dd>
        </div>
        <div>
          <dt class="text-muted-foreground">Rerunning</dt>
          <dd class="font-semibold tabular-nums">{{ total.rerun | number }}</dd>
        </div>
        <div>
          <dt class="text-muted-foreground">Cached</dt>
          <dd class="font-semibold tabular-nums">{{ total.cached | number }}</dd>
        </div>
      </dl>
      <p caption>
        Each row is a token and each square is one attention score. Rerunning recomputes the whole
        grid for every new token. Because of the causal mask, old rows can't change when new tokens
        arrive, so with the cache only the bottom row is new work. Over a 256-token reply, that's
        171 times fewer scores.
      </p>
    </app-figure-frame>
  `,
})
export class KvCacheFigure {
  protected readonly cell = CELL;
  protected readonly size = MAX_DRAWN * CELL;
  protected readonly total = attentionWork(256);
  protected readonly tokens = signal(8);
  protected readonly cached = signal(false);
  protected readonly rows = computed(() => Array.from({ length: this.tokens() }, (_, i) => i));
  protected readonly thisStep = computed(() =>
    this.cached() ? this.tokens() : this.tokens() ** 2,
  );

  protected active(row: number): boolean {
    return !this.cached() || row === this.tokens() - 1;
  }
}
