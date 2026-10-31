import { ChangeDetectionStrategy, Component, computed, signal } from '@angular/core';

import { FigureFrame } from './figure-frame';
import { TOKEN_ROWS, TOKEN_SENTENCES } from './lib/word2vec';

const BY_ID = new Map(TOKEN_ROWS.map((t) => [t.id, t]));

/** A sentence through GPT-2's real tokenizer, into real rows of its embedding table. */
@Component({
  selector: 'app-token-to-vector-figure',
  imports: [FigureFrame],
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `
    <app-figure-frame
      heading="Inside GPT-2: from token to vector"
      hint="Pick a sentence, then click a token."
    >
      <div class="flex flex-wrap gap-1.5" role="group" aria-label="Sentence">
        @for (sentence of sentences; track sentence.text; let i = $index) {
          <button
            type="button"
            class="rounded-md border px-2 py-1 text-xs hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring"
            [class.bg-muted]="at() === i"
            [attr.aria-pressed]="at() === i"
            (click)="choose(i)"
          >
            {{ sentence.text }}
          </button>
        }
      </div>

      <div class="mt-3 flex flex-wrap gap-1.5" role="group" aria-label="Tokens">
        @for (token of tokens(); track $index; let i = $index) {
          <button
            type="button"
            class="rounded-md border px-2 py-1 text-center focus-visible:outline-2 focus-visible:outline-ring"
            [class.border-fig-1]="picked() === i"
            [class.bg-fig-1/15]="picked() === i"
            [attr.aria-pressed]="picked() === i"
            (click)="picked.set(i)"
          >
            <span class="block font-mono text-sm whitespace-pre">{{ token.text }}</span>
            <span class="block text-[10px] text-muted-foreground tabular-nums">{{ token.id }}</span>
          </button>
        }
      </div>

      @if (current(); as token) {
        <p class="mt-3 text-xs text-muted-foreground">
          Row {{ token.id }} of the 50,257 × 768 table: its first 32 numbers.
          <span class="text-fig-1">Blue</span> is positive and
          <span class="text-fig-2">orange</span> is negative.
        </p>
        <svg
          viewBox="0 0 320 22"
          class="mt-1 w-full"
          role="img"
          [attr.aria-label]="'The first 32 of 768 embedding values for token ' + token.text"
        >
          @for (value of token.strip; track $index; let i = $index) {
            <rect
              [attr.x]="i * 10"
              y="1"
              width="9.2"
              height="20"
              rx="1"
              [class]="value >= 0 ? 'fill-fig-1' : 'fill-fig-2'"
              [attr.fill-opacity]="opacity(value)"
            />
          }
        </svg>
        <p class="mt-3 text-xs font-semibold">Its nearest tokens, out of all 50,257</p>
        <ul class="list-none pl-0 mt-1 flex flex-wrap gap-1.5 text-xs" aria-live="polite">
          @for (near of token.near; track near.text) {
            <li class="rounded-md bg-muted px-2 py-1 font-mono whitespace-pre">
              {{ near.text }} <span class="text-muted-foreground">{{ near.cos }}</span>
            </li>
          }
        </ul>
      }
      <p caption>
        These are GPT-2's real tokens and weights. Pick the river sentence, click " bank", and note
        its row. Pick the money sentence: it's the same id, with exactly the same numbers. An
        embedding is only a lookup, so it can't know which bank you meant. Nobody wrote these
        numbers by hand. They were learnt, and tokens that act alike ended up close together. Fixing
        the "bank" problem is the job of self-attention.
      </p>
    </app-figure-frame>
  `,
})
export class TokenToVectorFigure {
  protected readonly sentences = TOKEN_SENTENCES;
  protected readonly at = signal(1);
  protected readonly picked = signal(5);
  protected readonly tokens = computed(() =>
    this.sentences[this.at()].ids.map((id) => BY_ID.get(id)!),
  );
  protected readonly current = computed(() => this.tokens()[this.picked()] ?? this.tokens()[0]);

  protected choose(i: number): void {
    this.at.set(i);
    this.picked.set(this.sentences[i].ids.length - 1);
  }

  /** The table's values are small, so stretch them to read as shades. */
  protected opacity(value: number): number {
    return Math.min(1, 0.12 + Math.abs(value) * 6);
  }
}
